import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "ponv_model.pkl"))
META_PATH = Path(os.getenv("META_PATH", ROOT / "models" / "ponv_meta.pkl"))
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
TOKEN_MINUTES = int(os.getenv("TOKEN_MINUTES", "30"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'backend' / 'ponv_cases.db'}")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET must be set before starting the backend")

model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
features = meta["features"]
security = HTTPBearer()
app = FastAPI(title="PONV Risk API", version="1.0.0")

origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:8888").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class PatientInput(BaseModel):
    age: float = Field(ge=0, le=120)
    bmi: float = Field(ge=10, le=80)
    bellville_score: int | None = Field(default=None, ge=0, le=3)
    surgery: str = Field(min_length=1, max_length=120)
    anaesthesia: str = Field(min_length=1, max_length=120)
    asa: int = Field(ge=0, le=3)
    glycopyrrolate: str = "No"
    fentanyl: str = "No"
    propofol: str = "No"
    nmba: str = "No"
    paracetamol: str = "No"
    ondansetron: str = "No"
    local_anaesthetic: str = "No"
    motion_sickness: str = "No"


class CaseNote(BaseModel):
    case_id: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=60)
    note: str = Field(min_length=1, max_length=5000)
    diet_status: str = Field(default="Not recorded", max_length=100)
    diet_notes: str = Field(default="", max_length=2000)


def db_connection():
    if not DATABASE_URL.startswith("sqlite:///"):
        raise RuntimeError("Set DATABASE_URL to sqlite:///... for local use or add a PostgreSQL adapter for production")
    return sqlite3.connect(DATABASE_URL.removeprefix("sqlite:///"))


def initialize_db() -> None:
    with db_connection() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY, username TEXT NOT NULL, action TEXT NOT NULL, case_id TEXT, details TEXT NOT NULL, created_at TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS case_notes (id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, username TEXT NOT NULL, role TEXT NOT NULL, note TEXT NOT NULL, diet_status TEXT NOT NULL, diet_notes TEXT NOT NULL, created_at TEXT NOT NULL)")


def verify_password(password: str, encoded: str) -> bool:
    try:
        salt, expected = encoded.split("$", 1)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 250_000).hex()
        return hmac.compare_digest(actual, expected)
    except ValueError:
        return False


def users() -> dict[str, str]:
    try:
        return json.loads(os.environ.get("APP_USERS", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("APP_USERS must be valid JSON mapping usernames to PBKDF2 password hashes") from exc


def current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username or username not in users():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication")
        return username
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc


def audit(username: str, action: str, case_id: str | None, details: dict[str, Any]) -> None:
    with db_connection() as connection:
        connection.execute("INSERT INTO audit_log (username, action, case_id, details, created_at) VALUES (?, ?, ?, ?, ?)", (username, action, case_id, json.dumps(details), datetime.now(timezone.utc).isoformat()))


def prediction_row(patient: PatientInput) -> dict[str, Any]:
    values = patient.model_dump()
    return {
        "Age": values["age"],
        "BMI": values["bmi"],
        "Bellville score": values["bellville_score"],
        "Surgery": values["surgery"],
        "Anaesthesia": values["anaesthesia"],
        "ASA": values["asa"],
        "Glycopyrrolate": values["glycopyrrolate"],
        "Fentanyl": values["fentanyl"],
        "Propofol": values["propofol"],
        "NMBA": values["nmba"],
        "Paracetamol": values["paracetamol"],
        "Ondansetron": values["ondansetron"],
        "LocalAnaesthetic": values["local_anaesthetic"],
        "MotionSickness": values["motion_sickness"],
    }


@app.on_event("startup")
def startup() -> None:
    initialize_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ponv-risk-api"}


@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict[str, str]:
    encoded = users().get(request.username)
    if not encoded or not verify_password(request.password, encoded):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    expires = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)
    token = jwt.encode({"sub": request.username, "exp": expires}, JWT_SECRET, algorithm=JWT_ALGORITHM)
    audit(request.username, "login", None, {})
    return {"access_token": token, "token_type": "bearer", "expires_at": expires.isoformat()}


@app.post("/api/predict")
def predict(patient: PatientInput, request: Request, username: str = Depends(current_user)) -> dict[str, Any]:
    row = prediction_row(patient)
    probability = float(model.predict_proba(pd.DataFrame([row])[features])[:, 1][0])
    threshold = float(meta.get("thresholds_by_asa", {}).get(str(patient.asa), meta.get("thresholds_by_asa", {}).get(patient.asa, 0.30)))
    risk = "LOW" if probability < 0.15 else "MODERATE" if probability < threshold else "HIGH" if probability < 0.50 else "VERY HIGH"
    result = {"probability": probability, "alert": probability >= threshold, "risk_tier": risk, "threshold": threshold, "bellville_alert": patient.bellville_score is not None and patient.bellville_score >= 2, "model": meta.get("best_model_type", "unknown")}
    audit(username, "prediction", None, {"risk_tier": risk, "probability": round(probability, 6), "client": request.client.host if request.client else None})
    return result


@app.post("/api/cases/notes")
def save_note(note: CaseNote, username: str = Depends(current_user)) -> dict[str, str]:
    now = datetime.now(timezone.utc).isoformat()
    with db_connection() as connection:
        connection.execute("INSERT INTO case_notes (case_id, username, role, note, diet_status, diet_notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (note.case_id, username, note.role, note.note, note.diet_status, note.diet_notes, now))
    audit(username, "case_note_created", note.case_id, {"role": note.role})
    return {"status": "saved", "created_at": now}


@app.get("/api/cases/notes")
def list_notes(case_id: str = "", username: str = Depends(current_user)) -> list[dict[str, Any]]:
    query = "SELECT case_id, username, role, note, diet_status, diet_notes, created_at FROM case_notes"
    params: tuple[str, ...] = ()
    if case_id.strip():
        query += " WHERE case_id = ?"
        params = (case_id.strip(),)
    query += " ORDER BY created_at DESC LIMIT 200"
    with db_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    audit(username, "case_notes_viewed", case_id or None, {})
    return [dict(zip(("case_id", "username", "role", "note", "diet_status", "diet_notes", "created_at"), row)) for row in rows]
