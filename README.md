# Post-Operative-Nausea-Vomiting-48-hour-Prediction-Modelling-System-
A comprehensive predictive analysis of Post-Operative Nausea &amp; Vomiting (PONV) within 48 hours of surgery. A predictor analytical tool was developed, trained, and evaluated on 1,750 patient records using five different algorithms: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting 
# PONV Risk Predictor 🏥

A **production-ready Streamlit dashboard** for predicting Post-Operative Nausea & Vomiting (PONV) within 48 hours of surgery using machine learning.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](Dockerfile)

## � Datasets
- [DATASET_1500.md](DATASET_1500.md) — original 1,500 records
- [DATASET_1750.md](DATASET_1750.md) — extended 1,750 records (1,500 + 250), default for training

## �📋 Overview
The PONV Risk Predictor integrates clinical data (ASA score, surgery type, patient factors, anesthesia, drugs) into a multi-model ML pipeline that predicts PONV probability with color-coded risk tiers and actionable clinical alerts.

### Key Features
- ✅ **5 ML Algorithms**: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, XGBoost
- ✅ **Auto-Model Selection**: Trains all models, picks best performer on 5-fold CV AUC
- ✅ **Color-Coded Risk Tiers**: 🟢 LOW / 🟡 MODERATE / 🔴 HIGH / ⛔ VERY HIGH
- ✅ **Plain-Language Explanations**: Clinician-friendly risk factor summaries
- ✅ **Per-ASA Thresholds**: Optimized decision boundaries for each ASA severity grade
- ✅ **Production-Ready**: Docker, docker-compose, nginx reverse proxy support
- ✅ **Deployment-Ready**: GitHub Actions CI/CD, cloud-agnostic
### ⚠️ Disclaimer
**This tool is a PROTOTYPE for research/education only.** It is NOT a validated clinical tool and should NOT be used as the sole basis for clinical decisions. External validation and regulatory approvals required before clinical deployment.
---
## 🚀 Quick Start
### 1. **Installation Setup**
### 1. **Local Setup (VS Code)**

```bash
# Clone or download the repo
cd /path/to/ponv-dashboard

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model on your dataset
python train.py data/Data_1750.xlsx

# Launch the Streamlit app
streamlit run app.py
```
The dashboard will open at **http://localhost:8501**

### 2. **Docker Deployment**
```bash
# Build the image
docker build -t ponv-dashboard .

# Run the container
docker run -p 8501:8501 ponv-dashboard
```
Access at **http://localhost:8501**
---
## 🔧 Model Training

### Training on Your Dataset

```bash
python train.py data/Data_1750.xlsx
```

**Expected output:**
- `ponv_model.pkl` — The trained and calibrated model (best performer)
- `ponv_meta.pkl` — Metadata including surgery types, anaesthesia types, per-ASA thresholds, model comparison results, and disclaimer

### Supported Column Names (Case-Insensitive)

The script auto-detects column names with flexible aliases:

| Field | Aliases |
|-------|---------|
| Age | `age`, `Age` |
| BMI | `BMI`, `bmi` |
| Bellville | `bellville_score`, `Bellville`, `bellville`, `BellvilleScore` |
| ASA | `ASA`, `asa`, `ASA_score`, `asa_score` |
| Surgery Type | `surgery_type`, `surgery`, `SurgeryType` |
| Anaesthesia | `anaesthesia_type`, `anaesthesia`, `anaesthesia_administered` |
| Motion Sickness | `motion_sickness`, `MotionSickness`, `motionSickness` |
| Prior PONV | `prior_ponv`, `priorPONV`, `prior_ponv_history` |
| Prior Surgery | `history_post_op_surgery`, `prior_surgery`, `previous_surgery` |
| Drugs | `glycopyrrolate`, `fentanyl`, `propofol`, `NMBA`, `paracetamol`, `ondansetron`, `local_anaesthetic` |
| Target | `PONV_48h` (required) |

### Model Comparison

The training script automatically:
1. **Trains** all 5 model types (Logistic, RF, DT, GB, XGBoost)
2. **Evaluates** each on held-out test set (20% split)
3. **Selects** the best-performing model (highest cross-validated AUC)
4. **Calibrates** predictions using Sigmoid scaling
5. **Computes** per-ASA optimal thresholds (Youden's J index)
6. **Saves** comparison table in metadata for dashboard display

**Example output (`python train.py`, Data_1750.xlsx):**
```
--- Training GB ---
  CV AUC: 0.622, Holdout AUC: 0.616, thr=0.209, Acc: 0.557, Prec: 0.218, Rec: 0.507

✓ Best model: GB (5-fold CV AUC=0.622)
✓ Saved ponv_model.pkl and ponv_meta.pkl
```
### 3. **Docker Compose (with Nginx)**
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f ponv-app

# Stop services
docker-compose down
```

Access at **http://localhost** (nginx proxy) or **http://localhost:8501** (direct)

---
## 📱 Mobile Access with ngrok (Google Chrome)

1. Install ngrok and add your token (from https://dashboard.ngrok.com): `ngrok config add-authtoken <TOKEN>`
2. Run `./run_ngrok.sh` (starts Streamlit on port 8501 and `ngrok http 8501`).
3. Copy the `https://xxxx.ngrok-free.app` URL shown by ngrok and open it in Chrome on your phone (click "Visit Site" on the ngrok notice page).

---
## 🏥 Using the Dashboard

1. **Enter patient data** in the form (demographics, risk factors, procedure details, drugs)
2. **Click "Predict PONV Risk"**
3. **View results:**
   - **Risk Tier** (color-coded: LOW/MODERATE/HIGH/VERY HIGH)
   - **PONV Probability** (0–100%)
   - **Decision Threshold** (per ASA grade)
   - **Plain-Language Explanation** (why this patient is at risk)
   - **Model Performance** (AUC and comparison table)
---
## 📁 Project Structure 
```
ponv-risk-predictor/
├── src/
│   ├── app.py                 # Streamlit dashboard
│   └── train.py               # Training pipeline
├── models/                    # Trained models (auto-generated)
│   ├── ponv_model.pkl         # Best model artifact
│   └── ponv_meta.pkl          # Metadata & thresholds
├── data/
│   └── raw/                   # Raw datasets
├── .streamlit/
│   └── config.toml            # Streamlit config
├── Dockerfile                 # Container image
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependencies
├── config.json                # Pipeline config & schema
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
└── nginx.conf                 # Reverse proxy config (optional)
```

---
## 📁 Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (input form + predictions) |
| `train_model.py` | Training pipeline; retrains on new data |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image definition |
| `config.json` | Full pipeline configuration & documentation |
| `ponv_model.pkl` | Pre-trained/trained model (auto-generated) |
| `ponv_meta.pkl` | Model metadata + thresholds (auto-generated) |
| `README.md` | This file |

---
## 🧠 Model Features

### Input Variables

| Category | Features |
|----------|----------|
| **Demographics** | Age, BMI |
| **Clinical Scores** | Bellville Score (0–10), ASA Grade (0–3) |
| **Risk Factors** | Motion sickness, Prior PONV, Prior post-op surgery |
| **Procedure** | Surgery type, Anaesthesia type |
| **Drugs** | Glycopyrrolate, Fentanyl, Propofol, NMBA, Paracetamol, Ondansetron, Local anaesthetic |

### Risk Factors
- History of motion sickness (Yes/No)
- Prior PONV history (Yes/No)
- Prior post-operative surgery (Yes/No)

### Procedure Details
- **Surgery type**: General, Orthopaedic, ENT, Gynae, Cardiac (extensible)
- **Anaesthesia type**: GA (General Anaesthesia), Regional, MAC (Monitored Anaesthesia Care)

### Intra-operative Drugs
- Glycopyrrolate (anticholinergic; may reduce PONV)
- Fentanyl (opioid)
- Propofol (IV anaesthetic; anti-emetic properties)
- NMBA (neuromuscular blocker)
- Paracetamol
- Ondansetron (5-HT3 antagonist; prophylactic anti-emetic)
- Local anaesthetic

### Target
- **PONV_48h**: Binary (0 = No PONV, 1 = PONV within 48h)
### Output

- **PONV Probability**: 0–100% (48-hour window)
- **Risk Tier**: LOW / MODERATE / HIGH / VERY HIGH
- **Plain-Language Explanation**: Key risk factors
- **Decision Threshold**: Per-ASA optimized (Youden's J)

---
## 📊 Model Performance
Trained on the provided dataset (n=1,750; 80/20 stratified holdout + 5-fold CV; Bellville score excluded to avoid outcome leakage; Youden threshold from training data):
| Model | CV AUC | Holdout AUC | Accuracy | Precision | Recall |
|-------|--------|-------------|----------|-----------|--------|
| Logistic Regression | 0.598 | 0.578 | 0.563 | 0.235 | 0.567 |
| Random Forest | 0.606 | 0.610 | 0.643 | 0.241 | 0.403 |
| Decision Tree | 0.574 | 0.613 | 0.543 | 0.240 | 0.642 |
| **Gradient Boosting** ⭐ | **0.622** | 0.616 | 0.557 | 0.218 | 0.507 |

**Note**: Discrimination is modest (AUC ~0.6); outputs are probabilities for research use, not diagnoses.
---
## 🧮 Models Supported

| Model | Type | Key Features |
|-------|------|--------------|
| **Logistic Regression** | Linear | Interpretable, fast, well-calibrated |
| **Random Forest** | Ensemble | Handles non-linearity, feature importances |
| **Decision Tree** | Tree-based | Highly interpretable, fast |
| **Gradient Boosting** | Ensemble | Strong performance, interactions |
| **XGBoost** | Boosted tree | State-of-the-art, imbalance handling |

**Selection Strategy:** Auto-select on validation AUC; show all results in dashboard.
---
## 🎨 Risk Tier System

| Tier | Probability | Color | Action |
|------|-------------|-------|--------|
| **LOW** | 0–15% | 🟢 Green | Routine post-op care |
| **MODERATE** | 15–30% | 🟡 Yellow | Monitor; consider anti-emetic prep |
| **HIGH** | 30–50% | 🔴 Orange | Prophylactic anti-emetics recommended |
| **VERY HIGH** | 50–100% | ⛔ Red | Aggressive prophylaxis & monitoring |

---
## 📈 Performance

**Current Dataset (Data_1750.xlsx):**
- **Held-out AUC:** ~0.58–0.62 (varies by model)
- **PONV Prevalence:** ~19% (realistic imbalance)
- **Sample Size:** 1,750 patients
- **Test Set:** 20% (350 patients)

**Recommendations:**
- Collect more data (>5,000 patients) for production robustness
- Perform external validation on independent cohorts
- Fine-tune hyperparameters with Optuna or grid search
- Implement SHAP for feature explainability
- Add audit logging for compliance

---
## 🚀 Advanced Usage

### Retrain Periodically
```bash
# After collecting new cases
python train_model.py path/to/updated_data.xlsx
```
## 🔧 Retraining on Your Data

```bash
# Place your Excel file in data/raw/
cp /path/to/your_data.xlsx data/Data_1750.xlsx

# Retrain
python train.py data/Data_1750.xlsx

# Expected output:
# ✓ Best model: LOGISTIC (AUC=0.XXX)
# ✓ Saved models/ponv_model.pkl and models/ponv_meta.pkl
```

The script auto-detects columns (case-insensitive):
- `age`, `Age` → Auto-found ✓
- `ASA`, `asa_score` → Auto-found ✓
- `surgery_type`, `SurgeryType` → Auto-found ✓
---
### Modify Thresholds
Edit `app.py` to adjust risk tier boundaries:
```python
def get_risk_tier(prob, threshold=0.30):  # Change threshold here
    ...
```
### Add Custom Models
Extend `build_pipeline()` in `train_model.py` to include other algorithms (LightGBM, CatBoost, etc.)

### Deploy with Authentication
Use Streamlit secrets and authentication middleware (e.g., OAuth, LDAP)

---
## ☁️ Cloud Deployment
### AWS EC2
```bash
# SSH into instance
ssh -i key.pem ec2-user@your-instance-ip

# Clone repo and install
git clone https://github.com/yourusername/ponv-risk-predictor.git
cd ponv-risk-predictor
docker-compose up -d
```

### Google Cloud Run
```bash
# Build and push to GCR
docker build -t gcr.io/YOUR_PROJECT/ponv-dashboard .
docker push gcr.io/YOUR_PROJECT/ponv-dashboard

# Deploy
gcloud run deploy ponv-dashboard \
  --image gcr.io/YOUR_PROJECT/ponv-dashboard \
  --platform managed \
  --port 8501
```
### Heroku (via Procfile)
```
web: streamlit run src/app.py --server.port=$PORT
```

---
## 🔐 Security ,Configuration Compliance : 
Compliance : 
- All data stays local (no external API calls)
- Model predictions are for decision support only
- Ensure HIPAA/GDPR compliance when processing real patient data
- Use VPN or firewall for secure deployment
- Implement user authentication (optional Streamlit plugins)

---
### Environment Variables
Create `.env` file:
```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_LOGGER_LEVEL=info
STREAMLIT_CLIENT_SHOWSTDERR=false
```

### Nginx Reverse Proxy
See `nginx.conf` for HTTPS, rate limiting, compression.

### User Authentication (Optional)
```python
# Add to app.py for basic auth
import streamlit_authenticator as stauth
```

---

## 📚 Documentation

- **README.md** — This file
- **[config.json](config.json)** — Full pipeline spec (features, models, thresholds)
- **[Dockerfile](Dockerfile)** — Container image
- **[docker-compose.yml](docker-compose.yml)** — Multi-container setup
- **[LICENSE](LICENSE)** — MIT License + Clinical Disclaimer

---

## 🧪 Testing

```bash
# Unit tests (optional)
pytest tests/

# Manual smoke test
python train.py data/Data_1750.xlsx
streamlit run src/app.py --logger.level=debug
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

Please include:
- Clear commit messages
- Unit tests for new features
- Updated documentation

---

## 📝 Citation

If you use this tool in research, please cite:

```bibtex
@software{ponv_predictor_2026,
  title={PONV Risk Predictor: A Machine Learning Dashboard for Post-Operative Nausea & Vomiting},
  author={Clinical Decision Support Team},
  year={2026},
  url={https://github.com/yourusername/ponv-risk-predictor}
}
```

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

---

## ⚠️ Clinical Disclaimer

This software is provided **AS-IS** for research and educational purposes only. It is **NOT**:
- A validated clinical tool
- FDA-approved or CE-marked
- A replacement for clinical judgment
- Suitable for regulatory/compliance use without external validation

**Before clinical deployment:**
1. Perform external validation on independent cohorts
2. Obtain regulatory approvals (FDA, CE, etc.)
3. Implement audit logging and compliance tracking
4. Ensure HIPAA/GDPR compliance
5. Train clinical staff on correct usage

---
## 💬 Support & Issues

- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/ponv-risk-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ponv-risk-predictor/discussions)
- **Email**: support@yourorganization.com

---

## 📚 References

- **ASA Classification**: [American Society of Anesthesiologists](https://www.asahq.org/)
- **PONV Prediction**: Apfel et al., Anesthesiology 1999 (Bellville score & risk factors)
- **Calibration**: Niculescu-Mizil & Caruana, "Obtaining Calibrated Probabilities from Boosting"
- **XGBoost**: Chen & Guestrin, KDD 2016

---
## 📞 Support & Contributing
For issues, bugs, or feature requests, please provide:
1. Steps to reproduce
2. Dataset characteristics (sample size, PONV rate)
3. Model performance metrics
4. Expected vs. actual behavior

---
## 🙏 Acknowledgments
- Streamlit for the web framework
- scikit-learn, XGBoost communities for ML tools
- Clinical advisors for feedback
---
**Last Updated:** August 2026  
**Version:** 1.0.0  
**Status:** Production-Ready (Research/Education; External validation required for clinical use)



