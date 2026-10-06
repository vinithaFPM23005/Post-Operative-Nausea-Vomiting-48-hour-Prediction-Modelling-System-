"""Merge the 1,500-record and 250-record PONV datasets into one 1,750-record file."""
import sys
from pathlib import Path

import pandas as pd

RENAME = {
    "Previous (Post Operative Nausea and Vomitting)": "PreviousPONV",
    "Post Operative Nausea Vomitting_48h": "PONV_48h",
    "Bellville score": "Bellville",
    "Belleville scoring": "Bellville",
}


def load(path):
    df = pd.read_excel(path)
    df.columns = [RENAME.get(str(c).strip(), str(c).strip()) for c in df.columns]
    return df


def main(first="data/Data 1500.xlsx", second="data/1501-1750 excel.xls",
         out="data/Data_1750.xlsx"):
    a, b = load(first), load(second)
    combined = pd.concat([a, b], ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(out, index=False)
    print(f"{len(a)} + {len(b)} = {len(combined)} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
