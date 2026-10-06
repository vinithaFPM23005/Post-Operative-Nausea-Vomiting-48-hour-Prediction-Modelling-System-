# Dataset 2 — 1,750 PONV Records (1,500 + 250)

| Item | Value |
|------|-------|
| Files | `data/Data 1500.xlsx` + `data/1501-1750 excel.xls` |
| Combined file | `data/Data_1750.xlsx` (default for `app.py` and `train.py`) |
| Records | 1,750 (1,500 original + 250 new) |
| PONV within 48 h | 335 Yes (19.1%) / 1,415 No |
| New 250 records | 45 Yes (18.0%) / 205 No; mean age 44.6 y, BMI 25.6 |

## Build the combined file
`merge_datasets.py` harmonises column names (`Previous (...)` -> `PreviousPONV`, `Belleville scoring` / `Bellville score` -> `Bellville`, long target name -> `PONV_48h`) and concatenates both sheets.

```bash
pip install -r requirements.txt
python merge_datasets.py "data/Data 1500.xlsx" "data/1501-1750 excel.xls" data/Data_1750.xlsx
python train.py data/Data_1750.xlsx      # default dataset
```

## Retrained results (`train.py`)
80/20 stratified holdout plus 5-fold CV; Bellville score excluded (possible outcome leakage); threshold chosen on training data by Youden's J.

| Model | CV AUC | Holdout AUC | Accuracy | Precision | Recall |
|-------|--------|-------------|----------|-----------|--------|
| Logistic | 0.598 | 0.578 | 0.563 | 0.235 | 0.567 |
| Random Forest | 0.606 | 0.610 | 0.643 | 0.241 | 0.403 |
| Decision Tree | 0.574 | 0.613 | 0.543 | 0.240 | 0.642 |
| Gradient Boosting (selected) | 0.622 | 0.616 | 0.557 | 0.218 | 0.507 |

Discrimination is modest (AUC about 0.6). Outputs are risk probabilities for research use only and need external validation.

See [DATASET_1500.md](DATASET_1500.md) for the original dataset.
