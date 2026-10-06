# Dataset 1 — 1,500 PONV Records

| Item | Value |
|------|-------|
| File | `data/Data 1500.xlsx` |
| Records | 1,500 (IDs 1–1500) |
| PONV within 48 h | 290 Yes (19.3%) / 1,210 No |
| Mean age / BMI | 42.8 y / 25.6 |

## Columns
`ID`, `Age`, `Gender`, `BMI`, `ASA`, `Surgery`, `Anaesthesia`, `MotionSickness`,
`Previous (Post Operative Nausea and Vomitting)`, `Glycopyrrolate`, `Fentanyl`, `Propofol`, `NMBA`,
`Paracetamol`, `Ondansetron`, `LocalAnaesthetic`, `Post Operative Nausea Vomitting_48h` (target), `Bellville score`.

## Train on this dataset only
```bash
python train.py "data/Data 1500.xlsx"
```

See [DATASET_1750.md](DATASET_1750.md) for the extended dataset used by default.
