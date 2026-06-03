# scripts_clean/

Clean versions of the analysis pipeline. Run all scripts from the **project root** (`tfg-clo-analysis/`), in order.

| Script | Input | Output | Description |
|---|---|---|---|
| `01_extract_text.py` | `data/raw/*.pdf` | `data/processed/*.txt` | Extracts raw text from each CLO Offering Circular using pdfplumber. Skips files already converted. |
| `02_parse_oc.py` | `data/processed/*.txt` | `data/processed/clo_dataset_raw.csv` | Applies regex-based extraction to pull 11 structural variables from each OC (manager, deal size, periods, tranche sizes, etc.). |
| `03_clean_dataset.py` | `clo_dataset_raw.csv` | `data/processed/clo_dataset_clean.csv` | Casts types, derives three new variables (`class_a_pct`, `sub_notes_pct`, `deal_size_log`), and validates the dataset. |
| `04_eda.py` | `clo_dataset_clean.csv` | `outputs/figures/*.png`, `outputs/tables/descriptive_stats.*` | Produces descriptive statistics and six figures (histograms, boxplots by vintage, outlier scatter). |
| `05_analysis.py` | `clo_dataset_clean.csv` | `outputs/figures/*.png`, `outputs/tables/regression_*.csv` | Runs a correlation heatmap and two OLS regressions with VIF diagnostics. |

## How to run

```bash
cd tfg-clo-analysis

python scripts_clean/01_extract_text.py
python scripts_clean/02_parse_oc.py
python scripts_clean/03_clean_dataset.py
python scripts_clean/04_eda.py
python scripts_clean/05_analysis.py
```

Dependencies: `pdfplumber`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `tqdm`.
