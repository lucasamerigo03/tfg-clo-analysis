# CLAUDE.md — TFG: Text-Mining of CLO Offering Circulars

## Project context

This project is a Final Year Dissertation (TFG) for the Business Analytics degree at ICADE (Universidad Pontificia Comillas). The goal is to build a reproducible text-mining pipeline that extracts structural variables from European CLO Offering Circulars (OCs), consolidates them into a structured dataset, and analyses patterns across vintages and managers.

**Source:** Euronext Dublin (Irish Stock Exchange) — public OC filings
**Corpus:** ~15-20 manually downloaded PDFs
**Stack:** Python, pdfplumber, pandas, Claude API (for assisted extraction)

---

## Repository structure

```
tfg-clo-analysis/
├── data/
│   ├── raw/          # Original PDF files (one per CLO deal)
│   └── processed/    # Extracted data in CSV format
├── scripts/
│   ├── parse_oc.py   # Main extraction script
│   └── utils.py      # Helper functions (text cleaning, regex patterns)
├── notebooks/
│   └── analysis.ipynb  # Exploratory analysis and visualisations
├── output/
│   └── dataset.csv   # Final consolidated dataset (one row per deal)
├── CLAUDE.md         # This file
└── README.md
```

---

## Naming conventions

- Scripts: `snake_case.py`
- Variables and functions: `snake_case`
- CSV columns: `snake_case` (e.g., `oc_ratio_ab`, `reinvestment_period_end`)
- PDF files in `data/raw/`: keep original filename from Euronext Dublin

---

## Target fields

Fields are organised in two tiers based on extraction difficulty.

### Tier 1 — Always present in the Overview section (pages 1–15)

| Field | Column name | Notes |
|---|---|---|
| CLO name | `clo_name` | e.g. "Barings Euro CLO 2023-2" |
| Collateral manager | `manager` | e.g. "Barings (U.K.) Limited" |
| Issue date | `issue_date` | Format: YYYY-MM-DD |
| Vintage (year) | `vintage` | Derived from issue_date |
| Non-call period end | `non_call_end` | Format: YYYY-MM-DD |
| Reinvestment period end | `reinvestment_end` | Format: YYYY-MM-DD |
| Target par amount (€M) | `target_par_eur_m` | Numeric, in millions |
| OC trigger — Class A/B | `oc_trigger_ab` | Percentage as float, e.g. 129.89 |
| OC trigger — Class C | `oc_trigger_c` | |
| OC trigger — Class D | `oc_trigger_d` | |
| OC trigger — Class E | `oc_trigger_e` | |
| IC trigger — Class A/B | `ic_trigger_ab` | |
| IC trigger — Class C | `ic_trigger_c` | |
| IC trigger — Class D | `ic_trigger_d` | |

### Tier 2 — Deeper in the document (Portfolio Profile Tests section)

| Field | Column name | Notes |
|---|---|---|
| CCC limit — Fitch (%) | `ccc_limit_fitch` | e.g. 7.5 |
| CCC limit — S&P (%) | `ccc_limit_sp` | e.g. 7.5 |
| Fixed rate limit (%) | `fixed_rate_limit` | |
| PIK limit (%) | `pik_limit` | |
| Second lien limit (%) | `second_lien_limit` | |
| WAL test (years) | `wal_test` | Weighted average life |

### Tranche table (one row per tranche, separate CSV)

| Field | Column name |
|---|---|
| CLO name | `clo_name` |
| Tranche class | `tranche_class` | e.g. "A", "B-1", "B-2", "C" |
| Notional (€M) | `notional_eur_m` | |
| Spread (bps over EURIBOR) | `spread_bps` | Null if fixed rate |
| Fixed coupon (%) | `fixed_coupon` | Null if floating |
| Issue price (%) | `issue_price` | |
| Rating — S&P | `rating_sp` | |
| Rating — Fitch | `rating_fitch` | |

---

## Extraction approach

OCs are long legal documents (300–450 pages). Key fields are concentrated in the **Overview section** (first ~15 pages). Extraction strategy:

1. Use `pdfplumber` to extract text page by page.
2. Locate the Overview section by searching for the string "OVERVIEW".
3. Apply regex patterns to extract Tier 1 fields from this section.
4. For Tier 2 fields, search for section headers ("Portfolio Profile Tests", "Collateral Quality Tests") and apply targeted regex.
5. Flag fields that could not be extracted automatically for manual review.

Text in OCs is machine-readable (not scanned). Layout is consistent across deals from the same era, but field labels may vary slightly across managers — patterns should be written defensively.

---

## Known document characteristics (based on Barings Euro CLO 2023-2)

- Producer: Aspose.PDF — text extraction works cleanly with pdfplumber
- Page count: ~417 pages (typical range: 300–450)
- OC/IC triggers appear as a table in the Overview under "Coverage Tests"
- Reinvestment Period end date is in a definition block: `"Reinvestment Period" means the period from...`
- Non-Call Period end date appears under the heading "Non-Call Period" in the Overview
- Target Par Amount appears as: `"Target Par Amount" means €[amount]`
- CCC limits appear in a min/max table in the Portfolio Profile Tests section
- Tranche table is on page 1 of the document body

---

## Output files

- `data/processed/deals.csv` — one row per deal, all Tier 1 + Tier 2 fields
- `data/processed/tranches.csv` — one row per tranche
- `output/dataset.csv` — merged and validated final dataset
