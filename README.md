# Data Quality Triage (POC)

An AI data-quality agent demo. It learns what "normal" looks like from a good CSV, checks a bad CSV against it,
flags the exact broken cells (nulls, blanks, wrong types, anomalies), and drafts a support ticket in the data owner's
name with a suggested fix per problem. It never changes data on its own: a person clicks to apply each fix, and only an
in-session copy changes.

## Run

```bash
pip install -r requirements.txt
python generate_data.py      # optional: recreates data/good_data.csv and data/bad_data.csv (synthetic)
streamlit run app.py
```

## Files
- `app.py`: Streamlit dashboard (About, Overview, Good data, Bad data, Support ticket)
- `dq_checks.py`: read-only checks and suggested fixes
- `generate_data.py`: builds the two sample CSVs (500 rows x 20 columns, synthetic data)
- `dq_agent.py`: earlier standalone prototype of the triage loop with column-level lineage and quarantine
