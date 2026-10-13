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

## Optional: LLM-written ticket analysis (Groq)
Detection and the suggested-fix values are rule-based and deterministic. If a [Groq](https://console.groq.com) API key
is available, an LLM also writes the root-cause analysis and ticket summary. Without a key, built-in text is used.

- Local: paste the key in the sidebar (kept in the browser session only), or set `GROQ_API_KEY` in the environment.
- Streamlit Cloud: add `GROQ_API_KEY = "gsk_..."` under the app's **Secrets**. Never commit the key.
- Optional `GROQ_MODEL` overrides the default model (`llama-3.3-70b-versatile`).
- Sent to Groq: issue counts and up to 5 sample bad values per problem, never the full file.

## Files
- `app.py`: Streamlit dashboard (About, Overview, Good data, Bad data, Support ticket)
- `llm.py`: optional Groq call (text only; falls back to built-in text on any error)
- `dq_checks.py`: read-only checks and suggested fixes
- `generate_data.py`: builds the two sample CSVs (500 rows x 20 columns, synthetic data)
- `dq_agent.py`: earlier standalone prototype of the triage loop with column-level lineage and quarantine
