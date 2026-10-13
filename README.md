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

## How the "AI" works
Two layers, so every number stays checkable:

1. **Detection (always on, rule-based, deterministic).** The good file is profiled (type, range, allowed values) and the
   bad file is checked against it: nulls, blanks, wrong types, out-of-range values, unexpected categories, bad formats,
   duplicate keys and cross-field rules. For each problem the rule engine can compute exact candidate fixes.
2. **Analysis and fix selection (optional, Groq LLM).** With a [Groq](https://console.groq.com) API key, one request per
   ticket asks the model to explain the likely cause of each problem, **choose the best fix** from the candidates the rule
   engine offers (or recommend manual review when filling would fabricate data, e.g. emails), and write the ticket summary.
   The model cannot invent values: an unknown choice falls back to the rule-based fix. Without a key (or on any error),
   built-in rules and text are used.

Nothing is changed until a person clicks **Apply suggested fix**, and then only an in-session copy changes.
Each ticket shows who wrote the analysis (`Groq (<model>)` or `Rule-based templates`).

### Choosing the analysis method
When raising a ticket, the user picks one of two methods:

| Method | What it does | When to use |
|---|---|---|
| 🤖 **Groq AI** | LLM explains causes, picks the best fix per problem, writes the summary | Default when a key is configured |
| ⚙️ **Standard** | Built-in rules and text, no AI, no external calls | Always available; use when Groq is rate-limited or down |

- If Groq fails (rate limit, outage, bad reply), the ticket **falls back to Standard automatically** and says why.
- From inside a ticket, **Retry with Groq AI** / **Switch to Standard analysis** re-runs the analysis (this resets the
  working copy, because the fix choices change).
- With no key configured, only Standard is offered.

### Setting up Groq
- Streamlit Cloud: add `GROQ_API_KEY = "gsk_..."` under the app's **Secrets** (optionally `GROQ_MODEL`). Never commit the key.
  The key's free-tier quota is shared by everyone using the app.
- Local: put the same lines in `.streamlit/secrets.toml` (git-ignored), set `GROQ_API_KEY` in the environment, or paste a
  key in the sidebar under *Use my own Groq key or model* (kept in the browser session only).
- Default model is `llama-3.3-70b-versatile`; change it via `GROQ_MODEL` or the sidebar if Groq retires it.
- Sent to Groq: issue counts, expected ranges/values and up to 5 sample bad values per problem, never the full file.

## Limitations
- **Reference-dependent checks.** The profile comes from the good file, so anything it does not cover (new valid
  categories, seasonal ranges) can be flagged wrongly or missed.
- **Groq free tier.** Rate limits apply per organisation and vary by model; reported figures for the default model are
  roughly 30 requests/min, 12k tokens/min and 1k requests/day, but these change, so check your Groq console. There is no
  uptime guarantee and models get retired (a 404 means: change the model name). The app uses one request per ticket and
  falls back to the Standard method if the limit is hit or Groq is unreachable, and users can also choose Standard up front.
- **LLM output can be wrong or vary between runs.** It only chooses among computable fixes or recommends manual review.
- **Privacy.** With a key set, the summary above goes to a third party. This demo uses synthetic data only.
- **Demo scope.** Mock tickets (nothing goes to a real ticketing system), one pair of files at a time, and state is kept
  per browser session.

## Files
- `app.py`: Streamlit dashboard (About, Overview, Good data, Bad data, Support ticket)
- `llm.py`: optional Groq call (text only; falls back to built-in text on any error)
- `dq_checks.py`: read-only checks and suggested fixes
- `generate_data.py`: builds the two sample CSVs (500 rows x 20 columns, synthetic data)
- `dq_agent.py`: earlier standalone prototype of the triage loop with column-level lineage and quarantine
