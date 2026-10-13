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
A filter in the sidebar (**Analysis method**) lets the user pick one of two options for the next ticket. There is no way to enter an API key in the app:

| Option | In plain words | When it is used |
|---|---|---|
| 🤖 **Smart AI analysis (Groq)** | An AI reads the problems like a data expert, explains what probably went wrong and suggests the best fix for each one | Default when the app has a key configured |
| ⚙️ **Standard analysis** | Simple built-in checks and fixed advice. No AI. Always works, instantly | Always available, or when the AI is busy |

- If the Smart AI is rate-limited, down or returns a bad reply, the ticket **falls back to Standard automatically** and says why.
- Inside a ticket, **Try Smart AI analysis (Groq) again** / **Switch to Standard analysis** re-runs the analysis (this resets
  the working copy, because the fix choices change).
- If no key is configured, Standard is selected by default and the sidebar says Smart AI is not set up (choosing it anyway still produces a Standard ticket, with a note).

### Setting up Groq (app owner only)
The key is never typed into the app. Set it once:
- **Streamlit Cloud:** add `GROQ_API_KEY = "gsk_..."` under the app's **Secrets** (optionally `GROQ_MODEL`).
- **Local:** put the same lines in `.streamlit/secrets.toml` (git-ignored) or set `GROQ_API_KEY` in the environment.
- Never commit the key. Its free-tier quota is shared by everyone using the app.
- Model: defaults to `llama-3.3-70b-versatile`. If Groq retires it (a 404 "model not found"), the app asks Groq which
  models the account can use and switches to the best available one automatically. `GROQ_MODEL` in Secrets is optional.
- Sent to Groq: issue counts, expected ranges/values and up to 5 sample bad values per problem, never the full file.

## Limitations
- **Reference-dependent checks.** The profile comes from the good file, so anything it does not cover (new valid
  categories, seasonal ranges) can be flagged wrongly or missed.
- **Groq free tier.** Rate limits apply per organisation and vary by model; reported figures for the default model are
  roughly 30 requests/min, 12k tokens/min and 1k requests/day, but these change, so check your Groq console. There is no
  uptime guarantee and models get retired (the app then switches to another available model by itself). The app uses one request per ticket and
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
