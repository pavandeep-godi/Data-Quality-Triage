"""End-to-end test run for the Data Quality Triage app.  Run from anywhere:  python tests/run_tests.py
Covers: data files, check engine, fix accuracy, every page, ticket flow, Groq/Standard modes (mocked HTTP),
Secrets layouts, awkward inputs, security, and repo hygiene. Never calls the real Groq API."""
import hashlib, json, os, pathlib, re, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import pandas as pd
from streamlit.testing.v1 import AppTest
import dq_checks as dq
import generate_data as gd
import llm

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))

def section(t): print(f"\n=== {t} ===")
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

DATA_HASH = {f: sha(ROOT / "data" / f) for f in ("good_data.csv", "bad_data.csv")}
APP = str(ROOT / "app.py")

def new_app(path=APP, key=None, secrets=None, nav=None, workdir=None):
    os.environ.pop("GROQ_API_KEY", None)
    if key: os.environ["GROQ_API_KEY"] = key
    at = AppTest.from_file(path, default_timeout=90).run()
    if nav:
        at.session_state["nav"] = nav; at.run()
    return at

def errs(at): return [e.value for e in at.exception]
PAGES = ["ℹ️ About the demo", "📊 Overview", "✅ Good data", "❌ Bad data", "🎫 Support ticket"]

class R:  # fake requests.Response
    def __init__(self, code, body=None, text=None):
        self.status_code, self._b = code, body
        self.text = text if text is not None else json.dumps(body)
    def json(self): return self._b

def groq_ok(choice="rule", model_seen=None, only=None):
    def post(url, **kw):
        if model_seen is not None: model_seen.append(kw["json"]["model"])
        items = json.loads(kw["json"]["messages"][1]["content"])
        f = [{"column": i["column"], "check": i["check"], "likely_cause": "mock cause.", "next_step": "mock step.",
              "recommended_option": choice if (only is None or i["column"] in only) and choice in [o["id"] for o in i["options"]] else "rule",
              "rationale": "mock why"} for i in items]
        return R(200, {"choices": [{"message": {"content": json.dumps({"summary": "Mock summary.", "findings": f})}}]})
    return post

def raise_ticket(at, owner="Priya Sharma"):
    [t for t in at.text_input if t.key == "owner"][0].set_value(owner).run()
    [b for b in at.button if b.label.startswith("Yes")][0].click(); at.run()
    return at.session_state.ticket

# ======================================================================================
section("1. Data files")
good = dq.load(ROOT / "data/good_data.csv"); bad = dq.load(ROOT / "data/bad_data.csv")
check("both files are 500 rows x 20 columns", good.shape == (500, 20) and bad.shape == (500, 20), f"{good.shape} {bad.shape}")
check("same 20 column names in both", list(good.columns) == list(bad.columns))
check("good file has no nulls / blanks", good.isna().sum().sum() == 0 and not (good.apply(lambda s: s.str.strip() == "")).any().any())
nulls = bad.isna().sum(); nulls = nulls[nulls > 0].to_dict()
check("bad file: nulls only in 3 columns (email, city, rating)", set(nulls) == {"email", "city", "rating"}, str(nulls))
check("regenerating data is deterministic",
      gd.make_good(42).astype(str).equals(pd.read_csv(ROOT / "data/good_data.csv", dtype=str)))

section("2. Check engine on the shipped files")
prof = dq.learn_reference(good)
gi, bi = dq.run_checks(good, prof), dq.run_checks(bad, prof)
check("good file: 0 issues", len(gi) == 0, str(len(gi)))
counts = bi.groupby(["column", "check"]).size().to_dict()
expect = {("email", "null"): 45, ("city", "null"): 25, ("city", "empty"): 15, ("rating", "null"): 50,
          ("unit_price", "anomaly"): 12, ("quantity", "type"): 15}
check("bad file: exactly the planted problems, nothing extra", counts == expect, str(counts))
check("every issue row index is valid and unique per (row,column,check)",
      bi["row"].between(0, 499).all() and not bi.duplicated(["row", "column", "check"]).any())

section("3. Detector coverage on a mutated copy of the good file")
m = good.copy()
m.loc[3, "status"] = "Deliverd"                        # typo of a valid category
m.loc[5, "order_id"] = m.loc[4, "order_id"]            # duplicate key
m.loc[7, "email"] = "not-an-email"                     # bad format
m.loc[9, "ship_date"] = "2024-01-01"                   # ship before order
m.loc[11, "total_amount"] = "1.00"                     # total mismatch, price untouched
m.loc[13, "order_date"] = "13/45/2025"                 # unparseable date
m.loc[15, "city"] = "   "                              # whitespace only
m.loc[17, "discount_pct"] = None                       # null in numeric
m.loc[19, "is_member"] = "Maybe"                       # bool column, bad value
mi = dq.run_checks(m, prof)
got = {(r.column, r.check, r.row) for r in mi.itertuples()}
for (col, chk, row), label in {("status", "category", 3): "category typo", ("order_id", "duplicate", 5): "duplicate key",
                               ("email", "format", 7): "invalid email", ("ship_date", "rule", 9): "ship before order",
                               ("total_amount", "rule", 11): "total mismatch", ("order_date", "type", 13): "invalid date",
                               ("city", "empty", 15): "whitespace-only", ("discount_pct", "null", 17): "numeric null",
                               ("is_member", "category", 19): "bad boolean"}.items():
    check(f"detects {label}", (col, chk, row) in got)
fix = mi[(mi.column == "status") & (mi.row == 3)].iloc[0]
check("typo 'Deliverd' is mapped to the closest valid value 'Delivered'", fix.fix_value == "Delivered", str(fix.fix_value))
check("no false positives outside the planted rows", len({(c, r) for c, _, r in got} - {("status", 3), ("order_id", 5),
      ("email", 7), ("ship_date", 9), ("total_amount", 11), ("order_date", 13), ("city", 15), ("discount_pct", 17),
      ("is_member", 19)}) == 0, str(sorted(got)))

section("4. Fix accuracy against the hidden ground truth")
truth = gd.make_good(7).astype(str)  # bad file = make_good(7) + corruption
fixed = bad.copy()
for r in bi.dropna(subset=["fix_value"]).itertuples(): fixed.at[r.row, r.column] = r.fix_value
q_rows, p_rows = bi[bi.column == "quantity"].row, bi[bi.column == "unit_price"].row
q_acc = (fixed.loc[q_rows, "quantity"].astype(float) == truth.loc[q_rows, "quantity"].astype(float)).mean()
p_acc = ((fixed.loc[p_rows, "unit_price"].astype(float) - truth.loc[p_rows, "unit_price"].astype(float)).abs() < 0.011).mean()
check(f"quantity fixes recover the true value ({q_acc:.0%})", q_acc >= 0.9, f"{q_acc:.2f}")
check(f"unit_price fixes recover the true value ({p_acc:.0%})", p_acc >= 0.9, f"{p_acc:.2f}")
c_rows = bi[(bi.column == "city")].row
c_acc = (fixed.loc[c_rows, "city"] == truth.loc[c_rows, "city"]).mean()
print(f"INFO city fill matches the true city in {c_acc:.0%} of rows (a statistical guess by country, labelled as a suggestion)")
check("after applying all rule fixes the file re-validates clean", len(dq.run_checks(fixed, prof)) == 0)
check("original CSVs on disk untouched so far", all(sha(ROOT / "data" / f) == h for f, h in DATA_HASH.items()))

# ======================================================================================
section("5. Every page renders")
at = new_app()
check("app starts on About with no exception", at.session_state.nav == PAGES[0] and not errs(at), str(errs(at)))
check("About has limitations + example + start button",
      any("Limitations" in h.value for h in at.markdown) and any(b.label.startswith("Start") for b in at.button))
for p in PAGES:
    at = new_app(nav=p)
    check(f"page renders: {p}", not errs(at), str(errs(at)))

section("6. Navigation links")
at = new_app(); [b for b in at.button if b.label.startswith("Start")][0].click(); at.run()
check("About 'Start the demo' -> Overview", at.session_state.nav == PAGES[1])
[b for b in at.button if "good data" in b.label][0].click(); at.run(); check("Overview -> Good data", at.session_state.nav == PAGES[2])
at = new_app(nav=PAGES[1]); [b for b in at.button if "where it breaks" in b.label][0].click(); at.run()
check("Overview -> Bad data", at.session_state.nav == PAGES[3] and not errs(at))
at = new_app(nav=PAGES[4]); check("Ticket page with no ticket shows only a back button (no 'nothing raised' text)",
      [b.label for b in at.button] == ["← Back to Bad data"] and not any("Nothing has been raised" in i.value for i in at.info))

section("7. Overview content")
at = new_app(nav=PAGES[1])
html = " ".join(m.value for m in at.markdown)
check("Overview shows PASS for good and FAIL for bad", "PASS" in html and "FAIL" in html)
check("Overview shows 162 issues and 71% clean ring", "162" in html and "71%" in html)

section("8. Bad data tab: views, filter, toggle")
at = new_app(nav=PAGES[3])
radio = [r for r in at.radio if r.options and "Affected columns" in r.options][0]
for opt in radio.options:
    radio.set_value(opt).run(); check(f"chart view '{opt}' renders", not errs(at), str(errs(at)))
ms = at.multiselect[0]; ms.set_value(["city"]).run()
check("column filter works without error", not errs(at), str(errs(at)))
at.toggle[0].set_value(False).run(); check("'only rows with issues' off renders full table", not errs(at), str(errs(at)))

# ======================================================================================
section("9. Ticket flow (Standard method)")
at = new_app(nav=PAGES[3])
yes = [b for b in at.button if b.label.startswith("Yes")][0]
check("raise-ticket button disabled until an owner is typed", yes.disabled)
[t for t in at.text_input if t.key == "owner"][0].set_value("   ").run()
check("whitespace-only owner keeps it disabled", [b for b in at.button if b.label.startswith("Yes")][0].disabled)
t = raise_ticket(at)
check("ticket created, nav -> Support ticket, lands on list view",
      at.session_state.nav == PAGES[4] and not at.session_state.ticket_open and not errs(at), str(errs(at)))
check("ticket fields: owner, created-by AI Agent, status Open, SLA 2 days",
      t["owner"] == "Priya Sharma" and any("AI Agent" in m.value for m in at.markdown)
      and any("Open" in m.value for m in at.markdown) and any("2 days" in m.value for m in at.markdown))
check("engine is Standard", t["engine"].startswith("Standard"))
[b for b in at.button if b.label == t["id"]][0].click(); at.run()
check("clicking the ticket ID opens the detail view", at.session_state.ticket_open and not errs(at), str(errs(at)))
fixes = [b for b in at.button if b.label == "Apply suggested fix"]
check("6 fix cards, all enabled", len(fixes) == 6 and not any(b.disabled for b in fixes), str(len(fixes)))
fixes[0].click(); at.run()
check("single fix -> status In Progress", any("In Progress" in m.value for m in at.markdown if m.value.startswith("###")) and not errs(at))
before = at.session_state["work"].copy()
[b for b in at.button if b.label == "Undo"][0].click(); at.run()
check("undo restores the original values", at.session_state["work"].equals(bad), "")
[b for b in at.button if b.label == "Apply ALL suggested fixes"][0].click(); at.run()
check("apply ALL -> Resolved, 0 remaining", any("Resolved" in m.value for m in at.markdown if m.value.startswith("###")) and
      len(dq.run_checks(at.session_state["work"], prof)) == 0)
check("download button present", len(at.get("download_button")) == 1)
[b for b in at.button if b.label == "Reset working copy"][0].click(); at.run()
check("reset restores the original data", at.session_state["work"].equals(bad))
check("source CSVs on disk still untouched", all(sha(ROOT / "data" / f) == h for f, h in DATA_HASH.items()))
at.session_state["nav"] = PAGES[3]; at.run()
check("Bad data tab shows 'ticket raised' + link after a ticket exists", any("was raised in the name of" in s.value for s in at.success))

# ======================================================================================
section("10. Groq vs Standard (HTTP mocked)")
llm._models_cache.clear()
calls = {"n": 0}
def counting(inner):
    def f(url, **kw): calls["n"] += 1; return inner(url, **kw)
    return f
at = new_app(key="gsk_test", nav=PAGES[3]); r = at.sidebar.radio[0]
check("sidebar filter has exactly the 2 options, no key box",
      list(r.options) == ["🤖 Smart AI analysis (Groq)", "⚙️ Standard analysis"] and len(at.sidebar.text_input) == 0, str(r.options))
check("with a key, Smart AI is the default and no sidebar warning", r.value.startswith("🤖") and not at.sidebar.warning)
llm.requests.post = counting(groq_ok("manual", only={"email"})); calls["n"] = 0
t = raise_ticket(at)
check("Smart AI: one API call, engine says Groq", calls["n"] == 1 and t["engine"].startswith("Groq"), f"{calls['n']} {t['engine']}")
check("Groq chose 'manual' for emails -> that card has no fix button", t["choices"]["email|null"]["option"] == "manual")
at.session_state["ticket_open"] = True; at.run()
check("detail view shows 'chosen by Groq'", any("chosen by Groq" in m.value for m in at.markdown))
[b for b in at.button if b.label == "Apply ALL suggested fixes"][0].click(); at.run()
left = dq.run_checks(at.session_state["work"], prof)
check("apply ALL leaves only the manual-review emails (45) and status In Progress",
      left.groupby(["column", "check"]).size().to_dict() == {("email", "null"): 45}
      and any("In Progress" in m.value for m in at.markdown if m.value.startswith("###")), str(left.shape))
at = new_app(key="gsk_test", nav=PAGES[3]); at.sidebar.radio[0].set_value("⚙️ Standard analysis").run(); calls["n"] = 0
t = raise_ticket(at); check("Standard chosen with key set: ZERO API calls", calls["n"] == 0 and t["engine"].startswith("Standard"))
at.session_state["ticket_open"] = True; at.run()
check("ticket offers 'Try Smart AI again'", any("Smart AI analysis (Groq) again" in b.label for b in at.button))
llm.requests.post = counting(groq_ok()); [b for b in at.button if "again" in b.label][0].click(); at.run()
check("retry -> Groq engine and working copy reset", at.session_state.ticket["engine"].startswith("Groq") and at.session_state["work"].equals(bad))
[b for b in at.button if "Switch to Standard" in b.label][0].click(); at.run()
check("switch back -> Standard", at.session_state.ticket["engine"].startswith("Standard") and not errs(at))
for code, label in ((429, "rate limit"), (401, "bad key"), (500, "server error")):
    llm.requests.post = lambda url, c=code, **kw: R(c, {"error": {"message": "x"}})
    at = new_app(key="gsk_test", nav=PAGES[3]); t = raise_ticket(at)
    check(f"Groq {code} ({label}) -> falls back to Standard with a visible reason",
          t["engine"].startswith("Standard") and f"{code}" in t["note"] and not errs(at), t["note"][:60])
def offline(url, **kw): raise llm.requests.ConnectionError("down")
llm.requests.post = offline; at = new_app(key="gsk_test", nav=PAGES[3]); t = raise_ticket(at)
check("Groq unreachable -> falls back to Standard", t["engine"].startswith("Standard") and "reach" in t["note"])
llm.requests.post = lambda url, **kw: R(200, {"choices": [{"message": {"content": "no json here"}}]})
at = new_app(key="gsk_test", nav=PAGES[3]); t = raise_ticket(at)
check("Groq garbage reply -> falls back to Standard", t["engine"].startswith("Standard") and "format" in t["note"])
llm._models_cache.clear(); seen = []
def retired(url, **kw):
    seen.append(kw["json"]["model"])
    return R(404, {"error": {"message": "The model `x` does not exist"}}) if len(seen) == 1 else groq_ok()(url, **kw)
llm.requests.post = retired
llm.requests.get = lambda url, **kw: R(200, {"data": [{"id": "openai/gpt-oss-120b", "active": True}, {"id": "whisper-large-v3", "active": True}]})
at = new_app(key="gsk_test", nav=PAGES[3]); t = raise_ticket(at)
check("default model 404 -> auto-switches to an available model", t["engine"] == "Groq (openai/gpt-oss-120b)" and len(seen) == 2, t["engine"])
llm.requests.post = groq_ok("bogus-option")
at = new_app(key="gsk_test", nav=PAGES[3]); t = raise_ticket(at)
check("model returns an invalid option id -> safely falls back to the rule-based fix",
      all(v["option"] in ("rule", "generic", "manual") for v in t["choices"].values()) and not errs(at))
at = new_app(key=None, nav=PAGES[3]); r = at.sidebar.radio[0]
check("no key: Standard is the default and there is no sidebar warning", r.value.startswith("⚙️") and not at.sidebar.warning)
at.sidebar.radio[0].set_value("🤖 Smart AI analysis (Groq)").run(); calls["n"] = 0; llm.requests.post = counting(groq_ok())
t = raise_ticket(at)
check("no key but Smart picked: zero calls, Standard ticket with an explanatory note", calls["n"] == 0 and "not set up" in t["note"])
secrets_file = ROOT / ".streamlit" / "secrets.toml"; had = secrets_file.exists()
secrets_file.parent.mkdir(exist_ok=True)
for toml, label in (('GROQ_API_KEY = "gsk_x"\n', "GROQ_API_KEY"), ('groq_api_key = "gsk_x"\n', "groq_api_key"),
                    ('[groq]\napi_key = "gsk_x"\n', "[groq] api_key"), ('GROQ_API_KEY = ""\n', "empty value")):
    secrets_file.write_text(toml, encoding="utf-8")
    at = new_app(nav=PAGES[3]); want = "⚙️" if label == "empty value" else "🤖"
    check(f"Secrets layout '{label}' is read correctly", at.sidebar.radio[0].value.startswith(want) and not errs(at))
if not had: secrets_file.unlink(missing_ok=True); secrets_file.parent.rmdir()
os.environ.pop("GROQ_API_KEY", None)

# ======================================================================================
section("11. Awkward inputs (temporary copies of the app with swapped data)")
def scenario(name, good_df=None, bad_df=None, raw_bad=None, nav=PAGES[3]):
    d = pathlib.Path(tempfile.mkdtemp(prefix="dqt_"))
    for f in ("app.py", "dq_checks.py", "llm.py"): shutil.copy(ROOT / f, d / f)
    (d / "data").mkdir()
    (good_df if good_df is not None else pd.read_csv(ROOT / "data/good_data.csv", dtype=str, keep_default_na=False, na_values=[""])).to_csv(d / "data/good_data.csv", index=False)
    if raw_bad is not None: (d / "data/bad_data.csv").write_text(raw_bad, encoding="utf-8")
    else: (bad_df if bad_df is not None else pd.read_csv(ROOT / "data/good_data.csv", dtype=str)).to_csv(d / "data/bad_data.csv", index=False)
    out = {}
    for p in PAGES:
        at = new_app(str(d / "app.py"), nav=p); out[p] = errs(at)
    shutil.rmtree(d, ignore_errors=True)
    return out
def all_ok(res): return {p: e for p, e in res.items() if e}

g0 = pd.read_csv(ROOT / "data/good_data.csv", dtype=str, keep_default_na=False, na_values=[""])
res = scenario("identical files"); check("bad file identical to good (0 issues): every page renders", not all_ok(res), str(all_ok(res))[:300])
res = scenario("missing column", bad_df=g0.drop(columns=["email"]))
check("bad file missing a column: no crash on any page (friendly handling)", not all_ok(res), str(all_ok(res))[:300])
res = scenario("extra column", bad_df=g0.assign(surprise="x"))
check("bad file with an extra column: no crash on any page", not all_ok(res), str(all_ok(res))[:300])
res = scenario("header only", raw_bad=",".join(g0.columns) + "\n")
check("bad file with headers but no rows: no crash on any page", not all_ok(res), str(all_ok(res))[:300])
res = scenario("one row", bad_df=g0.head(1))
check("bad file with a single row: no crash on any page", not all_ok(res), str(all_ok(res))[:300])
evil = g0.rename(columns={"email": "<img src=x onerror=alert(1)>"})
evil_bad = evil.copy(); evil_bad.loc[:3, "<img src=x onerror=alert(1)>"] = None
res = scenario("hostile column name", good_df=evil, bad_df=evil_bad)
check("hostile column name in the CSV header: no crash", not all_ok(res), str(all_ok(res))[:300])
d = pathlib.Path(tempfile.mkdtemp(prefix="dqx_"))
for f in ("app.py", "dq_checks.py", "llm.py"): shutil.copy(ROOT / f, d / f)
(d / "data").mkdir(); evil.to_csv(d / "data/good_data.csv", index=False); evil_bad.to_csv(d / "data/bad_data.csv", index=False)
at = new_app(str(d / "app.py"), nav=PAGES[1]); html_all = " ".join(m.value for m in at.markdown)
check("hostile column name is HTML-escaped in the Overview cards (no raw <img> tag reaches the page)",
      "<img src=x" not in html_all and "&lt;img src=x" in html_all)
shutil.rmtree(d, ignore_errors=True)

section("12. Security and hygiene")
tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.split()
leak = [f for f in tracked if re.search(r"gsk_[A-Za-z0-9]{20,}", (ROOT / f).read_text(encoding="utf-8", errors="ignore"))]
check("no API key committed in any tracked file", not leak, str(leak))
gi_text = (ROOT / ".gitignore").read_text()
check(".gitignore protects secrets.toml and caches", ".streamlit/secrets.toml" in gi_text and "__pycache__" in gi_text)
src = (ROOT / "app.py").read_text(encoding="utf-8")
check("no API key input widget anywhere in the app", "type=\"password\"" not in src and "text_input(\"Groq" not in src)
check("llm.py only talks to api.groq.com", set(re.findall(r"https?://[^\"'\s]+", (ROOT / "llm.py").read_text())) <= {
      "https://api.groq.com/openai/v1/chat/completions", "https://api.groq.com/openai/v1/models"})
sent = llm.summarise(bi, prof, 500)
check("data sent to Groq is limited to counts + <=5 sample values per problem (never the full file)",
      all(len(i["sample_bad_values"]) <= 5 for i in sent) and len(json.dumps(sent)) < 8000, str(len(json.dumps(sent))))
check("source CSVs on disk byte-identical after the whole run", all(sha(ROOT / "data" / f) == h for f, h in DATA_HASH.items()))
st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
print("INFO git working tree:", "clean" if not st else st.replace("\n", " | "))
req = (ROOT / "requirements.txt").read_text().lower()
imports = set(re.findall(r"^\s*(?:import|from)\s+([a-z_]+)", "".join((ROOT / f).read_text(encoding="utf-8") for f in ("app.py", "dq_checks.py", "llm.py", "generate_data.py")), re.M))
third = {"streamlit", "pandas", "altair", "numpy", "requests"} & imports
check("requirements.txt lists every third-party import", all(m in req for m in third), str(third))

# ======================================================================================
passed = sum(1 for ok, *_ in RESULTS if ok); failed = [r for r in RESULTS if not r[0]]
print(f"\n{'=' * 60}\nRESULT: {passed}/{len(RESULTS)} checks passed")
for _, n, d in failed: print("  FAILED:", n, f"[{d}]" if d else "")
sys.exit(1 if failed else 0)
