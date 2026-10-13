"""Streamlit data-quality dashboard.  Run: streamlit run app.py
The AI never edits data: it flags issues and drafts a ticket. A person clicks to apply a suggestion,
and even then only an in-session copy changes; the CSV files on disk are never touched."""
import datetime as dt
import os
import pathlib
import altair as alt
import pandas as pd
import streamlit as st
import dq_checks as dq
import llm

st.set_page_config(page_title="Data Quality Triage", page_icon="🔎", layout="wide")
DATA = pathlib.Path(__file__).parent / "data"
COLORS = {"null": "#f8b4b4", "empty": "#fdd5a5", "type": "#d4b8f5", "anomaly": "#fbe88a",
          "category": "#a8d8f0", "format": "#a8d8f0", "rule": "#b8e0b8", "duplicate": "#e0e0e0"}

# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Input files")
    up_good = st.file_uploader("Good (reference) CSV", type="csv")
    up_bad = st.file_uploader("Bad (to inspect) CSV", type="csv")
    st.caption("Leave empty to use the bundled sample files in /data.")
    st.divider()
    st.caption("Read-only analysis: the AI flags problems but never changes your data.")
    st.divider()
    st.subheader("AI engine (optional)")


    def _secret(name):
        try:
            return st.secrets.get(name)
        except Exception:  # no secrets.toml
            return None


    GROQ_KEY = _secret("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
    GROQ_MODEL = _secret("GROQ_MODEL") or os.environ.get("GROQ_MODEL") or llm.DEFAULT_MODEL
    if GROQ_KEY:
        st.success("Smart AI analysis is available. Pick a method when raising a ticket.")
        st.caption("The key's free-tier quota is shared by everyone using this app. Sent to Groq: issue counts and up to "
                   "5 sample bad values per problem, never the full file.")
    else:
        st.info("Smart AI analysis is not set up on this app, so the standard analysis is used.")
    with st.expander("Free-tier limits"):
        st.markdown("- **Rate limits** apply per organisation and vary by model. Roughly 30 requests/min, 12k tokens/min "
                    "and 1k requests/day have been reported for the default model; check your Groq console for live numbers.\n"
                    "- This app makes **one request per ticket**. If the limit is hit, it falls back to built-in rules.\n"
                    "- No uptime guarantee, and models get retired: change the model name if you see a 404.\n"
                    "- Output can vary between runs. Data goes to a third party.")

good = dq.load(up_good or DATA / "good_data.csv")
bad = dq.load(up_bad or DATA / "bad_data.csv")
sig = (up_good.name if up_good else "default", up_bad.name if up_bad else "default", len(bad))
if st.session_state.get("sig") != sig:  # new files -> reset ticket state
    st.session_state.clear()
    st.session_state.sig = sig


@st.cache_data(show_spinner=False)
def analyse(good_df, bad_df):
    prof = dq.learn_reference(good_df)
    return prof, dq.run_checks(good_df, prof), dq.run_checks(bad_df, prof)


prof, good_issues, bad_issues = analyse(good, bad)


def highlight(df, issues):
    sty = pd.DataFrame("", index=df.index, columns=df.columns)
    for r in issues.itertuples():
        sty.at[r.row, r.column] = f"background-color:{COLORS[r.check]};color:#111;font-weight:600"
    return df.style.apply(lambda _: sty, axis=None)


def score(df, issues):
    cells = df.shape[0] * df.shape[1]
    return 100 * (1 - issues[["row", "column"]].drop_duplicates().shape[0] / cells)


def legend():
    st.markdown(" ".join(f"<span style='background:{COLORS[k]};color:#111;padding:2px 8px;border-radius:4px;"
                         f"margin-right:4px'>{v}</span>" for k, v in dq.CHECK_LABEL.items() if k in set(bad_issues.check) | {"null"}),
                unsafe_allow_html=True)


st.title("🔎 Data Quality Triage")
PAGES = ["ℹ️ About the demo", "📊 Overview", "✅ Good data", "❌ Bad data", "🎫 Support ticket"]
st.session_state.setdefault("nav", PAGES[0])
if "_goto" in st.session_state:  # navigation requested last run (a widget cannot be changed after it is drawn)
    st.session_state.nav = st.session_state.pop("_goto")
st.segmented_control("Page", PAGES, key="nav", label_visibility="collapsed")
page = st.session_state.nav or PAGES[0]


def go(p):  # jump to another page (takes effect on the next run)
    st.session_state["_goto"] = p


st.session_state.setdefault("work", bad.copy())  # working copy: only changes when a person clicks a fix
st.session_state.setdefault("applied", {})
st.session_state.setdefault("ticket", None)

CAUSE = {"null": "the field is optional or not captured upstream, or a join/export dropped it",
         "empty": "a form or import accepted whitespace as a value",
         "type": "free text reached a numeric field, so input validation is missing at the source",
         "anomaly": "a scale or unit error, or a data-entry slip",
         "category": "a new or misspelt category value", "format": "an unvalidated free-text field",
         "rule": "fields were edited independently and no longer agree", "duplicate": "a key was loaded twice"}


def analyse_issues(issues):
    """AI analysis: one finding per (column, problem) with a likely cause. Read-only."""
    out = []
    for (col, chk), g in issues.groupby(["column", "check"], sort=False):
        extra = ""
        if chk == "anomaly":
            med = pd.to_numeric(bad[col], errors="coerce").median()
            ratio = pd.to_numeric(g["value"], errors="coerce").median() / med
            extra = f" Flagged values are about {ratio:.0f}x the column median."
        out.append(f"**`{col}`**: {len(g)} x {dq.CHECK_LABEL[chk].lower()} ({100 * len(g) / len(bad):.1f}% of rows). "
                   f"Likely cause: {CAUSE[chk]}.{extra}")
    return out


# ---------------------------------------------------------------- about
if page == PAGES[0]:
    st.markdown("### What is this demo?")
    st.markdown(
        "Every day, data lands in your pipelines. Somewhere in it, a column goes empty, a number is typed as text, "
        "or a price is 50x too high. Normally someone finds out late, then spends an hour digging to find *what* is wrong "
        "and *where*. This proof of concept shows an **AI data-quality agent doing that first hour of digging for you**.")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**The AI does**")
        st.markdown("- Learns what *normal* looks like from a known-good file\n- Checks a new file against it\n"
                    "- Flags the exact cells that are wrong, and says why\n- Drafts a support ticket with a suggested fix for each problem")
    with c2, st.container(border=True):
        st.markdown("**The AI never**")
        st.markdown("- Changes your data on its own\n- Edits the files on disk\n- Pushes a failing file to the next stage\n"
                    "- Applies a fix without a person clicking it")

    st.info("**How the AI part works.** Finding problems is rule-based and deterministic. With a Groq API key in the sidebar, "
            "an LLM then analyses the problems, explains the likely cause, and **chooses the best fix for each one** from "
            "options the rule engine can compute exactly (or recommends manual review). You can instead pick the **Standard** "
            "method (built-in rules, no AI) when raising a ticket, and a ticket falls back to it automatically if Groq is "
            "rate-limited or down. Either way, nothing changes until a person clicks.")
    st.markdown("### How it works, in 4 steps")
    s1, s2, s3, s4 = st.columns(4)
    for col, n, title, body in ((s1, "1", "Learn", "Reads the good file and learns each column's type, range and allowed values."),
                                (s2, "2", "Check", "Runs the same checks on the bad file: nulls, blanks, wrong types, out-of-range values."),
                                (s3, "3", "Flag", "Highlights each broken cell and gives the file a PASS or FAIL verdict."),
                                (s4, "4", "Ticket", "On your say-so, raises a ticket in the data owner's name. A person applies the fixes.")):
        with col, st.container(border=True):
            st.markdown(f"**{n}. {title}**")
            st.caption(body)

    st.markdown("### A small example")
    st.write("Here are 4 rows in the style of the bad file (the real files have 500 rows x 20 columns). "
             "Colours show what the AI flagged. The first row passes every check.")
    ex = pd.DataFrame({"order_id": ["100007", "100014", "100031", "100052"],
                       "email": ["zoe.garcia1040@example.com", None, "lena.patel1189@example.com", "omar.khan1002@example.com"],
                       "quantity": ["3", "2", "three", "4"],
                       "unit_price": ["21.79", "40.10", "15.50", "1210.69"],
                       "total_amount": ["65.37", "80.20", "46.50", "48.40"]})
    flag = {(1, "email"): "null", (2, "quantity"): "type", (3, "unit_price"): "anomaly"}
    sty = pd.DataFrame("", index=ex.index, columns=ex.columns)
    for (r, c), k in flag.items():
        sty.at[r, c] = f"background-color:{COLORS[k]};color:#111;font-weight:600"
    st.dataframe(ex.style.apply(lambda _: sty, axis=None), hide_index=True, width="stretch")
    st.markdown(
        "- **Row 2, `email` is null.** A value is missing. The real address must come from the source system, so only a placeholder is offered.\n"
        "- **Row 3, `quantity` is `three`.** Text in a numeric column. Suggested fix: work it back from the total, "
        "46.50 / 15.50 = **3**.\n"
        "- **Row 4, `unit_price` is 1210.69.** Far outside the normal range. Suggested fix: total / quantity = 48.40 / 4 = **12.10**.")

    st.markdown("### Limitations")
    st.markdown("- **Checks are only as good as the reference file.** The profile is learned from the good file, so anything "
                "it doesn't cover (new valid categories, seasonal ranges) can be flagged wrongly or missed.\n"
                "- **Groq free tier:** rate-limited per organisation (reported ~30 requests/min, ~12k tokens/min, ~1k requests/day "
                "for the default model; verify in your Groq console), no uptime guarantee, and models can be retired. "
                "One request is used per ticket, and any failure falls back to built-in rules.\n"
                "- **An LLM can be wrong.** Groq only picks from fixes the rule engine can compute exactly, can recommend "
                "manual review, and nothing is applied until a person clicks. Its wording and choices can differ between runs.\n"
                "- **Privacy:** with a key set, issue counts and up to 5 sample bad values per problem are sent to Groq, "
                "never the full file. This demo uses synthetic data only.\n"
                "- **Demo scope:** mock tickets (nothing is sent to a real ticketing system), one pair of files at a time, "
                "state kept per browser session.")
    st.markdown("### Try the demo")
    st.markdown("1. **Overview**: see the verdicts. The good file passes, the bad file fails.\n"
                "2. **Bad data**: see *where* it breaks with charts, then the highlighted cells.\n"
                "3. Still in **Bad data**, type a data owner's name and let the AI raise a ticket.\n"
                "4. **Support ticket**: click the ticket ID, read the AI analysis, and click **Apply suggested fix**.")
    st.button("Start the demo →", on_click=go, args=(PAGES[1],), type="primary")

def build_analysis(issues, use_groq=True):
    """Returns summary + bullet lines + which engine wrote them. LLM text if a key is set, else templates."""
    base = analyse_issues(issues)
    summary = (f"{len(issues)} problems were found in {issues['column'].nunique()} of {bad.shape[1]} columns "
               "when bad_data.csv was checked against the profile learned from the reference file.")
    if not (GROQ_KEY and use_groq):
        return {"summary": summary, "analysis": base, "engine": "Standard (rule-based)", "note": "", "choices": {}}
    try:
        res = llm.analyse(llm.summarise(issues, prof, len(bad)), GROQ_KEY, GROQ_MODEL)
    except llm.LLMError as e:
        return {"summary": summary, "analysis": base, "engine": "Standard (rule-based)",
                "note": f"Groq analysis unavailable ({e}); showing built-in analysis and rule-based fixes instead.",
                "choices": {}}
    lines, choices = [], {}
    for (col, chk), g in issues.groupby(["column", "check"], sort=False):
        c = res["causes"].get((col, chk))
        head = f"**`{col}`**: {len(g)} x {dq.CHECK_LABEL[chk].lower()} ({100 * len(g) / len(bad):.1f}% of rows). "
        if c:
            lines.append(head + f"Likely cause: {c['cause']} Next step: {c['step']}")
            choices[f"{col}|{chk}"] = {"option": c["option"], "rationale": c["rationale"]}
        else:  # model skipped this one: keep the rule-based text and fix
            lines.append(head + base[len(lines)].split("). ", 1)[-1])
    return {"summary": res["summary"], "analysis": lines, "engine": f"Groq ({GROQ_MODEL})", "note": "",
            "choices": choices}


def reanalyse(use_groq):
    """Switch the ticket to the other analysis method. Fix choices change, so the working copy is reset."""
    st.session_state.ticket.update(build_analysis(bad_issues, use_groq))
    st.session_state.work, st.session_state.applied = bad.copy(), {}


def fixes_for(g, key):
    """The fix to apply for one problem group: whichever option the engine (Groq or rules) chose.
    Returns (rows to change with a `new` column, description, who chose it)."""
    t = st.session_state.ticket or {}
    ch = t.get("choices", {}).get(key)
    opt = ch["option"] if ch else "rule"
    who = f"chosen by {t['engine']}" if ch else "rule-based"
    why = f" Why: {ch['rationale']}" if ch and ch["rationale"] else ""
    if opt == "manual":
        return g.iloc[0:0].assign(new=None), f"Not changed automatically: needs the data owner or the source system.{why}", who
    vcol, dcol = ("alt_value", "alt_desc") if opt == "generic" else ("fix_value", "fix_desc")
    f = g.dropna(subset=[vcol])
    if f.empty:
        return g.iloc[0:0].assign(new=None), "No safe automatic fix can be inferred; needs a human or the source system.", who
    return f.assign(new=f[vcol]), g.iloc[0][dcol] + why, who


# ---------------------------------------------------------------- overview (verdicts only)
OV_CSS = """<style>
.dq-strip{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 18px}
.dq-tile{flex:1;min-width:150px;padding:12px 16px;border-radius:14px;background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.22)}
.dq-tile b{display:block;font-size:26px;line-height:1.1}.dq-tile span{font-size:12px;opacity:.7;letter-spacing:.04em;text-transform:uppercase}
.dq-tile.ok b{color:#1f9d55}.dq-tile.bad b{color:#e0383e}
.dq-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:2rem;margin-bottom:.6rem}
.dq-card{border-radius:18px;padding:20px 22px;border:1px solid rgba(128,128,128,.25);background:rgba(128,128,128,.06);position:relative;overflow:hidden}
.dq-card:before{content:"";position:absolute;left:0;top:0;right:0;height:5px;background:var(--acc)}
.dq-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.dq-file{font-weight:600;font-size:15px}
.dq-pill{padding:4px 14px;border-radius:999px;font-weight:700;font-size:13px;letter-spacing:.06em;color:#fff;background:var(--acc)}
.dq-body{display:flex;gap:18px;align-items:center;margin-bottom:14px}
.dq-ringw{flex:0 0 112px;height:112px;position:relative}
.dq-ring{position:absolute;inset:0;border-radius:50%;background:conic-gradient(var(--acc) calc(var(--p)*1%),rgba(128,128,128,.22) 0);
 -webkit-mask:radial-gradient(farthest-side,transparent 70%,#000 71%);mask:radial-gradient(farthest-side,transparent 70%,#000 71%)}
.dq-rt{position:absolute;inset:0;display:grid;place-content:center;text-align:center;line-height:1.1}
.dq-rt b{font-size:24px;display:block}.dq-rt small{font-size:10.5px;opacity:.65;text-transform:uppercase;letter-spacing:.05em}
.dq-msg h3{margin:0 0 4px;font-size:19px;padding:0}.dq-msg p{margin:0;font-size:13.5px;opacity:.8}
.dq-track{display:flex;align-items:center;gap:6px;margin:12px 0 0;font-size:12px;flex-wrap:wrap}
.dq-step{padding:3px 10px;border-radius:999px;border:1px solid rgba(128,128,128,.3)}
.dq-step.done{background:rgba(31,157,85,.15);border-color:#1f9d55;color:#1f9d55;font-weight:600}
.dq-step.stop{background:rgba(224,56,62,.15);border-color:#e0383e;color:#e0383e;font-weight:600}
.dq-stats{display:flex;gap:10px;margin:2px 0 12px}
.dq-stat{flex:1;padding:8px 12px;border-radius:12px;background:rgba(128,128,128,.09)}
.dq-stat b{display:block;font-size:18px}.dq-stat span{font-size:11px;opacity:.65;text-transform:uppercase;letter-spacing:.05em}
.dq-checks{border-top:1px solid rgba(128,128,128,.22);padding-top:8px}
.dq-row{display:flex;align-items:center;gap:10px;padding:5px 0;font-size:13.5px}
.dq-dot{flex:0 0 20px;height:20px;border-radius:50%;display:grid;place-items:center;font-size:11px;font-weight:700;color:#fff}
.dq-dot.ok{background:#1f9d55}.dq-dot.no{background:#e0383e}
.dq-row .n{margin-left:auto;font-weight:700;color:#e0383e;white-space:nowrap}
.dq-row .c{opacity:.6;font-size:12px}
</style>"""

CHECK_GROUPS = [("Null values", ("null",)), ("Blank values", ("empty",)), ("Data types", ("type",)),
                ("Value ranges", ("anomaly",)), ("Allowed values and formats", ("category", "format")),
                ("Business rules", ("rule", "duplicate"))]


def file_card(name, df, iss):
    ok = iss.empty
    acc = "#1f9d55" if ok else "#e0383e"
    rows_bad = iss["row"].nunique()
    clean = 100 * (1 - rows_bad / len(df))
    rows = ""
    for label, kinds in CHECK_GROUPS:
        sub = iss[iss["check"].isin(kinds)]
        if sub.empty:
            rows += f'<div class="dq-row"><span class="dq-dot ok">✓</span>{label}</div>'
        else:
            cols = ", ".join(sub["column"].unique()[:3])
            rows += (f'<div class="dq-row"><span class="dq-dot no">✕</span>{label} <span class="c">in {cols}</span>'
                     f'<span class="n">{len(sub)}</span></div>')
    if ok:
        title, text = "Ready for the next stage", "Every check passed. Safe to push to the next data process / stream."
        track = ('<span class="dq-step done">✓ Received</span>→<span class="dq-step done">✓ Checked</span>'
                 '→<span class="dq-step done">✓ Promoted</span>')
    else:
        title = f"Blocked: {len(iss)} issues found"
        text = f"{iss['column'].nunique()} of {df.shape[1]} columns have problems. Held back until the data owner fixes them."
        track = ('<span class="dq-step done">✓ Received</span>→<span class="dq-step done">✓ Checked</span>'
                 '→<span class="dq-step stop">✕ Held back</span>')
    return (f'<div class="dq-card" style="--acc:{acc}">'
            f'<div class="dq-head"><span class="dq-file">📄 {name}</span><span class="dq-pill">{"PASS" if ok else "FAIL"}</span></div>'
            f'<div class="dq-body"><div class="dq-ringw"><div class="dq-ring" style="--p:{clean:.1f}"></div><div class="dq-rt"><b>{clean:.0f}%</b><small>rows clean</small></div></div>'
            f'<div class="dq-msg"><h3>{title}</h3><p>{text}</p><div class="dq-track">{track}</div></div></div>'
            f'<div class="dq-stats"><div class="dq-stat"><b>{len(df)}</b><span>Rows</span></div>'
            f'<div class="dq-stat"><b>{df.shape[1]}</b><span>Columns</span></div>'
            f'<div class="dq-stat"><b>{len(iss)}</b><span>Issues</span></div></div>'
            f'<div class="dq-checks">{rows}</div></div>')


if page == PAGES[1]:
    g_ok, b_ok = good_issues.empty, bad_issues.empty
    st.markdown(OV_CSS, unsafe_allow_html=True)
    n_pass, n_fail = int(g_ok) + int(b_ok), int(not g_ok) + int(not b_ok)
    st.markdown(f'<div class="dq-strip"><div class="dq-tile"><b>2</b><span>Files checked</span></div>'
                f'<div class="dq-tile ok"><b>{n_pass}</b><span>Ready to promote</span></div>'
                f'<div class="dq-tile bad"><b>{n_fail}</b><span>Blocked</span></div>'
                f'<div class="dq-tile"><b>{len(good_issues) + len(bad_issues)}</b><span>Issues found</span></div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="dq-grid">' + file_card("good_data.csv", good, good_issues)
                + file_card("bad_data.csv", bad, bad_issues) + '</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    c1.button("Explore good data →", key="go_good", on_click=go, args=(PAGES[2],), width="stretch")
    c2.button("See exactly where it breaks →", key="go_bad", on_click=go, args=(PAGES[3],), type="primary", width="stretch")
    st.caption("Checks learned from the good file: nulls, blanks, data types, value ranges, allowed values and business rules. "
               "Read-only: the AI never changes your data.")

# ---------------------------------------------------------------- good
if page == PAGES[2]:
    st.success(f"{good.shape[0]} rows x {good.shape[1]} columns. {len(good_issues)} issues found. "
               "This file is the reference the checks learn from.")
    st.dataframe(good, width="stretch", height=420)
    with st.expander("Learned profile per column"):
        st.json(prof)

# ---------------------------------------------------------------- bad
if page == PAGES[3]:
    st.error(f"{len(bad_issues)} issues across {bad_issues['column'].nunique()} columns. Highlighted cells are the exact problems.")
    st.markdown("#### 🤖 Want the AI to analyse this and suggest fixes?")
    with st.container(border=True):
        if st.session_state.ticket is None:
            st.write("The AI will analyse every issue, work out likely causes, and draft a **support ticket in the "
                     "data owner's name** with a suggested fix for each problem. It will not change any data.")
            owner = st.text_input("Data owner (the ticket will be raised in this name)", key="owner",
                                  placeholder="e.g. Priya Sharma")
            GROQ_LABEL = "🤖 Smart AI analysis (Groq)"
            STD_LABEL = "⚙️ Standard analysis"
            captions = {GROQ_LABEL: "An AI reads the problems like a data expert, explains what probably went wrong and "
                                    "suggests the best fix for each one. Uses a limited free quota, so it can occasionally be busy.",
                        STD_LABEL: "Simple built-in checks and fixed advice. No AI involved. Always works, instantly."}
            options = [GROQ_LABEL, STD_LABEL] if GROQ_KEY else [STD_LABEL]
            mode = st.radio("How should the problems be analysed?", options, key="analysis_mode",
                            captions=[captions[o] for o in options],
                            help="If the Smart AI is busy or unavailable, the ticket automatically uses the Standard "
                                 "analysis instead, and you can switch from the ticket at any time.")
            if not GROQ_KEY:
                st.caption("Smart AI analysis is not set up on this app, so the standard analysis is used.")
            if st.button("Yes, analyse and raise a ticket", type="primary", disabled=not owner.strip()):
                with st.spinner("Analysing issues and drafting the ticket..."):
                    st.session_state.ticket = {"id": f"DQ-{dt.datetime.now():%Y%m%d-%H%M}", "opened": dt.datetime.now(),
                                               "owner": owner.strip(),
                                               **build_analysis(bad_issues, mode == GROQ_LABEL)}
                st.session_state.ticket_open = False
                go(PAGES[4])
                st.rerun()
        else:
            t0 = st.session_state.ticket
            st.success(f"Ticket **{t0['id']}** was raised in the name of **{t0['owner']}**.")
            st.button("Open the support ticket →", on_click=go, args=(PAGES[4],), type="primary")
    ci = bad_issues.assign(label=bad_issues["check"].map(dq.CHECK_LABEL))
    k = st.columns(4)
    k[0].metric("Total issues", len(ci))
    k[1].metric("Columns affected", f"{ci['column'].nunique()} of {bad.shape[1]}")
    k[2].metric("Rows affected", ci["row"].nunique())
    k[3].metric("Cells affected", f"{len(ci[['row', 'column']].drop_duplicates())} / {bad.size}")
    st.markdown("#### Where is it breaking?")
    chart_view = st.radio("View", ["Affected columns", "Summary table", "Row map (where in the file)"], horizontal=True,
                    label_visibility="collapsed")
    domain = [dq.CHECK_LABEL[c] for c in COLORS if c in set(bad_issues["check"])]
    rng = [COLORS[c] for c in COLORS if c in set(bad_issues["check"])]
    colour = alt.Color("label:N", title="Problem", scale=alt.Scale(domain=domain, range=rng))
    if chart_view == "Affected columns":  # only the broken columns, ranked
        agg = ci.groupby(["column", "label"]).size().reset_index(name="issues")
        order = ci["column"].value_counts().index.tolist()
        st.altair_chart(alt.Chart(agg).mark_bar().encode(
            y=alt.Y("column:N", sort=order, title=None), x=alt.X("issues:Q", title="Number of issues"),
            color=colour, tooltip=["column", "label", "issues"]).properties(height=40 * len(order) + 40),
            width="stretch")
        st.caption(f"The other {bad.shape[1] - len(order)} columns passed every check, so they are left out.")
    elif chart_view == "Summary table":
        rows = []
        for c in bad.columns:
            sub = ci[ci["column"] == c]
            rows.append({"column": c, "expected type": prof[c]["kind"], "status": "🔴 Issues" if len(sub) else "🟢 OK",
                         "issues": len(sub), "% of rows": 100 * len(sub) / len(bad),
                         "problem": ", ".join(sorted(set(sub["label"]))) or "-"})
        t = pd.DataFrame(rows).sort_values("issues", ascending=False)
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "% of rows": st.column_config.ProgressColumn("% of rows", format="%.1f%%", min_value=0, max_value=100)})
    else:  # heatmap: which stretches of the file are affected, per column
        bucket = 25
        hm = ci.assign(rows=(ci["row"] // bucket * bucket).map(lambda r: f"{r + 2}-{r + bucket + 1}"),
                       start=ci["row"] // bucket * bucket)
        agg = hm.groupby(["column", "rows", "start"]).size().reset_index(name="issues")
        st.altair_chart(alt.Chart(agg).mark_rect().encode(
            x=alt.X("rows:N", sort=alt.SortField("start"), title="CSV lines (buckets of 25)"),
            y=alt.Y("column:N", title=None), color=alt.Color("issues:Q", scale=alt.Scale(scheme="reds"), title="Issues"),
            tooltip=["column", "rows", "issues"]).properties(height=40 * ci["column"].nunique() + 60),
            width="stretch")
        st.caption("Darker cell = more bad values in that column for that stretch of rows.")
    st.markdown("#### Exact cells")
    legend()
    f1, f2 = st.columns([2, 1])
    cols = f1.multiselect("Filter issues by column", sorted(bad_issues["column"].unique()))
    only = f2.toggle("Only rows with issues", value=True)
    iss = bad_issues[bad_issues["column"].isin(cols)] if cols else bad_issues
    view = bad.loc[sorted(iss["row"].unique())] if only else bad
    st.dataframe(highlight(view, iss), width="stretch", height=420)
    st.subheader("Issue log")
    log = iss.assign(row_in_csv=iss["row"] + 2, check=iss["check"].map(dq.CHECK_LABEL))
    st.dataframe(log[["row_in_csv", "column", "check", "value", "message"]].rename(columns={"row_in_csv": "csv line"}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- ticket
SLA_DAYS = 2
if page == PAGES[4]:
    t = st.session_state.ticket
    if t is None:
        st.button("← Back to Bad data", on_click=go, args=(PAGES[3],))
    else:
        work = st.session_state.work
        live = dq.run_checks(work, prof)
        groups = bad_issues.groupby(["column", "check"], sort=False)
        status = "Resolved" if live.empty else ("In Progress" if st.session_state.applied else "Open")
        badge = {"Open": ":orange-badge[Open]", "In Progress": ":blue-badge[In Progress]",
                 "Resolved": ":green-badge[Resolved]"}[status]
        due = t["opened"] + dt.timedelta(days=SLA_DAYS)
        left = due - dt.datetime.now()
        sla_left = "OVERDUE" if left.total_seconds() < 0 else f"{left.days}d {left.seconds // 3600}h left"
        title = "Data quality failures in bad_data.csv"

        if not st.session_state.get("ticket_open"):
            # ---- ticket list, like a helpdesk queue
            st.markdown("### Support tickets")
            w = [2.6, 2.6, 1.4, 1.2, 1.1, 1.7, 1.5]
            for col, h in zip(st.columns(w), ["Ticket ID", "Title", "Ticket owner", "Created by", "Status", "Created", "SLA"]):
                col.caption(h.upper())
            with st.container(border=True):
                c = st.columns(w, vertical_alignment="center")
                c[0].button(t["id"], key="open_ticket", type="tertiary", help="Click to see ticket details",
                            on_click=lambda: st.session_state.update(ticket_open=True))
                c[1].markdown(title)
                c[2].markdown(t["owner"])
                c[3].markdown("🤖 AI Agent")
                c[4].markdown(badge)
                c[5].markdown(f"{t['opened']:%d %b %Y, %H:%M}")
                c[6].markdown(f"{SLA_DAYS} days  \n:gray[{sla_left}]")
            st.caption("Click the ticket ID to open the ticket.")
        else:
            # ---- ticket detail
            st.button("← All tickets", on_click=lambda: st.session_state.update(ticket_open=False))
            st.markdown(f"### {t['id']}  {badge}")
            st.markdown(f"**{title}**")
            with st.container(border=True):
                f = st.columns(4)
                f[0].caption("TICKET OWNER"); f[0].markdown(f"**{t['owner']}**")
                f[1].caption("CREATED BY"); f[1].markdown("**🤖 AI Agent**")
                f[2].caption("CREATED"); f[2].markdown(f"**{t['opened']:%d %b %Y, %H:%M}**")
                f[3].caption("SLA"); f[3].markdown(f"**{SLA_DAYS} days**  \n:gray[due {due:%d %b, %H:%M} ({sla_left})]")
                g2 = st.columns(4)
                g2[0].caption("PRIORITY"); g2[0].markdown(f"**{'High' if len(bad_issues) / bad.size > .005 else 'Medium'}**")
                g2[1].caption("SOURCE FILE"); g2[1].markdown("**bad_data.csv**")
                g2[2].caption("ISSUES FOUND"); g2[2].markdown(f"**{len(bad_issues)}** in {bad_issues['column'].nunique()} columns")
                g2[3].caption("REMAINING"); g2[3].markdown(f"**{len(live)}**")
            st.markdown("#### Description")
            st.write(t["summary"] + " The file has been blocked from the next data process. The agent has **not** "
                     "modified any data: review each finding below and click **Apply suggested fix** to change the "
                     "working copy yourself.")
            if t.get("note"):
                st.warning(t["note"])
            if GROQ_KEY:
                is_groq = t["engine"].startswith("Groq")
                m1, m2 = st.columns([1, 2], vertical_alignment="center")
                if is_groq:
                    m1.button("⚙️ Switch to Standard analysis", on_click=reanalyse, args=(False,))
                else:
                    m1.button("🤖 Try Smart AI analysis (Groq) again", on_click=reanalyse, args=(True,), type="primary")
                m2.caption("Switching re-runs the analysis and resets the working copy.")
            with st.expander(f"🤖 AI analysis: likely causes  ·  written by {t['engine']}", expanded=True):
                for line in t["analysis"]:
                    st.markdown("- " + line)
            st.markdown("#### Findings and suggested fixes")
            for (col, chk), g in groups:
                key = f"{col}|{chk}"
                done = key in st.session_state.applied
                with st.container(border=True):
                    a, b = st.columns([4, 1])
                    a.markdown(f"##### {'✅' if done else '🔴'} `{col}`: {dq.CHECK_LABEL[chk]}  ({len(g)} rows)")
                    a.markdown(f"**What is wrong:** {g.iloc[0]['message']}" + (" (example)" if len(g) > 1 else ""))
                    fx, fdesc, who = fixes_for(g, key)
                    a.markdown(f"**Suggested fix ({who}):** {fdesc}")
                    prev = fx.assign(csv_line=fx["row"] + 2, current=fx["value"], suggested=fx["new"])
                    with a.expander("Show affected rows and before/after"):
                        st.dataframe(prev[["csv_line", "current", "suggested"]] if len(fx) else
                                     g.assign(csv_line=g["row"] + 2, current=g["value"])[["csv_line", "current"]],
                                     hide_index=True, width="stretch")
                    fixable = len(fx) > 0
                    if done:
                        b.success("Applied")
                        if b.button("Undo", key=f"undo_{key}"):
                            for r in g.itertuples():
                                work.at[r.row, r.column] = r.value
                            st.session_state.applied.pop(key)
                            st.rerun()
                    elif b.button("Apply suggested fix", key=f"fix_{key}", type="primary", disabled=not fixable):
                        for r in fx.itertuples():
                            work.at[r.row, r.column] = r.new
                        st.session_state.applied[key] = len(g)
                        st.rerun()
            c1, c2, c3 = st.columns(3)
            if c1.button("Apply ALL suggested fixes"):
                for (col, chk), g in groups:
                    fx, _, _ = fixes_for(g, f"{col}|{chk}")
                    for r in fx.itertuples():
                        work.at[r.row, r.column] = r.new
                    if len(fx):
                        st.session_state.applied[f"{col}|{chk}"] = len(g)
                st.rerun()
            if c2.button("Reset working copy"):
                st.session_state.work, st.session_state.applied = bad.copy(), {}
                st.rerun()
            c3.download_button("Download corrected copy", work.to_csv(index=False), "bad_data_corrected.csv",
                               "text/csv", help="Original files on disk are never modified.")
            if not live.empty and st.session_state.applied:
                with st.expander(f"Still failing after your fixes ({len(live)})"):
                    st.dataframe(live[["row", "column", "check", "value", "message"]], hide_index=True)
            st.markdown("#### Activity")
            st.markdown(f"- {t['opened']:%d %b %H:%M}: ticket created by **AI Agent**\n"
                        f"- {t['opened']:%d %b %H:%M}: assigned to **{t['owner']}** (data owner), SLA {SLA_DAYS} days")
            for k, n in st.session_state.applied.items():
                st.markdown(f"- fix applied by a person to **{k.split('|')[0]}** ({n} rows)")
