"""Read-only data-quality checks. Nothing here mutates the input; it only returns findings
(each with a *suggested* fix that a human may choose to apply later)."""
import difflib
import re
import pandas as pd

WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
CHECK_LABEL = {"null": "Null value", "empty": "Empty value", "type": "Wrong data type",
               "anomaly": "Anomaly (out of range)", "category": "Unexpected category",
               "format": "Bad format", "rule": "Rule violation", "duplicate": "Duplicate key"}


def load(src):
    """Strings only; only a truly blank cell is NaN, so 'N/A' or '  ' stay visible as values."""
    return pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[""])


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def learn_reference(ref: pd.DataFrame) -> dict:
    """Learn the expected profile of each column from the known-good file."""
    prof = {}
    for c in ref.columns:
        s = ref[c].dropna()
        n = _num(s)
        if n.notna().all():
            integer = (n % 1 == 0).all() and not s.str.contains(r"\.").any()
            span = n.max() - n.min()
            prof[c] = {"kind": "int" if integer else "float", "lo": n.min() - 0.25 * span,
                       "hi": n.max() + 0.25 * span, "median": float(n.median()), "unique": n.is_unique}
        elif pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").notna().all():
            prof[c] = {"kind": "date"}
        elif s.isin(["True", "False"]).all():
            prof[c] = {"kind": "bool", "allowed": ["True", "False"], "mode": s.mode()[0]}
        elif s.nunique() <= 12:
            prof[c] = {"kind": "cat", "allowed": sorted(s.unique()), "mode": s.mode()[0]}
        else:
            prof[c] = {"kind": "text", "email": bool(s.str.match(EMAIL).all())}
    return prof


def run_checks(df: pd.DataFrame, prof: dict) -> pd.DataFrame:
    out = []

    cur = {}  # per-column generic alternative (statistical fill), attached to every issue in that column

    def add(row, col, check, msg, fix=None, fix_desc="Manual review needed"):
        out.append({"row": int(row), "column": col, "check": check, "value": df.at[row, col],
                    "message": msg, "fix_value": fix, "fix_desc": fix_desc,
                    "alt_value": cur.get("v"), "alt_desc": cur.get("d")})

    # figures used to build suggestions come from valid values in *this* file
    valid_num = {c: _num(df[c]).dropna() for c in df.columns if prof[c]["kind"] in ("int", "float")}
    for c, p in prof.items():
        s = df[c]
        kind = p["kind"]
        mode = s[s.isin(p.get("allowed", []))].mode()
        mode = mode.iloc[0] if len(mode) else None
        if kind == "text" and not p.get("email"):  # free text: most common valid value
            vc = s[s.notna() & (s.str.strip() != "")].mode()
            mode = vc.iloc[0] if len(vc) else None

        def by_country(i):  # city depends on country: most common city among rows of the same country
            if c != "city" or "country" not in df.columns or pd.isna(df.at[i, "country"]):
                return mode, f"Fill with most common value ('{mode}')"
            same = df.loc[(df["country"] == df.at[i, "country"]) & df["city"].notna() & (df["city"].str.strip() != ""), "city"]
            if same.empty:
                return mode, f"Fill with most common value ('{mode}')"
            m = same.mode().iloc[0]
            return m, f"Fill with most common city for country {df.at[i, 'country']} ('{m}')"
        med = valid_num[c].median() if c in valid_num and len(valid_num[c]) else p.get("median")
        med_txt = None if med is None else (str(int(round(med))) if kind == "int" else f"{med:.2f}")
        if med_txt is not None:
            cur["v"], cur["d"] = med_txt, f"Fill with the column median ({med_txt})"
        elif mode is not None:
            cur["v"], cur["d"] = mode, f"Fill with the column's most common value ('{mode}')"
        else:
            cur.clear()

        for i in s.index[s.isna()]:  # null
            if kind in ("int", "float"):
                add(i, c, "null", "Value is missing (null)", med_txt, f"Fill with column median ({med_txt})")
            elif mode is not None:
                fv, fd = by_country(i)
                add(i, c, "null", "Value is missing (null)", fv, fd)
            elif p.get("email"):
                add(i, c, "null", "Email is missing (null)", "missing@placeholder.invalid",
                    "Insert placeholder 'missing@placeholder.invalid' (real address must come from the source)")
            else:
                add(i, c, "null", "Value is missing (null)")
        present = s.dropna()
        for i in present.index[present.str.strip() == ""]:  # empty / whitespace-only
            fv, fd = by_country(i) if mode else (None, "Manual review needed")
            add(i, c, "empty", "Value is blank (whitespace only)", fv, fd)
        present = present[present.str.strip() != ""]

        if kind in ("int", "float"):
            n = _num(present)
            for i in n.index[n.isna()]:  # wrong type
                raw = present[i].strip().lower()
                m = re.match(r"^(\d+(\.\d+)?)", raw)
                guess = str(WORDS[raw]) if raw in WORDS else (m.group(1) if m else med_txt)
                how = ("Convert text to number" if raw in WORDS or m else f"Replace with column median ({med_txt})")
                if c == "quantity":  # the row's own total/price/discount pin the quantity down
                    try:
                        u, d, t = float(df.at[i, "unit_price"]), float(df.at[i, "discount_pct"]), float(df.at[i, "total_amount"])
                        q = round(t / (u * (1 - d / 100)))
                        if 1 <= q <= 10 and abs(q * u * (1 - d / 100) - t) < 0.05:
                            guess, how = str(q), "Recompute from total_amount / (unit_price x (1 - discount))"
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
                add(i, c, "type", f"Text '{present[i]}' in a numeric column (expected {kind})", guess,
                    f"{how} -> {guess}")
            ok = n.dropna()
            for i in ok.index[(ok < p["lo"]) | (ok > p["hi"])]:  # anomaly
                fix, desc = med_txt, f"Replace with column median ({med_txt})"
                if c == "unit_price":  # back out the price from the (consistent) total
                    try:
                        q, d, t = float(df.at[i, "quantity"]), float(df.at[i, "discount_pct"]), float(df.at[i, "total_amount"])
                        fix = f"{t / (q * (1 - d / 100)):.2f}"
                        desc = f"Recompute from total_amount / (quantity x (1 - discount)) -> {fix}"
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
                add(i, c, "anomaly", f"{ok[i]:g} is outside the expected range {p['lo']:.2f} to {p['hi']:.2f}", fix, desc)
            if p.get("unique"):
                for i in present.index[present.duplicated(keep="first")]:
                    add(i, c, "duplicate", "Duplicate of an earlier key")
        elif kind == "date":
            d = pd.to_datetime(present, format="%Y-%m-%d", errors="coerce")
            for i in d.index[d.isna()]:
                add(i, c, "type", f"'{present[i]}' is not a valid YYYY-MM-DD date")
        elif kind in ("cat", "bool"):
            for i in present.index[~present.isin(p["allowed"])]:
                near = difflib.get_close_matches(present[i], p["allowed"], n=1, cutoff=0.6)
                fix = near[0] if near else p["mode"]
                add(i, c, "category", f"'{present[i]}' not in expected set {p['allowed']}", fix, f"Replace with '{fix}'")
        elif p.get("email"):
            for i in present.index[~present.str.match(EMAIL)]:
                add(i, c, "format", f"'{present[i]}' is not a valid email")

    cur.clear()
    found = {(o["row"], o["column"]) for o in out}
    od, sd = pd.to_datetime(df["order_date"], errors="coerce"), pd.to_datetime(df["ship_date"], errors="coerce")
    for i in df.index[(sd < od)]:
        add(i, "ship_date", "rule", "ship_date is earlier than order_date")
    if {"quantity", "unit_price", "discount_pct", "total_amount"} <= set(df.columns):
        q, u, d, t = (_num(df[k]) for k in ("quantity", "unit_price", "discount_pct", "total_amount"))
        bad = ((q * u * (1 - d / 100) - t).abs() > 0.05) & q.notna()
        for i in df.index[bad]:
            if (i, "unit_price") not in found:  # otherwise it is the same root cause
                add(i, "total_amount", "rule", "total_amount != quantity x unit_price x (1 - discount)")
    return pd.DataFrame(out, columns=["row", "column", "check", "value", "message", "fix_value", "fix_desc",
                                          "alt_value", "alt_desc"])
