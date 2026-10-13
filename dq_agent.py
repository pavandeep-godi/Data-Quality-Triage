"""Autonomous data-quality triage agent (prototype, stdlib only).

Loop per partition:  profile -> detect vs learned baseline -> trace column lineage upstream
-> rank culprit from change events -> quarantine -> open ticket -> (human-approved) backfill.
Run:  python dq_agent.py
"""
import random, statistics, pathlib, json
from dataclasses import dataclass, field

# ----------------------------------------------------------------- simulated warehouse
LINEAGE = {  # downstream column -> upstream columns (column-level lineage)
    "stg_orders.amount": ["raw_orders.amount"],
    "stg_orders.order_id": ["raw_orders.order_id"],
    "stg_payments.paid": ["raw_payments.paid"],
    "stg_payments.order_id": ["raw_payments.order_id"],
    "mart_revenue.revenue": ["stg_orders.amount", "stg_payments.paid"],
    "mart_revenue.customer_id": ["stg_orders.order_id", "stg_payments.order_id"],  # join keys count as lineage
}
TABLES = {  # table -> {column: kind}; kind "key" is checked for uniqueness
    "raw_orders": {"order_id": "key", "amount": "num"},
    "raw_payments": {"order_id": "id", "paid": "num"},
    "stg_orders": {"order_id": "key", "amount": "num"},
    "stg_payments": {"order_id": "id", "paid": "num"},
    "mart_revenue": {"customer_id": "key", "revenue": "num"},
}
ORDER = list(TABLES)  # already topologically sorted
JOB_VERSION = {"stg_orders": "v1"}  # deployed transform versions
EVENTS = []  # change feed: dicts(day, kind, target, detail)
FIXED = {"late_files": set(), "deploys": set()}  # what an engineer has since fixed


def build(day, rng, src):
    """Run the whole pipeline for one partition. `src` holds source-side conditions."""
    n = int(1000 * rng.uniform(0.95, 1.05))
    raw_o = [{"order_id": f"{day}-{i}", "amount": rng.lognormvariate(3.5, 0.4)} for i in range(n)]
    keep = src.get("payments_fraction", 1.0)
    raw_p = [{"order_id": o["order_id"], "paid": o["amount"]} for o in raw_o if rng.random() < 0.9 * keep]
    stg_o = [dict(r) for r in raw_o]
    if JOB_VERSION["stg_orders"] == "v2":  # the bad deploy: dollars -> cents
        for r in stg_o:
            r["amount"] *= 100
    stg_p = [dict(r) for r in raw_p]
    paid = {p["order_id"] for p in stg_p}
    rev = {}
    for o in stg_o:
        if o["order_id"] in paid:
            c = hash(o["order_id"]) % 300
            rev[c] = rev.get(c, 0) + o["amount"]
    mart = [{"customer_id": str(c), "revenue": v} for c, v in rev.items()]
    return {"raw_orders": raw_o, "raw_payments": raw_p, "stg_orders": stg_o,
            "stg_payments": stg_p, "mart_revenue": mart}


# ----------------------------------------------------------------- profiling + baselines
def profile(table, rows):
    m = {f"{table}.__rows": float(len(rows))}
    for col, kind in TABLES[table].items():
        vals = [r[col] for r in rows if r[col] is not None]
        m[f"{table}.{col}.null_rate"] = 1 - len(vals) / max(len(rows), 1)
        if kind == "num" and vals:
            m[f"{table}.{col}.mean"] = statistics.fmean(vals)
        if kind == "key":
            m[f"{table}.{col}.dup_rate"] = 1 - len(set(vals)) / max(len(vals), 1)
    return m


class Baseline:
    """Learns each metric's normal range from recent *clean* partitions (median/MAD)."""
    def __init__(self, window=14, z=6.0, min_hist=7):
        self.hist, self.window, self.z, self.min_hist = {}, window, z, min_hist

    def check(self, metric, x):
        h = self.hist.get(metric, [])[-self.window:]
        if len(h) < self.min_hist:
            return None
        med = statistics.median(h)
        mad = 1.4826 * statistics.median(abs(v - med) for v in h)
        scale = max(mad, 0.03 * abs(med), 0.002)  # floor avoids hair-trigger on flat metrics
        z = (x - med) / scale
        return (z, med) if abs(z) > self.z else None

    def learn(self, metrics):
        for k, v in metrics.items():
            self.hist.setdefault(k, []).append(v)


# ----------------------------------------------------------------- lineage + culprit
def upstream_tables(node):
    return {u for u in LINEAGE.get(node, [])}


def trace_origin(node, anomalous_cols, path=()):
    """Walk column lineage upstream; origin = deepest anomalous column with no anomalous parent."""
    bad_parents = [u for u in LINEAGE.get(node, []) if u in anomalous_cols]
    if not bad_parents:
        return [(node, path + (node,))]
    out = []
    for u in bad_parents:
        out += trace_origin(u, anomalous_cols, path + (node,))
    return out


def rank_culprits(origin_table, metric_kind, day):
    """Score candidate causes by matching change events to the origin table's anomaly shape."""
    cands = []
    for e in EVENTS:
        if e["day"] != day or e["target"] != origin_table:
            continue
        fit = {"late_file": {"rows"}, "deploy": {"mean", "null_rate", "dup_rate", "rows"},
               "schema_change": {"null_rate", "mean"}}.get(e["kind"], set())
        score = 0.5 + (0.4 if metric_kind in fit else 0.0)
        cands.append({"kind": e["kind"], "detail": e["detail"], "score": round(score, 2)})
    if not cands:
        cands.append({"kind": "unknown", "detail": "no change event matches; likely upstream data drift", "score": 0.2})
    return sorted(cands, key=lambda c: -c["score"])


# ----------------------------------------------------------------- the agent
@dataclass
class Agent:
    outdir: pathlib.Path
    base: Baseline = field(default_factory=Baseline)
    quarantine: dict = field(default_factory=dict)  # (table, day) -> ticket id
    tickets: dict = field(default_factory=dict)
    warehouse: dict = field(default_factory=dict)  # day -> {table: rows}

    def observe(self, day, data):
        self.warehouse[day] = data
        metrics = {}
        for t in ORDER:
            metrics.update(profile(t, data[t]))
        anomalies = {}
        for k, x in metrics.items():
            hit = self.base.check(k, x)
            if hit:
                anomalies[k] = (x, *hit)
        if not anomalies:
            self.base.learn(metrics)  # only clean partitions teach the baseline
            return []
        return self.triage(day, anomalies)

    def triage(self, day, anomalies):
        # Anomalous *columns* drive lineage; row-count anomalies attach to the table's first column.
        cols = {}
        for k in anomalies:
            parts = k.split(".")
            if parts[1] == "__rows":  # row loss/gain taints every column derived from the table
                for c in TABLES[parts[0]]:
                    cols.setdefault(f"{parts[0]}.{c}", []).append(k)
            else:
                cols.setdefault(f"{parts[0]}.{parts[1]}", []).append(k)
        origins = {}
        for col in cols:
            for o, path in trace_origin(col, set(cols)):
                key = o.split(".")[0]  # one incident per origin table
                origins.setdefault(key, {"col": o, "paths": [], "impacted": set()})
                origins[key]["paths"].append(path)
                origins[key]["impacted"].add(col.split(".")[0])
        opened = []
        for o, info in origins.items():
            otable, o = o, info["col"]
            first = next(k for k in cols[o] if k in anomalies)
            kind = "rows" if "__rows" in first else first.split(".")[-1]
            culprits = rank_culprits(otable, kind, day)
            impacted = sorted(info["impacted"] | self.downstream(otable), key=ORDER.index)
            tid = f"DQ-{day:03d}-{otable}"
            for t in impacted:
                self.quarantine[(t, day)] = tid
            evidence = {k: {"observed": round(v[0], 3), "baseline_median": round(v[2], 3), "robust_z": round(v[1], 1)}
                        for k, v in anomalies.items() if k.split(".")[0] in info["impacted"]}
            self.tickets[tid] = {
                "id": tid, "day": day, "origin_column": o, "hypothesis": culprits[0],
                "alternatives": culprits[1:], "lineage_paths": sorted({" <- ".join(p) for p in info["paths"]}),
                "quarantined": impacted, "evidence": evidence,
                "proposed_action": f"Backfill partition day={day} for {impacted} after fix",
                "status": "AWAITING_APPROVAL"}
            self.write_ticket(self.tickets[tid])
            opened.append(tid)
        return opened

    @staticmethod
    def downstream(table):
        down, frontier = set(), {table}
        while frontier:
            nxt = {c.split(".")[0] for c, ups in LINEAGE.items() if any(u.split(".")[0] in frontier for u in ups)} - down
            down |= nxt
            frontier = nxt
        return down

    def write_ticket(self, t):
        self.outdir.mkdir(exist_ok=True)
        h = t["hypothesis"]
        md = [f"# {t['id']}  [{t['status']}]", "",
              f"**Root-cause hypothesis ({h['score']:.0%} confidence):** `{h['kind']}` - {h['detail']}", "",
              f"**Origin column:** `{t['origin_column']}`  |  **Partition:** day {t['day']}", "",
              "**Lineage path(s):** " + "; ".join(f"`{p}`" for p in t["lineage_paths"]), "",
              f"**Quarantined:** {', '.join(t['quarantined'])}", "", "**Evidence**", "",
              "| metric | observed | baseline | robust z |", "|---|---|---|---|"]
        md += [f"| {k} | {v['observed']} | {v['baseline_median']} | {v['robust_z']} |" for k, v in t["evidence"].items()]
        if t["alternatives"]:
            md += ["", "**Alternatives:** " + "; ".join(f"{a['kind']} ({a['score']})" for a in t["alternatives"])]
        md += ["", f"**Proposed action:** {t['proposed_action']} - requires human approval."]
        (self.outdir / f"{t['id']}.md").write_text("\n".join(md), encoding="utf-8")

    # --- human-gated remediation: the agent can propose, never execute without approval
    def backfill(self, tid, approved_by, rng):
        t = self.tickets[tid]
        if not approved_by:
            raise PermissionError("backfill requires human approval")
        day = t["day"]
        data = build(day, rng, {})  # rerun with source/deploy fixed
        m = {}
        for tb in ORDER:
            m.update(profile(tb, data[tb]))
        still_bad = [k for k, x in m.items() if self.base.check(k, x)]
        if still_bad:
            t["status"] = f"BACKFILL_FAILED_VALIDATION {still_bad}"
            return False
        self.warehouse[day] = data
        self.base.learn(m)
        for tb in t["quarantined"]:
            self.quarantine.pop((tb, day), None)
        t["status"] = f"RESOLVED (backfill approved by {approved_by})"
        self.write_ticket(t)
        return True


# ----------------------------------------------------------------- demo
def main():
    rng = random.Random(7)
    agent = Agent(outdir=pathlib.Path("tickets"))
    for day in range(1, 31):
        src = {}
        if day == 27:  # incident 1: payments file arrives late
            src["payments_fraction"] = 0.35
            EVENTS.append({"day": 27, "kind": "late_file", "target": "raw_payments",
                           "detail": "payments_20261013.csv expected 02:00, not landed by 06:00 SLA (partial load)"})
        if day == 29:  # incident 2: bad deploy
            JOB_VERSION["stg_orders"] = "v2"
            EVENTS.append({"day": 29, "kind": "deploy", "target": "stg_orders",
                           "detail": "stg_orders transform v1 -> v2 (PR #4412 'normalize currency units')"})
        opened = agent.observe(day, build(day, rng, src))
        if day == 29:
            JOB_VERSION["stg_orders"] = "v1"  # engineer rolls back after reading the ticket
        for tid in opened:
            t = agent.tickets[tid]
            print(f"[day {day}] ANOMALY -> {tid}: {t['hypothesis']['kind']} ({t['hypothesis']['score']:.0%}); "
                  f"quarantined {t['quarantined']}")
    print("\nQuarantine:", {f"{t}@{d}": v for (t, d), v in agent.quarantine.items()})
    print("Agent will NOT backfill on its own. Pending tickets:", [k for k, v in agent.tickets.items() if v["status"].startswith("AWAIT")])
    try:
        agent.backfill("DQ-029-stg_orders", None, rng)
    except PermissionError as e:
        print("Unapproved backfill attempt blocked:", e)
    for tid in list(agent.tickets):
        ok = agent.backfill(tid, "oncall@example.com", rng)
        print(f"Human approved {tid}: backfill {'validated, quarantine released' if ok else 'FAILED validation'}")
    print("Quarantine now:", agent.quarantine)


if __name__ == "__main__":
    main()
