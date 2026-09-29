"""Weekly shift roster with Google OR-Tools CP-SAT.

Usage:
    python roster_model.py [roster_preferences.csv]     # solve + validate
    python roster_model.py --selftest [csv]             # validator/edge-case tests
"""
import sys
import time
import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHIFTS = ["Morning", "Evening", "Night"]
START_HOUR = {"Morning": 7, "Evening": 15, "Night": 23}
SHIFT_LEN = 8
COVER = {  # employees required per shift
    "weekday": {"Morning": 4, "Evening": 3, "Night": 2},
    "weekend": {"Morning": 3, "Evening": 2, "Night": 2},
}
MIN_REST_H = 11

# ---- tunable choices (explained in the write-up) ----
W_PREF = 1               # weight per preference point (-2..+2)
W_WEEKEND = 3            # penalty per shift of (max - min) weekend load
REQUIRE_SENIOR_NIGHT = True   # ADDED: >=1 Senior on every Night shift
WRAP_WEEK = False        # False: week is isolated (Sun -> next Mon not checked)


def demand(day, shift):
    return COVER["weekend" if day in ("Sat", "Sun") else "weekday"][shift]


def slot_start(d, s):   # absolute hour from Mon 00:00
    return 24 * DAYS.index(d) + START_HOUR[s]


def slot_end(d, s):
    return slot_start(d, s) + SHIFT_LEN


def load(path):
    df = pd.read_csv(path)
    assert len(df) == 420, f"expected 420 rows, got {len(df)}"
    assert not df.duplicated(["employee_id", "day", "shift"]).any(), "duplicate rows"
    assert df.day.isin(DAYS).all() and df["shift"].isin(SHIFTS).all()
    assert df.available.isin([0, 1]).all() and df.preference.between(-2, 2).all()
    return df


def conflicting_pairs():
    """All ordered slot pairs (a, b) with a earlier than b whose gap < MIN_REST_H.
    Derived from clock times, so the 'Night->Morning' rule falls out automatically
    (gap 0h), alongside Evening->Morning (8h) and Night->Evening (8h)."""
    slots = [(d, s) for d in DAYS for s in SHIFTS]
    pairs = []
    for a in slots:
        for b in slots:
            if slot_start(*b) > slot_start(*a) and slot_start(*b) - slot_end(*a) < MIN_REST_H:
                pairs.append((a, b))
    if WRAP_WEEK:  # Sun slot vs following Mon slot (same week template repeated)
        for a in [("Sun", s) for s in SHIFTS]:
            for b in [("Mon", s) for s in SHIFTS]:
                if 168 + slot_start(*b) - slot_end(*a) < MIN_REST_H:
                    pairs.append((a, b))
    return pairs


def precheck(df):
    """Cheap infeasibility diagnostics -> list of human-readable problems."""
    problems = []
    av = df[df.available == 1]
    for d in DAYS:
        for s in SHIFTS:
            n = ((av.day == d) & (av["shift"] == s)).sum()
            if n < demand(d, s):
                problems.append(f"{d} {s}: need {demand(d, s)}, only {n} available")
            if REQUIRE_SENIOR_NIGHT and s == "Night":
                k = ((av.day == d) & (av["shift"] == s) & (av.seniority == "Senior")).sum()
                if k < 1:
                    problems.append(f"{d} Night: no available Senior")
    need = sum(demand(d, s) for d in DAYS for s in SHIFTS)
    cap = df.drop_duplicates("employee_id").max_shifts_per_week.sum()
    if cap < need:
        problems.append(f"total capacity {cap} < total demand {need}")
    return problems


def solve(df, time_limit=30, seed=0):
    from ortools.sat.python import cp_model
    emps = df.drop_duplicates("employee_id").set_index("employee_id")
    E = list(emps.index)
    info = df.set_index(["employee_id", "day", "shift"])

    m = cp_model.CpModel()
    x = {(e, d, s): m.NewBoolVar(f"x_{e}_{d}_{s}") for e in E for d in DAYS for s in SHIFTS}

    for d in DAYS:                                   # coverage (exact)
        for s in SHIFTS:
            m.Add(sum(x[e, d, s] for e in E) == demand(d, s))
            if REQUIRE_SENIOR_NIGHT and s == "Night":
                m.Add(sum(x[e, d, s] for e in E if emps.loc[e, "seniority"] == "Senior") >= 1)
    for e in E:
        m.Add(sum(x[e, d, s] for d in DAYS for s in SHIFTS) <= int(emps.loc[e, "max_shifts_per_week"]))
        for d in DAYS:                               # <=1 shift / day
            m.Add(sum(x[e, d, s] for s in SHIFTS) <= 1)
            for s in SHIFTS:                         # availability
                if info.loc[(e, d, s), "available"] == 0:
                    m.Add(x[e, d, s] == 0)
        for (da, sa), (db, sb) in conflicting_pairs():  # rest / N->M
            m.AddBoolOr([x[e, da, sa].Not(), x[e, db, sb].Not()])

    wk = {e: m.NewIntVar(0, 2, f"wk_{e}") for e in E}    # weekend shifts per employee
    for e in E:
        m.Add(wk[e] == sum(x[e, d, s] for d in ("Sat", "Sun") for s in SHIFTS))
    wmax, wmin = m.NewIntVar(0, 2, "wmax"), m.NewIntVar(0, 2, "wmin")
    for e in E:
        m.Add(wmax >= wk[e])
        m.Add(wmin <= wk[e])

    pref = sum(int(info.loc[(e, d, s), "preference"]) * x[e, d, s]
               for e in E for d in DAYS for s in SHIFTS)
    m.Maximize(W_PREF * pref - W_WEEKEND * (wmax - wmin))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = 8
    t0 = time.time()
    status = solver.Solve(m)
    dt = time.time() - t0
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return name, None, None, dt
    sol = {k for k, v in x.items() if solver.Value(v)}
    return name, sol, solver.ObjectiveValue(), dt


def validate(df, sol):
    """Independent re-check of every rule from raw data (no model code reused,
    except the coverage table and shift clock times)."""
    errs = []
    emps = df.drop_duplicates("employee_id").set_index("employee_id")
    info = df.set_index(["employee_id", "day", "shift"])
    for d in DAYS:
        for s in SHIFTS:
            got = [e for (e, dd, ss) in sol if dd == d and ss == s]
            if len(got) != demand(d, s):
                errs.append(f"coverage {d} {s}: {len(got)} != {demand(d, s)}")
            if REQUIRE_SENIOR_NIGHT and s == "Night" and not any(emps.loc[e, "seniority"] == "Senior" for e in got):
                errs.append(f"no senior on {d} Night")
    for e in emps.index:
        mine = sorted([(d, s) for (ee, d, s) in sol if ee == e], key=lambda t: slot_start(*t))
        if len(mine) > emps.loc[e, "max_shifts_per_week"]:
            errs.append(f"{e}: {len(mine)} shifts > cap")
        for d, s in mine:
            if info.loc[(e, d, s), "available"] == 0:
                errs.append(f"{e} assigned to unavailable {d} {s}")
        for i, (d, s) in enumerate(mine):
            for d2, s2 in mine[i + 1:]:
                if d == d2:
                    errs.append(f"{e}: two shifts on {d}")
                gap = slot_start(d2, s2) - slot_end(d, s)
                if gap < MIN_REST_H:
                    errs.append(f"{e}: only {gap}h rest between {d} {s} and {d2} {s2}")
                if s == "Night" and s2 == "Morning" and DAYS.index(d2) == DAYS.index(d) + 1:
                    errs.append(f"{e}: Night->Morning {d}->{d2}")
    return errs


def report(df, sol):
    emps = df.drop_duplicates("employee_id").set_index("employee_id")
    info = df.set_index(["employee_id", "day", "shift"])
    print("\nRoster (names per shift):")
    for d in DAYS:
        for s in SHIFTS:
            names = sorted(emps.loc[e, "name"].split()[0] for (e, dd, ss) in sol if dd == d and ss == s)
            print(f"  {d} {s:8s}: {', '.join(names)}")
    pts = [info.loc[k, "preference"] for k in sol]
    print(f"\nPreference sum: {sum(pts)} | positive: {sum(p > 0 for p in pts)} "
          f"neutral: {sum(p == 0 for p in pts)} negative: {sum(p < 0 for p in pts)}")
    wk = {e: sum(1 for (ee, d, s) in sol if ee == e and d in ('Sat', 'Sun')) for e in emps.index}
    tot = {e: sum(1 for (ee, *_ ) in sol if ee == e) for e in emps.index}
    print(f"Weekend shifts per employee: min {min(wk.values())} max {max(wk.values())}")
    print(f"Total shifts per employee:   min {min(tot.values())} max {max(tot.values())}")


def selftest(df):
    """Negative tests: corrupt a valid roster / the data and confirm detection."""
    name, sol, _, _ = solve(df, time_limit=10)
    assert sol and not validate(df, sol), "baseline roster must validate"
    ok = True

    def expect(label, errs, kw):
        nonlocal ok
        hit = any(kw in e for e in errs)
        ok &= hit
        print(f"  [{'PASS' if hit else 'FAIL'}] {label}")

    emps = list(df.employee_id.unique())
    # 1 remove someone from a shift -> coverage error
    bad = set(sol); bad.discard(next(k for k in bad if k[1] == "Mon" and k[2] == "Morning"))
    expect("missing coverage detected", validate(df, bad), "coverage")
    # 2 Night then Morning next day
    e = emps[0]
    bad = {k for k in sol if k[0] != e} | {(e, "Tue", "Night"), (e, "Wed", "Morning")}
    expect("Night->Morning detected", validate(df, bad), "Night->Morning")
    # 3 Evening then Morning (8h rest)
    bad = {k for k in sol if k[0] != e} | {(e, "Tue", "Evening"), (e, "Wed", "Morning")}
    expect("<11h rest detected", validate(df, bad), "rest")
    # 4 two shifts one day
    bad = {k for k in sol if k[0] != e} | {(e, "Tue", "Morning"), (e, "Tue", "Night")}
    expect("two shifts/day detected", validate(df, bad), "two shifts")
    # 5 over weekly cap
    cap = int(df[df.employee_id == e].max_shifts_per_week.iloc[0])
    bad = {k for k in sol if k[0] != e} | {(e, d, "Morning") for d in DAYS[:cap]} | {(e, DAYS[cap], "Evening")}
    expect("weekly cap detected", validate(df, bad), "cap")
    # 6 unavailable slot
    row = df[(df.employee_id == e) & (df.available == 0)].iloc[0]
    bad = {k for k in sol if k[0] != e} | {(e, row.day, row["shift"])}
    expect("availability detected", validate(df, bad), "unavailable")
    # 7 infeasible instance: nobody available Sat Night -> solver must say INFEASIBLE, precheck explains
    d2 = df.copy(); d2.loc[(d2.day == "Sat") & (d2["shift"] == "Night"), "available"] = 0
    st, s2, _, _ = solve(d2, time_limit=10)
    hit = st == "INFEASIBLE" and any("Sat Night" in p for p in precheck(d2))
    ok &= hit; print(f"  [{'PASS' if hit else 'FAIL'}] infeasible instance -> {st} + diagnosis")
    # 8 no seniors available on a night -> senior rule (not coverage) is the culprit
    d3 = df.copy(); d3.loc[(d3.day == "Wed") & (d3["shift"] == "Night") & (d3.seniority == "Senior"), "available"] = 0
    hit = any("no available Senior" in p for p in precheck(d3))
    ok &= hit; print(f"  [{'PASS' if hit else 'FAIL'}] senior-night infeasibility diagnosed")
    print("ALL PASSED" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "roster_preferences.csv"
    df = load(path)
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest(df) else 1)
    probs = precheck(df)
    if probs:
        print("Infeasible before solving:\n  " + "\n  ".join(probs)); sys.exit(1)
    status, sol, obj, dt = solve(df)
    print(f"Status: {status} | objective: {obj} | solve time: {dt:.2f}s")
    if sol is None:
        print("No roster found. Try REQUIRE_SENIOR_NIGHT=False or relax availability."); sys.exit(1)
    errs = validate(df, sol)
    print("Validation:", "OK - all hard constraints hold" if not errs else errs)
    report(df, sol)
