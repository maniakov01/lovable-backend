"""
optimizer.py — Fleet Transfer Scheduling
Production backend module for FastAPI / Render deployment.

Solver : PuLP + CBC (free, bundled with pulp)
Inputs : fleet_scheduling_data_v2.xlsx  (co-located or via DATA_PATH env var)
Outputs: JSON-serializable dict  {kpis, tables}
"""

import os
import math
from collections import defaultdict
from itertools import combinations

import pandas as pd
import pulp


# ── Constants ──────────────────────────────────────────────────────────────────
AVG_FULL_LOAD_LBS = 26 * 1445   # 26 pallets × ~1,445 lbs — full-truck baseline
UNSERVED_PENALTY  = 100_000     # $ penalty per unserved job
DATA_FILENAME     = "fleet_scheduling_data_v2.xlsx"


# ── Data loading ───────────────────────────────────────────────────────────────

def _locate_data_file():
    """
    Find the Excel input file. Search order:
      1. DATA_PATH environment variable (set on Render)
      2. Directory of this module file
      3. Current working directory
    """
    env_path = os.environ.get("DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILENAME),
        os.path.join(os.getcwd(), DATA_FILENAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find {DATA_FILENAME}. "
        f"Set the DATA_PATH environment variable or place the file alongside optimizer.py. "
        f"Searched: {candidates}"
    )


def _load_data(week, scenario):
    """
    Load and filter Excel data.

    Parameters
    ----------
    week     : int or None  — 1-indexed week number (1 = days 1-7, 2 = days 8-14, …)
                              None → use all days in the file
    scenario : str or None  — reserved for future scenario variants; currently unused
    """
    path = _locate_data_file()

    # Plants — exclude DC row (DC has no travel_time_min)
    df_plants = pd.read_excel(path, sheet_name="Plants")
    df_plants = df_plants[df_plants["travel_time_min"].notna()].copy()

    # Days — filter to requested week
    df_days_all = pd.read_excel(path, sheet_name="Days")
    if week is not None:
        week = int(week)
        start_row = (week - 1) * 7
        end_row   = start_row + 7
        df_days   = df_days_all.iloc[start_row:end_row].copy()
    else:
        df_days = df_days_all.copy()

    active_day_ids = df_days["day_id"].tolist()

    # Jobs — only jobs on active days
    df_jobs_all = pd.read_excel(path, sheet_name="Jobs")
    df_jobs     = df_jobs_all[df_jobs_all["day_id"].isin(active_day_ids)].reset_index(drop=True)

    # Params — vertical key/value table
    df_params = pd.read_excel(
        path, sheet_name="Params",
        header=None, names=["parameter", "value"], skiprows=1
    )
    df_params = df_params[df_params["value"].notna()].copy()
    params    = dict(zip(df_params["parameter"], df_params["value"]))

    return df_plants, df_days, df_jobs, params


# ── Helper ─────────────────────────────────────────────────────────────────────

def _minutes_to_hhmm(minutes):
    minutes = int(round(minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ── Main entry point ───────────────────────────────────────────────────────────

def run_optimization(week=None, scenario=None):
    """
    Build and solve the fleet scheduling MILP, return JSON-serializable results.

    Parameters
    ----------
    week     : int or None   1 = first 7 days, 2 = second 7 days, etc. None = full dataset
    scenario : str or None   reserved for future use

    Returns
    -------
    dict with keys:
        "kpis"   : {total_cost, trips_scheduled, unserved_jobs, trucks_used,
                    fleet_utilization_pct, total_pallets, total_weight_lb,
                    avg_pallets_per_trip, savings_pct, solver_status}
        "tables" : {schedule: [...], cost_breakdown: [...]}
    """

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df_plants, df_days, df_jobs, params = _load_data(week, scenario)

    plants      = df_plants["plant_id"].tolist()
    travel_time = dict(zip(df_plants["plant_id"], df_plants["travel_time_min"]))
    load_time   = dict(zip(df_plants["plant_id"], df_plants["load_time_min"]))

    days       = df_days["day_id"].tolist()
    is_weekend = dict(zip(df_days["day_id"], df_days["is_weekend"]))
    day_to_date = dict(zip(df_days["day_id"], pd.to_datetime(df_days["date"])))

    # Build job tuples
    all_jobs      = []
    job_id_lookup = {}
    job_window    = {}
    job_pallets   = {}
    job_weight    = {}
    seq_counter   = {}

    for _, row in df_jobs.iterrows():
        plant = row["plant_id"]
        day   = int(row["day_id"])
        key   = (plant, day)
        seq_counter[key] = seq_counter.get(key, 0) + 1
        seq = seq_counter[key]
        jt  = (plant, day, seq)

        all_jobs.append(jt)
        job_id_lookup[jt] = row["job_id"]
        job_pallets[jt]   = int(row["pallets"])
        job_weight[jt]    = int(row["weight_lb"])

        if pd.notna(row.get("earliest_start_min")) and pd.notna(row.get("latest_start_min")):
            job_window[jt] = (int(row["earliest_start_min"]), int(row["latest_start_min"]))

    # Operational parameters
    UNLOAD_TIME     = int(params["unload_time_min"])
    NUM_TRUCKS      = int(params["truck_count_bus_schedule"])
    COST_PER_HOUR   = float(params["hourly_cost_total_usd"])
    WEEKEND_PENALTY = float(params["weekend_cost_premium"])

    WD_START = int(params["weekday_start_min"])   # 420  = 07:00
    WD_END   = int(params["weekday_end_min"])     # 1380 = 23:00
    WE_START = int(params["weekend_start_min"])   # 420  = 07:00
    WE_END   = int(params["weekend_end_min"])     # 1080 = 18:00

    trucks = list(range(1, NUM_TRUCKS + 1))

    # ── 2. Derived parameters ─────────────────────────────────────────────────
    duration = {
        p: 2 * travel_time[p] + load_time[p] + UNLOAD_TIME
        for p in plants
    }

    trip_cost_base = {
        p: COST_PER_HOUR * (duration[p] / 60.0)
        for p in plants
    }

    trip_cost_per_job = {
        jt: trip_cost_base[jt[0]] * job_weight[jt] / AVG_FULL_LOAD_LBS
        for jt in all_jobs
    }

    max_trips_per_day = max(
        (WD_END - WD_START) // duration[p] for p in plants
    )

    BIG_M = WD_END  # upper bound on any start time (minutes from midnight)

    # Job pairs per day for sequencing constraints
    jobs_on_day = defaultdict(list)
    for jt in all_jobs:
        jobs_on_day[jt[1]].append(jt)

    # Flat 7-tuples: (truck, p1, d1, s1, p2, d2, s2)
    seq_pairs = []
    for day, job_list in jobs_on_day.items():
        for (p1, d1, s1), (p2, d2, s2) in combinations(job_list, 2):
            for truck in trucks:
                seq_pairs.append((truck, p1, d1, s1, p2, d2, s2))

    weekend_days = [d for d in days if is_weekend[d]]

    # ── 3. Build PuLP model ───────────────────────────────────────────────────
    model = pulp.LpProblem("FleetScheduling", pulp.LpMinimize)

    # x[truck, plant, day, seq] — binary assignment
    x = {
        (truck, p, d, s): pulp.LpVariable(
            f"x_{truck}_{p}_{d}_{s}", cat="Binary"
        )
        for truck in trucks
        for (p, d, s) in all_jobs
    }

    # s_var[truck, plant, day, seq] — continuous start time (minutes)
    s_var = {
        (truck, p, d, s): pulp.LpVariable(
            f"s_{truck}_{p}_{d}_{s}",
            lowBound=WD_START, upBound=WD_END,
            cat="Continuous"
        )
        for truck in trucks
        for (p, d, s) in all_jobs
    }

    # z[truck, p1, d1, s1, p2, d2, s2] — ordering binary
    z = {
        (truck, p1, d1, s1, p2, d2, s2): pulp.LpVariable(
            f"z_{truck}_{p1}_{d1}_{s1}_{p2}_{d2}_{s2}", cat="Binary"
        )
        for (truck, p1, d1, s1, p2, d2, s2) in seq_pairs
    }

    # u[plant, day, seq] — unserved flag
    u = {
        (p, d, s): pulp.LpVariable(f"u_{p}_{d}_{s}", cat="Binary")
        for (p, d, s) in all_jobs
    }

    # W[truck, day] — weekend activation
    W = {
        (truck, day): pulp.LpVariable(f"W_{truck}_{day}", cat="Binary")
        for truck in trucks
        for day in weekend_days
    }

    # ── Objective ─────────────────────────────────────────────────────────────
    model += (
        pulp.lpSum(
            trip_cost_per_job[p, d, s] * x[truck, p, d, s]
            for truck in trucks
            for (p, d, s) in all_jobs
        )
        + pulp.lpSum(
            WEEKEND_PENALTY * W[truck, day]
            for truck in trucks
            for day in weekend_days
        )
        + pulp.lpSum(
            UNSERVED_PENALTY * u[p, d, s]
            for (p, d, s) in all_jobs
        )
    )

    # ── Constraint 1: Every job served by exactly one truck or unserved ───────
    for (p, d, s) in all_jobs:
        model += (
            pulp.lpSum(x[truck, p, d, s] for truck in trucks) + u[p, d, s] == 1,
            f"C1_assign_{p}_{d}_{s}"
        )

    # ── Constraint 2a: Start time lower bound (within operating window) ───────
    for truck in trucks:
        for (p, d, s) in all_jobs:
            window_start = WE_START if is_weekend[d] else WD_START
            model += (
                s_var[truck, p, d, s] >= window_start * x[truck, p, d, s],
                f"C2a_lb_{truck}_{p}_{d}_{s}"
            )

    # ── Constraint 2b: Start time upper bound (must finish within window) ─────
    for truck in trucks:
        for (p, d, s) in all_jobs:
            window_end = WE_END if is_weekend[d] else WD_END
            latest = window_end - duration[p]
            model += (
                s_var[truck, p, d, s] <= latest + BIG_M * (1 - x[truck, p, d, s]),
                f"C2b_ub_{truck}_{p}_{d}_{s}"
            )

    # ── Constraint 3: No-overlap — Big-M sequencing ───────────────────────────
    for (truck, p1, d1, s1, p2, d2, s2) in seq_pairs:
        zv   = z[truck, p1, d1, s1, p2, d2, s2]
        x1   = x[truck, p1, d1, s1]
        x2   = x[truck, p2, d2, s2]
        sv1  = s_var[truck, p1, d1, s1]
        sv2  = s_var[truck, p2, d2, s2]

        # If z=1 (job1 before job2): sv2 >= sv1 + dur1
        model += (
            sv2 >= sv1 + duration[p1]
            - BIG_M * (1 - zv)
            - BIG_M * (1 - x1)
            - BIG_M * (1 - x2),
            f"C3a_{truck}_{p1}_{d1}_{s1}_{p2}_{d2}_{s2}"
        )

        # If z=0 (job2 before job1): sv1 >= sv2 + dur2
        model += (
            sv1 >= sv2 + duration[p2]
            - BIG_M * zv
            - BIG_M * (1 - x1)
            - BIG_M * (1 - x2),
            f"C3b_{truck}_{p1}_{d1}_{s1}_{p2}_{d2}_{s2}"
        )

    # ── Constraint 4: Weekend activation flag ─────────────────────────────────
    for truck in trucks:
        for day in weekend_days:
            jobs_today = [(p, d, s) for (p, d, s) in all_jobs if d == day]
            if jobs_today:
                model += (
                    pulp.lpSum(x[truck, p, d, s] for (p, d, s) in jobs_today)
                    <= max_trips_per_day * W[truck, day],
                    f"C4_weekend_{truck}_{day}"
                )

    # ── 4. Solve ──────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        gapRel=0.01,    # stop within 1% of optimal
        timeLimit=300,  # 5-minute cap
        threads=4,
    )
    result = model.solve(solver)

    solver_status = pulp.LpStatus[model.status]

    # ── 5. Extract results ────────────────────────────────────────────────────
    schedule_rows = []

    for truck in trucks:
        for (plant, day, seq) in all_jobs:
            if pulp.value(x[truck, plant, day, seq]) is not None and \
               pulp.value(x[truck, plant, day, seq]) > 0.5:

                jt        = (plant, day, seq)
                start_min = pulp.value(s_var[truck, plant, day, seq])
                if start_min is None:
                    start_min = WD_START if not is_weekend[day] else WE_START

                finish_min = start_min + duration[plant]

                schedule_rows.append({
                    "day_id"         : int(day),
                    "date"           : day_to_date[day].strftime("%Y-%m-%d"),
                    "truck_id"       : int(truck),
                    "plant_id"       : str(plant),
                    "job_id"         : str(job_id_lookup.get(jt, f"{plant}_D{day}_J{seq}")),
                    "pallets"        : int(job_pallets[jt]),
                    "weight_lb"      : int(job_weight[jt]),
                    "start_time_min" : int(round(start_min)),
                    "start_time"     : _minutes_to_hhmm(start_min),
                    "finish_time_min": int(round(finish_min)),
                    "finish_time"    : _minutes_to_hhmm(finish_min),
                    "duration_min"   : int(duration[plant]),
                    "trip_cost_usd"  : round(float(trip_cost_per_job[jt]), 2),
                    "is_weekend"     : bool(is_weekend[day]),
                })

    # Sort by day, truck, start time
    schedule_rows.sort(key=lambda r: (r["day_id"], r["truck_id"], r["start_time_min"]))

    # KPI calculations
    unserved_count = sum(
        1 for (p, d, s) in all_jobs
        if pulp.value(u[p, d, s]) is not None and pulp.value(u[p, d, s]) > 0.5
    )

    weekend_activations = sum(
        1 for truck in trucks for day in weekend_days
        if pulp.value(W[truck, day]) is not None and pulp.value(W[truck, day]) > 0.5
    )

    trucks_used   = len({r["truck_id"] for r in schedule_rows})
    trips_total   = len(schedule_rows)
    total_pallets = sum(r["pallets"]   for r in schedule_rows)
    total_weight  = sum(r["weight_lb"] for r in schedule_rows)
    obj_val       = float(pulp.value(model.objective)) if pulp.value(model.objective) is not None else 0.0

    # Trip cost broken down by plant
    trip_cost_by_plant = {}
    for plant in plants:
        trip_cost_by_plant[plant] = sum(
            r["trip_cost_usd"]
            for r in schedule_rows
            if r["plant_id"] == plant
        )

    total_trip_cost  = sum(trip_cost_by_plant.values())
    weekend_cost     = sum(
        WEEKEND_PENALTY * (pulp.value(W[t, d]) or 0)
        for t in trucks for d in weekend_days
    )
    unserved_cost    = unserved_count * UNSERVED_PENALTY

    # savings_pct: reduction vs naive baseline (every job done solo, no batching)
    # Baseline = all jobs served at full-truck cost, no sequencing optimisation
    baseline_cost = sum(trip_cost_base[p] for (p, d, s) in all_jobs)
    savings_pct   = round((1 - total_trip_cost / baseline_cost) * 100, 2) if baseline_cost > 0 else 0.0

    # Cost breakdown table
    cost_rows = [
        {"category": "Trip Cost",          "plant": "Montreal_1", "amount_usd": round(trip_cost_by_plant.get("Montreal_1", 0.0), 2)},
        {"category": "Trip Cost",          "plant": "Montreal_2", "amount_usd": round(trip_cost_by_plant.get("Montreal_2", 0.0), 2)},
        {"category": "Trip Cost",          "plant": "Laval",      "amount_usd": round(trip_cost_by_plant.get("Laval",       0.0), 2)},
        {"category": "Trip Cost",          "plant": "Terrebonne", "amount_usd": round(trip_cost_by_plant.get("Terrebonne",  0.0), 2)},
        {"category": "Weekend Penalty",    "plant": "All",        "amount_usd": round(weekend_cost,   2)},
        {"category": "Unserved Penalty",   "plant": "All",        "amount_usd": round(unserved_cost,  2)},
        {"category": "Total",              "plant": "All",        "amount_usd": round(obj_val,        2)},
    ]

    # ── 6. Return structured result ───────────────────────────────────────────
    return {
        "kpis": {
            "total_cost"            : round(obj_val, 2),
            "total_trip_cost"       : round(total_trip_cost, 2),
            "savings_pct"           : float(savings_pct),
            "trips_scheduled"       : int(trips_total),
            "unserved_jobs"         : int(unserved_count),
            "trucks_used"           : int(trucks_used),
            "trucks_available"      : int(NUM_TRUCKS),
            "fleet_utilization_pct" : round(trucks_used / NUM_TRUCKS * 100, 1),
            "total_pallets"         : int(total_pallets),
            "total_weight_lb"       : int(total_weight),
            "avg_pallets_per_trip"  : round(total_pallets / trips_total, 1) if trips_total > 0 else 0.0,
            "weekend_truck_days"    : int(weekend_activations),
            "solver_status"         : solver_status,
            "week"                  : int(week) if week is not None else None,
            "scenario"              : str(scenario) if scenario is not None else None,
        },
        "tables": {
            "schedule"        : schedule_rows,
            "cost_breakdown"  : cost_rows,
        },
    }
