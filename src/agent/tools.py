"""Agent tools: plain Python functions over Spark SQL. Call init(spark, catalog, schema) first."""
import json
from datetime import date, datetime, timedelta

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression

import ground_truth as gt  # only constants + holdout helper; the agent never sees the true coefficients

_S = {"spark": None, "fq": None, "model": None, "run_ts": None}
SOURCES = sorted(gt.SOURCE_EFFECT)


def init(spark, catalog, schema):
    _S.update(spark=spark, fq=f"{catalog}.{schema}", run_ts=datetime.utcnow(), model=None)


def _sql(q):
    return _S["spark"].sql(q.replace("{fq}", _S["fq"])).toPandas()


def _cutoff():
    return (date.today() - timedelta(days=gt.OPEN_WINDOW_DAYS)).isoformat()


@mlflow.trace(span_type="TOOL")
def get_pipeline_summary():
    """Open leads per location/status plus the historical conversion rate."""
    pipe = _sql("SELECT assigned_location_id AS location_id, status, count(*) AS n FROM {fq}.leads "
                f"WHERE created_date >= '{_cutoff()}' GROUP BY 1, 2 ORDER BY 1, 2")
    hist = _sql("SELECT avg(cast(converted_flag AS int)) AS rate, count(*) AS n, "
                "avg(first_response_hours) AS avg_resp FROM {fq}.leads "
                f"WHERE created_date < '{_cutoff()}'")
    wide = pipe.pivot(index="location_id", columns="status", values="n").fillna(0).astype(int)
    return {"historic_conversion_rate": round(float(hist.rate[0]), 4),
            "closed_leads": int(hist.n[0]),
            "avg_first_response_hours_historic": round(float(hist.avg_resp[0]), 1),
            "open_leads_total": int(pipe.n.sum()),
            "open_leads_by_location": wide.reset_index().to_dict(orient="records")}


_FACTOR_SQL = {
    "source": "source",
    "response_bucket": "CASE WHEN first_response_hours < 2 THEN '0-2h' WHEN first_response_hours < 8 "
                       "THEN '2-8h' WHEN first_response_hours < 24 THEN '8-24h' WHEN first_response_hours < 48 "
                       "THEN '24-48h' ELSE '48h+' END",
    "touchpoints": "cast(num_touchpoints AS string)",
    "location": "assigned_location_id",
}


@mlflow.trace(span_type="TOOL")
def conversion_by_factor(factor):
    """Historic conversion rate on closed leads, grouped by one factor."""
    if factor not in _FACTOR_SQL:
        return {"error": f"factor must be one of {list(_FACTOR_SQL)}"}
    df = _sql(f"SELECT {_FACTOR_SQL[factor]} AS level, count(*) AS n, "
              "round(avg(cast(converted_flag AS int)), 4) AS conversion_rate FROM {fq}.leads "
              f"WHERE created_date < '{_cutoff()}' GROUP BY 1 ORDER BY 3 DESC")
    rates = df.conversion_rate
    return {"factor": factor, "spread_max_minus_min": round(float(rates.max() - rates.min()), 4),
            "levels": df.to_dict(orient="records")}


def _features(df):
    tp = df["num_touchpoints"].to_numpy()
    X = [df["first_response_hours"].to_numpy(), np.minimum(tp, 4), np.maximum(tp - 6, 0)]
    X += [(df["source"] == s).to_numpy().astype(float) for s in SOURCES[1:]]  # drop-first one-hot
    return np.column_stack(X).astype(float)


@mlflow.trace(span_type="TOOL")
def fit_conversion_model():
    """Logistic regression on closed leads; returns coefficients and caches the model."""
    df = _sql("SELECT source, first_response_hours, num_touchpoints, cast(converted_flag AS int) AS y "
              f"FROM {{fq}}.leads WHERE created_date < '{_cutoff()}'")
    m = LogisticRegression(C=100.0, max_iter=2000).fit(_features(df), df["y"])
    _S["model"] = m
    names = ["first_response_hours", "min(touchpoints,4)", "max(touchpoints-6,0)"] + \
            [f"source={s} (vs {SOURCES[0]})" for s in SOURCES[1:]]
    coefs = {n: round(float(c), 4) for n, c in zip(names, m.coef_[0])}
    return {"n_train": len(df), "intercept": round(float(m.intercept_[0]), 4), "coefficients": coefs,
            "note": "source and touchpoint history are fixed per lead; first_response_hours is set by "
                    "the front desk. A 24h delay changes log-odds by %.2f"
                    % (24 * coefs["first_response_hours"])}


def score_open(location_id=None):
    """Open leads with model p(current response), p(1h response) and uplift."""
    if _S["model"] is None:
        fit_conversion_model()
    where = f"created_date >= '{_cutoff()}'"
    if location_id:
        where += f" AND assigned_location_id = '{location_id}'"
    df = _sql("SELECT lead_id, assigned_location_id AS location_id, source, first_response_hours, "
              f"num_touchpoints FROM {{fq}}.leads WHERE {where}")
    m = _S["model"]
    df["p_current"] = m.predict_proba(_features(df))[:, 1]
    fast = df.copy()
    fast["first_response_hours"] = gt.TREATED_RESPONSE_HOURS
    df["p_fast"] = m.predict_proba(_features(fast))[:, 1]
    df["uplift"] = df["p_fast"] - df["p_current"]
    return df


@mlflow.trace(span_type="TOOL")
def rank_open_leads(strategy="uplift", location_id=None, top_n=10):
    """Rank open leads by uplift (p(1h response) - p(current)) or by likelihood (p(current))."""
    key = {"uplift": "uplift", "likelihood": "p_current"}.get(strategy)
    if key is None:
        return {"error": "strategy must be 'uplift' or 'likelihood'"}
    df = score_open(location_id).sort_values(key, ascending=False)
    top = df.head(int(top_n or 10))
    return {"strategy": strategy, "location_id": location_id, "n_open": len(df),
            "sum_uplift_top_n": round(float(top.uplift.sum()), 3),
            "top": top[["lead_id", "location_id", "source", "first_response_hours", "num_touchpoints",
                        "p_current", "uplift"]].round(3).to_dict(orient="records")}


def capacity_map():
    df = _sql("SELECT location_id, capacity_patients_per_day FROM {fq}.locations")
    return {r.location_id: int(r.capacity_patients_per_day) // 5 for r in df.itertuples()}


@mlflow.trace(span_type="TOOL")
def get_call_capacity(location_id=None):
    """Daily new-lead calls the front desk can make = capacity_patients_per_day // 5."""
    cap = capacity_map()
    if location_id:
        return {"location_id": location_id, "daily_calls": cap.get(location_id, 0)}
    return {"total_daily_calls": sum(cap.values()), "daily_calls_by_location": cap}


DRAFTS = {
    "Referral": "Hi! A friend of yours thought we could help. We would love to get you in this week - "
                "what time works for a quick first visit?",
    "Walk-In": "Thanks for stopping by our clinic! We saved a few new-patient slots this week - "
               "want me to hold one for you?",
}
DEFAULT_DRAFT = ("Hi, thanks for reaching out to us! We would be glad to help you feel your best. "
                 "Do you have a few minutes today to find a first-visit time that works for you?")


@mlflow.trace(span_type="TOOL")
def queue_actions(strategy, rationale, lead_ids=None):
    """Queue up to capacity leads per location (ranked by strategy); 20% seeded holdout per location.

    If lead_ids is given, only those leads are eligible. Writes `lead_action_queue`.
    """
    df = score_open()
    if lead_ids:
        df = df[df.lead_id.isin(lead_ids)]
    key = "uplift" if strategy == "uplift" else "p_current"
    cap = capacity_map()
    rows = []
    for loc, g in df.groupby("location_id"):
        chosen = g.sort_values(key, ascending=False).head(cap.get(loc, 0))
        held = gt.assign_holdout(chosen.lead_id.tolist())
        for prio, r in enumerate(chosen.itertuples(), start=1):
            hold = r.lead_id in held
            rows.append((r.lead_id, loc, prio, float(r.uplift), float(r.p_current), strategy,
                         "HOLDOUT - no call (measurement control)" if hold else "Call within 1 hour",
                         DRAFTS.get(r.source, DEFAULT_DRAFT) if prio <= 5 and not hold else None,
                         rationale, hold, _S["run_ts"]))
    schema = ("lead_id string, location_id string, priority int, predicted_uplift double, "
              "predicted_p_current double, strategy string, next_action string, outreach_draft string, "
              "rationale string, holdout boolean, run_ts timestamp")
    _S["spark"].createDataFrame(rows, schema).write.mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(f"{_S['fq']}.lead_action_queue")
    n_hold = sum(r[9] for r in rows)
    return {"queued": len(rows), "to_call": len(rows) - n_hold, "holdout": n_hold,
            "locations": len(set(r[1] for r in rows)),
            "expected_extra_conversions_if_all_called": round(sum(r[3] for r in rows), 2)}


@mlflow.trace(span_type="TOOL")
def write_manager_briefs(brief):
    """Write one manager brief per location: the agent's paragraph + that location's queue stats."""
    q = _sql("SELECT location_id, count(*) AS n, sum(cast(holdout AS int)) AS held, "
             "sum(predicted_uplift) AS up FROM {fq}.lead_action_queue GROUP BY 1 ORDER BY 1")
    rows = [(r.location_id,
             f"{brief} At {r.location_id}: call {int(r.n - r.held)} leads within 1 hour today "
             f"({int(r.held)} held out for measurement); expected +{r.up:.2f} conversions vs. the "
             f"current process if all queued leads are called.",
             _S["run_ts"]) for r in q.itertuples()]
    _S["spark"].createDataFrame(rows, "location_id string, brief string, run_ts timestamp") \
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{_S['fq']}.agent_briefs")
    return {"briefs_written": len(rows)}


TOOL_FUNCS = {f.__name__: f for f in [get_pipeline_summary, conversion_by_factor, fit_conversion_model,
                                      rank_open_leads, get_call_capacity, queue_actions,
                                      write_manager_briefs]}


def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


TOOL_SPECS = [
    _fn("get_pipeline_summary", "Open leads per location/status and historic conversion rate."),
    _fn("conversion_by_factor", "Historic conversion rate on closed leads by one factor.",
        {"factor": {"type": "string", "enum": list(_FACTOR_SQL)}}, ["factor"]),
    _fn("fit_conversion_model", "Fit a logistic regression on closed leads; returns coefficients."),
    _fn("rank_open_leads", "Rank open leads by 'uplift' (gain from a 1h call) or 'likelihood'.",
        {"strategy": {"type": "string", "enum": ["uplift", "likelihood"]},
         "location_id": {"type": "string", "description": "optional, e.g. LOC001"},
         "top_n": {"type": "integer"}}, ["strategy"]),
    _fn("get_call_capacity", "Daily front-desk call capacity per location.",
        {"location_id": {"type": "string", "description": "optional"}}),
    _fn("queue_actions", "Queue the top leads per location up to daily capacity using the chosen "
        "strategy (20% held out for measurement). Writes lead_action_queue.",
        {"strategy": {"type": "string", "enum": ["uplift", "likelihood"]},
         "rationale": {"type": "string", "description": "why this strategy, citing evidence"}},
        ["strategy", "rationale"]),
    _fn("write_manager_briefs", "Write a one-paragraph manager brief (applied to every location "
        "with its own queue stats). Name the top controllable driver and the decision.",
        {"brief": {"type": "string"}}, ["brief"]),
]


def call_tool(name, arguments):
    try:
        args = json.loads(arguments or "{}") if isinstance(arguments, str) else (arguments or {})
        args = {k: v for k, v in args.items() if v is not None}
        return json.dumps(TOOL_FUNCS[name](**args), default=str)
    except Exception as e:  # report errors back to the LLM so it can recover
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
