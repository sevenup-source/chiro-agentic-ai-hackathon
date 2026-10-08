"""Step 3: grade the agent against the planted ground truth (closed-loop evaluation).

Three policies at EQUAL capacity per location (same number of leads as the agent queued):
  agent  - the leads in lead_action_queue
  naive  - top-K by true current conversion likelihood ("call the hottest leads first")
  random - seeded random K
Value = expected extra conversions = sum of true uplift (p at 1h response - p at projected response).
Also: simulated treated vs holdout outcomes inside the agent queue.
"""
import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
from pyspark.sql import SparkSession

p = argparse.ArgumentParser()
p.add_argument("--src_dir", default=os.getcwd())
p.add_argument("--catalog", default="workspace")
p.add_argument("--schema", default="chiro_hackathon")
p.add_argument("--target_arr", type=float, default=100e6)
args = p.parse_args()
sys.path.insert(0, args.src_dir)
import ground_truth as gt  # noqa: E402

spark = SparkSession.builder.getOrCreate()
fq = f"{args.catalog}.{args.schema}"
cutoff = (date.today() - timedelta(days=gt.OPEN_WINDOW_DAYS)).isoformat()

open_df = spark.sql(f"SELECT lead_id, assigned_location_id AS location_id, source, first_response_hours, "
                    f"num_touchpoints FROM {fq}.leads WHERE created_date >= '{cutoff}'").toPandas()
open_df["p_current"] = [gt.conversion_prob(s, h, t) for s, h, t in
                        zip(open_df.source, open_df.first_response_hours, open_df.num_touchpoints)]
open_df["true_uplift"] = [gt.true_uplift(s, h, t) for s, h, t in
                          zip(open_df.source, open_df.first_response_hours, open_df.num_touchpoints)]
queue = spark.table(f"{fq}.lead_action_queue").toPandas()
q = queue.merge(open_df, on=["lead_id", "location_id"])

rng = np.random.default_rng(gt.SEED)
agent_v = naive_v = random_v = 0.0
for loc, g in open_df.groupby("location_id"):
    k = int((queue.location_id == loc).sum())
    agent_v += q[q.location_id == loc].true_uplift.sum()
    naive_v += g.nlargest(k, "p_current").true_uplift.sum()
    random_v += g.iloc[rng.permutation(len(g))[:k]].true_uplift.sum()

baseline = open_df.p_current.sum()  # expected conversions of the open pipeline under current process

# Treated vs holdout (simulated outcomes from ground truth)
treated, held = q[~q.holdout], q[q.holdout]
p_treated = [gt.conversion_prob(s, gt.TREATED_RESPONSE_HOURS, t) for s, t in
             zip(treated.source, treated.num_touchpoints)]
treated_rate = float(np.mean(rng.random(len(p_treated)) < p_treated)) if len(p_treated) else 0.0
holdout_rate = float(np.mean(rng.random(len(held)) < held.p_current.to_numpy())) if len(held) else 0.0

# Dollars: LTV from visits revenue per patient; scale synthetic business to target ARR
v = spark.sql(f"SELECT sum(revenue) / count(DISTINCT patient_id) AS ltv, "
              f"sum(CASE WHEN visit_date >= date_sub(current_date(), 365) THEN revenue END) AS annual_rev "
              f"FROM {fq}.visits").toPandas()
ltv, annual_rev = float(v.ltv[0]), float(v.annual_rev[0])
scale = args.target_arr / annual_rev
RUNS_PER_YEAR = 12  # one rescue run per 30-day open-pipeline window (conservative)
extra = agent_v - naive_v
annual_value = agent_v * RUNS_PER_YEAR * scale * ltv
annual_vs_naive = extra * RUNS_PER_YEAR * scale * ltv

briefs = " ".join(spark.table(f"{fq}.agent_briefs").toPandas().brief).lower()
driver_pass = "response" in briefs

metrics = [
    ("capacity_leads_queued", float(len(queue)), "equal capacity for all three policies"),
    ("open_pipeline_expected_conversions_baseline", float(baseline), "current process"),
    ("extra_conversions_agent", float(agent_v), "sum true uplift, agent queue"),
    ("extra_conversions_naive_likelihood", float(naive_v), "sum true uplift, top-likelihood"),
    ("extra_conversions_random", float(random_v), "sum true uplift, seeded random"),
    ("agent_vs_naive_ratio", float(agent_v / naive_v) if naive_v else float("nan"), ""),
    ("conversion_lift_pct_agent", float(100 * agent_v / baseline), "vs baseline pipeline conversions"),
    ("treated_conversion_rate_sim", treated_rate, f"n={len(treated)}"),
    ("holdout_conversion_rate_sim", holdout_rate, f"n={len(held)}"),
    ("avg_patient_ltv_usd", ltv, "visits revenue / distinct patients"),
    ("synthetic_annual_revenue_usd", annual_rev, "visits revenue, last 365 days"),
    ("scale_factor_to_target_arr", scale, f"target ARR ${args.target_arr:,.0f} / synthetic annual revenue"),
    ("annual_value_agent_usd_scaled", annual_value,
     f"extra conversions x {RUNS_PER_YEAR} runs/yr x scale x LTV"),
    ("annual_value_agent_vs_naive_usd_scaled", annual_vs_naive, "incremental over naive policy"),
    ("driver_check_response_time_named", float(driver_pass), "brief names response time"),
]
spark.createDataFrame(metrics, "metric string, value double, note string").write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(f"{fq}.agent_eval")

print("=" * 72)
print("LEAD RESCUE AGENT - CLOSED-LOOP EVALUATION vs PLANTED GROUND TRUTH")
print("=" * 72)
for m, val, note in metrics:
    print(f"{m:<45} {val:>16,.3f}  {note}")
print("-" * 72)
print(f"Policy ranking: agent {agent_v:.2f} | naive {naive_v:.2f} | random {random_v:.2f} extra conversions")
print(f"Agent beats naive: {agent_v > naive_v}; agent beats random: {agent_v > random_v}")
print(f"Driver check (response time named): {'PASS' if driver_pass else 'FAIL'}")
print(f"Scaling assumption: synthetic chain earns ${annual_rev:,.0f}/yr; multiplied by {scale:.1f}x to a "
      f"${args.target_arr:,.0f} ARR business, {RUNS_PER_YEAR} runs/yr.")
