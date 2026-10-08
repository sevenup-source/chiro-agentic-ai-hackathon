# Lead Rescue Agent — build spec (Xorbix x UIowa Databricks Hackathon)

HARD DEADLINE: judging 6:30pm CT today. Feature freeze 5:40pm. Prefer working + simple over clever.
Judges (Xorbix, a Databricks consultancy) score 1-5 equally on: Business Impact, Technical Innovation
(real agentic reasoning + tools + decisions, not a chatbot), Prototype Quality (it works), Pitch & Dev
(public repo, valid DAB, deployable to another workspace via config only, README, no secrets).
Slide also asks for: modular code, config separated from code, tests, docs.

## Problem
Boutique chiropractic chain, $100M ARR -> $250M goal. New-patient acquisition leaks: leads go cold
because staff respond slowly and call the wrong leads first. Front desks have limited call capacity.

## The core idea (what makes this non-generic)
1. The provided synthetic data (generate_synthetic_data.py) is pure independent noise. We INJECT a
   documented ground-truth causal model so the agent has real signal to find AND we can grade it.
2. The agent ranks leads by UPLIFT (how much a fast call changes conversion), not by likelihood.
   Calling leads who'd convert anyway wastes scarce front-desk capacity.
3. Closed loop: holdout group + three-arm evaluation (agent policy vs naive "highest likelihood" vs
   random) against the planted ground truth. The agent is evaluated, not just run.

## Data (catalog/schema are bundle variables; defaults workspace / chiro_hackathon)
Existing tables from the Xorbix generator: locations, providers, patients, leads, appointments,
visits, referrals, marketing_campaigns. Leads cols: lead_id, source, assigned_location_id,
created_date, status, first_response_hours, num_touchpoints, converted_flag, converted_patient_id.

### Step 1 — src/inject_signal.py (deterministic, seed 42; overwrite `leads`)
Ground truth (also put these constants in src/ground_truth.py, pure Python, unit-tested):
  logit = -1.2 + SOURCE_EFFECT[source] - 0.05*first_response_hours
          + 0.35*min(num_touchpoints,4) - 0.10*max(num_touchpoints-6,0)
  SOURCE_EFFECT = {Referral:+1.0, Walk-In:+0.6, Phone Inquiry:+0.3, Website Form:0.0,
                   Insurance Directory:-0.1, Community Event:-0.2, Social Media Ad:-0.6}
- Closed leads (created_date < today-30d): converted_flag ~ Bernoulli(sigmoid(logit));
  status = Converted / Lost accordingly; converted_patient_id null if not converted.
- Open leads (created_date >= today-30d): status in New/Contacted/Qualified (keep existing if open,
  else map to New), converted_flag=false, converted_patient_id=null. Their first_response_hours =
  projected response time under the CURRENT process.
Write the ground-truth description to a table `ground_truth_model` (for transparency in the demo).

### Step 2 — src/agent/ (the agent; run as a job task)
LLM: Databricks Foundation Model endpoint, name = bundle variable `llm_endpoint`. Find valid names
with `databricks serving-endpoints list`. Client: WorkspaceClient().serving_endpoints.get_open_ai_client().
Use OpenAI-style tool calling in a loop (max ~10 steps). mlflow.openai.autolog() / @mlflow.trace so
every step shows in an MLflow trace (we show this in the demo).
Tools (plain Python functions over spark.sql, in src/agent/tools.py):
  - get_pipeline_summary(): open leads per location/status, historical conversion rate
  - conversion_by_factor(factor in [source, response_bucket, touchpoints, location]): historic rates
  - fit_conversion_model(): sklearn LogisticRegression on closed leads (source one-hot,
    first_response_hours, min(tp,4), max(tp-6,0)); returns coefficients; caches model
  - rank_open_leads(strategy in [uplift, likelihood], location_id): uplift = p(response=1h) - p(current)
  - get_call_capacity(location_id): daily calls = capacity_patients_per_day // 5
  - queue_actions(lead_ids, rationale): writes chosen leads to `lead_action_queue`
    (lead_id, location_id, priority, predicted_uplift, next_action, outreach_draft, rationale, run_ts)
System prompt: Observe -> Reason (find drivers, justify) -> Decide (pick strategy, respect capacity)
-> Act (queue leads + a short, warm outreach draft for the top 5 per location; NO PHI) -> write a
one-paragraph manager brief per location to `agent_briefs`.
20% of each location's chosen candidates are randomly held out (seeded) -> `holdout` flag.

### Step 3 — src/measure.py
Simulate outcomes from ground truth: treated leads get response 1h, others keep projected hours.
Compare three policies at equal capacity: agent queue, naive top-likelihood, random. Also treated vs
holdout. Report: extra conversions, conversion lift, $ = extra patients x avg patient LTV (from
visits revenue per patient). Annualize and scale to a $100M ARR business (state the scaling
assumption explicitly). Write `agent_eval` table and print a clean summary. Also check: did the agent
name response time as the top controllable driver? (string check on its brief) -> pass/fail.

## Bundle (DAB) — repo root
databricks.yml: bundle name chiro-lead-agent; variables catalog, schema, llm_endpoint (no defaults
that are workspace-specific except catalog=workspace, schema=chiro_hackathon); targets dev (default,
mode: development) and prod. NO hosts or tokens committed; host comes from the CLI profile.
resources/: job `setup_data` (generate_synthetic_data notebook -> inject_signal) and job
`lead_rescue_agent` (run_agent -> measure). Serverless compute (Free Edition): use environments /
serverless tasks, no clusters.
Layout: README.md, databricks.yml, resources/, src/, sample_data/generate_synthetic_data.py,
tests/ (pytest for ground_truth.py + holdout assignment; must run locally without Spark).

## Definition of done (verify each by running it)
- `databricks bundle validate` passes
- `databricks bundle deploy` then `databricks bundle run setup_data` and `run lead_rescue_agent` succeed
- agent_eval shows agent policy > naive > random; MLflow trace visible
- `pytest` passes; README has setup / deploy / run / validate; .gitignore covers .databricks/, .env
- Iterate: deploy, run, read errors, fix. Do not guess at Databricks APIs — check docs or CLI help.
