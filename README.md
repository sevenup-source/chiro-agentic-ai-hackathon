# Lead Rescue Agent: uplift-ranked lead triage for a chiropractic chain

**Problem.** A boutique chiropractic chain ($100M ARR, aiming for $250M) loses new patients because
leads go cold. Front desks answer slowly and call the wrong leads first, and they can only make a
limited number of calls each day.

**Solution.** An LLM agent on Databricks that **observes** the pipeline, **reasons** about which
conversion drivers the front desk can control, **decides** on a ranking policy within each clinic's
call capacity, and **acts**: it writes a prioritized call queue with outreach drafts and a brief for
each manager. The agent ranks leads by **uplift** (how much a fast call changes conversion), not by
likelihood. A lead who would convert anyway is a wasted call.

**Closed loop.** The provided synthetic data is independent noise, so we inject a documented
ground-truth causal model (`src/ground_truth.py`, also published as the `ground_truth_model` table).
This gives the agent real signal to find and gives us something to grade it against. Every run is
evaluated three ways at equal capacity: the agent's policy, a naive "highest likelihood" policy and
a random policy. We also compare treated leads with a seeded 20% holdout.

## Architecture

```
setup_data job:        generate_synthetic_data (notebook) -> inject_signal.py  => leads, ground_truth_model
lead_rescue_agent job: run_agent.py (LLM tool loop, MLflow traced)  -> measure.py
                         tools.py: get_pipeline_summary, conversion_by_factor, fit_conversion_model,
                                   rank_open_leads, get_call_capacity, queue_actions, write_manager_briefs
                         => lead_action_queue, agent_briefs                   => agent_eval
```

| Path | What |
|---|---|
| `databricks.yml` | Bundle `chiro-lead-agent`; variables `catalog`, `schema`, `llm_endpoint`; targets `dev` / `prod` |
| `resources/*.job.yml` | The two serverless jobs |
| `sample_data/generate_synthetic_data.py` | Xorbix generator, with catalog and schema as widgets |
| `src/ground_truth.py` | Planted causal model and holdout assignment (pure Python) |
| `src/inject_signal.py` | Step 1: rewrites `leads` according to the ground truth |
| `src/agent/` | Step 2: agent loop (`run_agent.py`) and tools (`tools.py`) |
| `src/measure.py` | Step 3: three-arm evaluation, $ impact, driver check |
| `tests/` | pytest, runs locally without Spark |

### Ground truth (planted)
```
logit = -1.2 + SOURCE_EFFECT[source] - 0.05*first_response_hours
        + 0.35*min(touchpoints,4) - 0.10*max(touchpoints-6,0)
SOURCE_EFFECT: Referral +1.0, Walk-In +0.6, Phone +0.3, Website 0, Insurance Dir -0.1,
               Community Event -0.2, Social Media Ad -0.6
```
Leads created more than 30 days ago are closed (converted ~ Bernoulli(sigmoid(logit))). Newer leads
are open, and their `first_response_hours` is the projected response time under the current process.
The agent never sees these coefficients. It fits its own model on closed leads.

## Setup / deploy / run

Prerequisites: Databricks CLI v0.230+ authenticated to your workspace (`databricks auth login`).
The host comes from your CLI profile, so no hosts or tokens are committed. Serverless compute only,
so it works on Free Edition.

```bash
databricks bundle validate
```
```bash
databricks bundle deploy
```
```bash
databricks bundle run setup_data
```
```bash
databricks bundle run lead_rescue_agent
```

To deploy to another workspace, change config only. For example:
```bash
databricks bundle deploy -t prod --var="catalog=main" --var="schema=chiro" --var="llm_endpoint=databricks-meta-llama-3-3-70b-instruct"
```
List the LLM endpoints available in a workspace with `databricks serving-endpoints list`. Any
Foundation Model endpoint that supports tool calling works.

## Validate

```bash
pip install pytest && pytest -q
```

- `agent_eval` table: extra conversions for agent vs naive vs random, treated vs holdout, $ impact
  scaled to $100M ARR (the scaling assumption is printed and stored in the `note` column), and the
  driver check (did the agent name response time?).
- MLflow trace: Experiments, then `/Users/<you>/chiro_lead_rescue_agent`, then Traces. Each LLM call
  and tool call is a span.
- Outputs: `lead_action_queue` (priority, predicted uplift, next action, outreach draft, rationale,
  holdout), `agent_briefs`.

## Notes
- No PHI. Leads are surrogate IDs and the outreach drafts are generic templates per source.
- Determinism: seed 42 for signal injection, holdout and evaluation.
