"""Step 2: Lead Rescue Agent - an LLM tool-calling loop (Observe -> Reason -> Decide -> Act)."""
import argparse
import os
import sys

import mlflow
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

SYSTEM_PROMPT = """You are the Lead Rescue Agent for a chain of chiropractic clinics.
New-patient leads go cold because front desks respond slowly and call the wrong leads first.
Front-desk call capacity is scarce. Work in this order, using tools for every fact:
1. OBSERVE: get_pipeline_summary, then conversion_by_factor for source, response_bucket and touchpoints.
2. REASON: fit_conversion_model. Identify which drivers matter and which one the front desk can CONTROL
   (lead source and touchpoint history are fixed; response time is controllable). Justify with numbers.
3. DECIDE: compare rank_open_leads with strategy 'uplift' vs 'likelihood' (compare sum_uplift_top_n).
   Calling leads who would convert anyway wastes capacity; prefer the strategy that maximizes the
   causal gain. Check get_call_capacity.
4. ACT: call queue_actions with your chosen strategy and an evidence-based rationale, then
   write_manager_briefs with ONE short paragraph for clinic managers that names the top controllable
   driver, the decision, and what to do today. Never include personal or health information.
Call one tool at a time. When done, reply with a 3-sentence summary."""

MAX_STEPS = 14


@mlflow.trace(span_type="AGENT", name="lead_rescue_agent")
def run(client, endpoint, tools):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Run today's lead rescue."}]
    called = []
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=endpoint, messages=messages,
                                              tools=tools.TOOL_SPECS, temperature=0.0, max_tokens=1500)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            if "write_manager_briefs" in called:
                return msg.content
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": "Continue with the next step using tools: you "
                             "must call queue_actions and then write_manager_briefs before finishing."})
            continue
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            print(f"-> {tc.function.name}({tc.function.arguments})", flush=True)
            called.append(tc.function.name)
            out = tools.call_tool(tc.function.name, tc.function.arguments)
            print(f"   {out[:300]}", flush=True)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:6000]})
    # Safety net: an agent that runs out of steps still leaves an auditable, sensible queue.
    if "queue_actions" not in called:
        tools.queue_actions("uplift", "Fallback: step budget exhausted; default uplift policy.")
    if "write_manager_briefs" not in called:
        tools.write_manager_briefs("Fallback brief: respond to queued leads within 1 hour; "
                                   "response time is the top controllable driver of conversion.")
    return "step budget exhausted (fallback actions applied)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default=os.getcwd())
    p.add_argument("--catalog", default="workspace")
    p.add_argument("--schema", default="chiro_hackathon")
    p.add_argument("--llm_endpoint", default="databricks-meta-llama-3-3-70b-instruct")
    a = p.parse_args()
    sys.path.insert(0, a.src_dir)
    sys.path.insert(0, os.path.join(a.src_dir, "agent"))
    import tools  # noqa: E402

    w = WorkspaceClient()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f"/Users/{w.current_user.me().user_name}/chiro_lead_rescue_agent")
    mlflow.openai.autolog()

    tools.init(SparkSession.builder.getOrCreate(), a.catalog, a.schema)
    client = w.serving_endpoints.get_open_ai_client()
    with mlflow.start_run(run_name="lead_rescue_agent"):
        mlflow.log_param("llm_endpoint", a.llm_endpoint)
        summary = run(client, a.llm_endpoint, tools)
    print("AGENT SUMMARY:\n", summary)


if __name__ == "__main__":
    main()
