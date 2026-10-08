"""Step 1: overwrite `leads` so conversions follow the documented ground-truth model."""
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
args = p.parse_args()
sys.path.insert(0, args.src_dir)
import ground_truth as gt  # noqa: E402

spark = SparkSession.builder.getOrCreate()
fq = f"{args.catalog}.{args.schema}"
cutoff = date.today() - timedelta(days=gt.OPEN_WINDOW_DAYS)

leads = spark.table(f"{fq}.leads").orderBy("lead_id").toPandas()
rng = np.random.default_rng(gt.SEED)
u = rng.random(len(leads))

closed = leads["created_date"].map(lambda d: d < cutoff).to_numpy()
prob = np.array([gt.conversion_prob(s, h, t) for s, h, t in
                 zip(leads["source"], leads["first_response_hours"], leads["num_touchpoints"])])
converted = closed & (u < prob)

status = np.where(converted, "Converted", "Lost")
open_status = leads["status"].where(leads["status"].isin(["New", "Contacted", "Qualified"]), "New")
leads["status"] = np.where(closed, status, open_status)
leads["converted_flag"] = converted
# keep an existing patient link for converted rows when present, else synthesize a surrogate id
leads["converted_patient_id"] = [
    (pid if pid else f"PT{9000000 + i:07d}") if c else None
    for i, (pid, c) in enumerate(zip(leads["converted_patient_id"], converted))]

schema = spark.table(f"{fq}.leads").schema
spark.createDataFrame(leads[[f.name for f in schema]], schema=schema) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fq}.leads")

spark.createDataFrame(gt.describe(), "term string, level string, coefficient double") \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fq}.ground_truth_model")

print(f"closed leads: {closed.sum():,}  open leads: {(~closed).sum():,}  "
      f"historic conversion rate: {converted[closed].mean():.3f}")
