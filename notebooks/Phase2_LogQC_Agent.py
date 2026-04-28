# Databricks notebook source
# MAGIC %md
# MAGIC # Offset Well Intelligence Crew
# MAGIC ## Phase 2: Log QC Agent
# MAGIC
# MAGIC The Log QC Agent inspects each well's curves and flags depth intervals
# MAGIC where data quality is suspect — bad hole, washed out zones, sensor failures,
# MAGIC or missing curves. It reasons like a formation evaluation engineer doing
# MAGIC a first-pass QC before interpretation.
# MAGIC
# MAGIC **Eyes:** Compute QC statistics per well per depth interval
# MAGIC **Brain:** Claude reasons about what the statistics mean
# MAGIC **Hands:** Write flagged intervals to `silver_log_qc_flags` Delta table

# COMMAND ----------
# MAGIC %md ### Step 1: Install Anthropic SDK

# COMMAND ----------

# MAGIC %pip install anthropic
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ### Step 2: Configuration

# COMMAND ----------

import os
# Set your Anthropic API key
# In Databricks: go to your cluster → Environment variables → add ANTHROPIC_API_KEY
# OR set it directly here for dev purposes (do not commit to git)
os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get(scope="anthropic", key="api_key") if False else "<YOUR_API_KEY_HERE>"

CURRENT_WELL = "15_9-F-1C"
OFFSET_WELLS = ["15_9-F-11A", "15_9-F-1A", "15_9-F-11B", "15_9-F-1B"]
INTERVAL_SIZE = 50  # meters — QC window size

print(f"Current well: {CURRENT_WELL}")
print(f"Offset wells: {OFFSET_WELLS}")
print(f"QC interval size: {INTERVAL_SIZE}m")

# COMMAND ----------
# MAGIC %md ### Step 3: Eyes — Compute QC Statistics
# MAGIC
# MAGIC For each well, compute per-interval statistics that signal data quality issues:
# MAGIC - **CALI vs BS:** caliper >> bit size = washed out hole
# MAGIC - **RHOB spikes:** density drops sharply = bad hole effect
# MAGIC - **NPHI-RHOB crossover:** gas effect or bad data
# MAGIC - **Flat lines:** curve stuck = sensor failure
# MAGIC - **Null coverage:** missing curves per interval

# COMMAND ----------

from pyspark.sql.functions import (
    col, floor, mean, stddev, min, max, count,
    when, isnull, lit, round as spark_round
)
from pyspark.sql import functions as F

# Load bronze table
df = spark.table("offset_well_crew.bronze_well_logs")

# Create depth interval bins (50m windows)
df = df.withColumn("depth_interval", (floor(col("DEPTH") / INTERVAL_SIZE) * INTERVAL_SIZE).cast("int"))

# Compute QC statistics per well per interval
df_qc_stats = df.groupBy("WELL", "well_role", "depth_interval").agg(
    count("*").alias("sample_count"),
    spark_round(mean("GR"), 2).alias("gr_mean"),
    spark_round(stddev("GR"), 2).alias("gr_std"),
    spark_round(mean("RHOB"), 3).alias("rhob_mean"),
    spark_round(stddev("RHOB"), 3).alias("rhob_std"),
    spark_round(min("RHOB"), 3).alias("rhob_min"),
    spark_round(mean("NPHI"), 3).alias("nphi_mean"),
    spark_round(stddev("NPHI"), 3).alias("nphi_std"),
    spark_round(mean("CALI"), 3).alias("cali_mean"),
    spark_round(mean("RT"), 3).alias("rt_mean"),
    spark_round(mean("PEF"), 3).alias("pef_mean"),
    # Null counts per curve
    count(when(isnull("DT"), True)).alias("dt_null_count"),
    count(when(isnull("RHOB"), True)).alias("rhob_null_count"),
    count(when(isnull("NPHI"), True)).alias("nphi_null_count"),
)

# Add derived QC indicators
# Bit size in Volve wells is typically 8.5 inches — flag if CALI > 9.5 (washout)
df_qc_stats = df_qc_stats.withColumn(
    "washout_flag",
    when(col("cali_mean") > 9.5, True).otherwise(False)
)

# Flag RHOB low values — density < 2.0 g/cc in a clastic section = bad hole
df_qc_stats = df_qc_stats.withColumn(
    "low_density_flag",
    when(col("rhob_min") < 2.0, True).otherwise(False)
)

# Flag high RHOB stddev — spiky density = rough borehole
df_qc_stats = df_qc_stats.withColumn(
    "density_spike_flag",
    when(col("rhob_std") > 0.15, True).otherwise(False)
)

# Flag missing DT (sonic absent for current well and F-11B)
df_qc_stats = df_qc_stats.withColumn(
    "dt_missing_flag",
    when(col("dt_null_count") == col("sample_count"), True).otherwise(False)
)

# Flag low sample count — incomplete interval
df_qc_stats = df_qc_stats.withColumn(
    "incomplete_interval_flag",
    when(col("sample_count") < (INTERVAL_SIZE * 0.5), True).otherwise(False)
)

df_qc_stats = df_qc_stats.orderBy("WELL", "depth_interval")

print(f"QC intervals computed: {df_qc_stats.count()}")
display(df_qc_stats.filter(col("WELL") == CURRENT_WELL))

# COMMAND ----------
# MAGIC %md ### Step 4: Brain — Claude reasons about QC findings
# MAGIC
# MAGIC For each well, Claude receives the QC statistics and reasons about:
# MAGIC - Which intervals are problematic and why
# MAGIC - What the likely cause is (washout, tool failure, lithology effect)
# MAGIC - Confidence level in each flagged interval
# MAGIC - Whether the issue affects interpretability

# COMMAND ----------

import anthropic
import json

client = anthropic.Anthropic()

def build_qc_context(well_name, well_role, qc_rows):
    """Prepare QC stats for a single well as structured context for Claude."""
    intervals = []
    for row in qc_rows:
        intervals.append({
            "depth_interval_m": row["depth_interval"],
            "sample_count": row["sample_count"],
            "gr_mean_api": row["gr_mean"],
            "gr_std": row["gr_std"],
            "rhob_mean_gcc": row["rhob_mean"],
            "rhob_std": row["rhob_std"],
            "rhob_min_gcc": row["rhob_min"],
            "nphi_mean_vv": row["nphi_mean"],
            "cali_mean_in": row["cali_mean"],
            "rt_mean_ohmm": row["rt_mean"],
            "flags": {
                "washout": row["washout_flag"],
                "low_density": row["low_density_flag"],
                "density_spike": row["density_spike_flag"],
                "dt_missing": row["dt_missing_flag"],
                "incomplete_interval": row["incomplete_interval_flag"],
            }
        })
    return {
        "well_name": well_name,
        "well_role": well_role,
        "interval_size_m": INTERVAL_SIZE,
        "total_intervals": len(intervals),
        "intervals": intervals
    }

def run_log_qc_agent(well_name, well_role, qc_rows):
    """
    3-turn reasoning loop for Log QC Agent.
    Turn 1 (Eyes): Present raw QC statistics
    Turn 2 (Brain): Claude reasons about data quality
    Turn 3 (Hands): Claude produces structured flagged interval list
    """
    context = build_qc_context(well_name, well_role, qc_rows)

    # --- Turn 1: Eyes --- present raw statistics
    turn1_prompt = f"""You are a senior formation evaluation engineer with 20 years of experience 
interpreting wireline logs from offshore wells in the North Sea.

You are performing a first-pass log quality control (QC) review on well {well_name} 
({well_role} well) from the Volve field, Norwegian Continental Shelf.

Here are the per-interval QC statistics computed from the log data:

{json.dumps(context, indent=2)}

Bit size for Volve wells is typically 8.5 inches. CALI > 9.5 inches indicates washout.
Normal RHOB for North Sea clastics: 2.1-2.65 g/cc. 
GR > 100 API typically indicates shale.
NPHI-RHOB relationship: in good hole, NPHI and RHOB track lithology consistently.

Please acknowledge the data and identify which intervals have one or more QC flags raised."""

    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": turn1_prompt}]
    )
    turn1_response = response1.content[0].text

    # --- Turn 2: Brain --- deep reasoning about causes
    turn2_prompt = f"""Good. Now reason deeply about each flagged interval.

For each flagged depth interval, determine:
1. What is the most likely CAUSE of the QC issue? 
   (e.g., washed out borehole, rugose hole, tool stick-slip, lithology effect, 
    gas effect, tool failure, missing run)
2. How SEVERE is the issue? (CRITICAL / MODERATE / MINOR)
3. Does this interval affect LOG INTERPRETABILITY? (yes/no)
4. What would you tell the client about this interval?

Use your formation evaluation expertise — consider that:
- High GR + high CALI = shale with washout (common, expected)
- Low GR + high CALI = sandstone washout (more concerning for porosity reads)
- Low RHOB + high NPHI = gas effect OR bad hole (must distinguish)
- Density spikes in good CALI = real lithology change, not bad hole
- Missing DT for {well_name} is a known data gap for this well, not a tool failure"""

    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[
            {"role": "user", "content": turn1_prompt},
            {"role": "assistant", "content": turn1_response},
            {"role": "user", "content": turn2_prompt}
        ]
    )
    turn2_response = response2.content[0].text

    # --- Turn 3: Hands --- structured output
    turn3_prompt = f"""Based on your QC analysis, produce a structured JSON output of flagged intervals.

Return ONLY valid JSON — no preamble, no markdown, no explanation outside the JSON.

Format:
{{
  "well_name": "{well_name}",
  "well_role": "{well_role}",
  "qc_summary": "2-3 sentence overall assessment of data quality for this well",
  "flagged_intervals": [
    {{
      "depth_from_m": <number>,
      "depth_to_m": <number>,
      "severity": "CRITICAL|MODERATE|MINOR",
      "primary_issue": "<short label e.g. Washout, Missing Sonic, Density Spike>",
      "cause": "<1 sentence explanation>",
      "affects_interpretation": true|false,
      "recommendation": "<1 sentence — what should be done>"
    }}
  ],
  "total_flagged_intervals": <number>,
  "intervals_critical": <number>,
  "intervals_moderate": <number>,
  "intervals_minor": <number>
}}"""

    response3 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": turn1_prompt},
            {"role": "assistant", "content": turn1_response},
            {"role": "user", "content": turn2_prompt},
            {"role": "assistant", "content": turn2_response},
            {"role": "user", "content": turn3_prompt}
        ]
    )
    turn3_response = response3.content[0].text

    return {
        "well_name": well_name,
        "well_role": well_role,
        "turn1": turn1_response,
        "turn2": turn2_response,
        "turn3_json": turn3_response
    }

print("Log QC Agent defined.")

# COMMAND ----------
# MAGIC %md ### Step 5: Run the agent across all wells

# COMMAND ----------

import pandas as pd

# Collect QC stats to driver as list of dicts
qc_stats_pd = df_qc_stats.toPandas()

all_results = []
all_wells = [CURRENT_WELL] + OFFSET_WELLS

for well in all_wells:
    well_data = qc_stats_pd[qc_stats_pd["WELL"] == well]
    if well_data.empty:
        print(f"Skipping {well} — no data found")
        continue

    well_role = well_data["well_role"].iloc[0]
    qc_rows = well_data.to_dict(orient="records")

    print(f"\nRunning QC agent for: {well} ({well_role}) — {len(qc_rows)} intervals...")
    result = run_log_qc_agent(well, well_role, qc_rows)
    all_results.append(result)
    print(f"  Done. Summary preview:")
    try:
        parsed = json.loads(result["turn3_json"])
        print(f"  {parsed['qc_summary']}")
        print(f"  Flagged: {parsed['total_flagged_intervals']} intervals — "
              f"CRITICAL: {parsed['intervals_critical']}, "
              f"MODERATE: {parsed['intervals_moderate']}, "
              f"MINOR: {parsed['intervals_minor']}")
    except Exception as e:
        print(f"  Parse error: {e}")
        print(f"  Raw: {result['turn3_json'][:300]}")

print(f"\nAgent complete. Processed {len(all_results)} wells.")

# COMMAND ----------
# MAGIC %md ### Step 6: Hands — Write flagged intervals to Silver Delta table

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, BooleanType, FloatType
)

flagged_rows = []

for result in all_results:
    try:
        parsed = json.loads(result["turn3_json"])
        for interval in parsed.get("flagged_intervals", []):
            flagged_rows.append({
                "well_name":             parsed["well_name"],
                "well_role":             parsed["well_role"],
                "depth_from_m":          float(interval.get("depth_from_m", 0)),
                "depth_to_m":            float(interval.get("depth_to_m", 0)),
                "severity":              interval.get("severity", "UNKNOWN"),
                "primary_issue":         interval.get("primary_issue", ""),
                "cause":                 interval.get("cause", ""),
                "affects_interpretation": bool(interval.get("affects_interpretation", False)),
                "recommendation":        interval.get("recommendation", ""),
                "qc_summary":            parsed.get("qc_summary", ""),
            })
    except Exception as e:
        print(f"Error parsing result for {result['well_name']}: {e}")

# Create DataFrame and write to Silver Delta table
if flagged_rows:
    df_flags = spark.createDataFrame(pd.DataFrame(flagged_rows))

    (df_flags
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable("offset_well_crew.silver_log_qc_flags")
    )
    print(f"Written {len(flagged_rows)} flagged intervals to silver_log_qc_flags")
else:
    print("No flagged intervals to write.")

# COMMAND ----------
# MAGIC %md ### Step 7: Validate and review results

# COMMAND ----------

df_silver = spark.table("offset_well_crew.silver_log_qc_flags")

print("=== Flagged intervals by well and severity ===")
df_silver.groupBy("well_name", "well_role", "severity").count() \
    .orderBy("well_name", "severity").show()

print("\n=== CRITICAL intervals ===")
df_silver.filter(col("severity") == "CRITICAL") \
    .select("well_name", "depth_from_m", "depth_to_m", "primary_issue", "recommendation") \
    .orderBy("well_name", "depth_from_m") \
    .show(truncate=60)

print("\n=== Current well QC summary ===")
df_silver.filter(col("well_name") == CURRENT_WELL) \
    .select("depth_from_m", "depth_to_m", "severity", "primary_issue", "cause") \
    .orderBy("depth_from_m") \
    .show(truncate=80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 2 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `offset_well_crew.bronze_well_logs` | Raw well log data — all wells |
# MAGIC | `offset_well_crew.silver_log_qc_flags` | AI-flagged depth intervals with severity + reasoning |
# MAGIC
# MAGIC **Next:** Phase 3 — Formation Tops Agent
