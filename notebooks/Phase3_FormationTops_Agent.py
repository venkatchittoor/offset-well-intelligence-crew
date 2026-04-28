# Databricks notebook source
# MAGIC %md
# MAGIC # Offset Well Intelligence Crew
# MAGIC ## Phase 3: Formation Tops Agent
# MAGIC
# MAGIC The Formation Tops Agent correlates formation tops across offset wells and
# MAGIC flags depth intervals where the current well deviates from the offset analog pattern.
# MAGIC
# MAGIC **Key formations — Volve field:**
# MAGIC - **Draupne Formation:** cap rock shale — high GR (>90 API), low RT, high RHOB
# MAGIC - **Hugin Formation:** reservoir sandstone — low GR (<50 API), high RT, lower RHOB
# MAGIC - **Hugin depth range:** ~2750–3700m across Volve wells
# MAGIC
# MAGIC **Eyes:** Compute formation signature statistics per well per depth interval
# MAGIC **Brain:** Claude correlates tops and identifies deviations in current well
# MAGIC **Hands:** Write correlated tops + deviations to `silver_formation_tops` Delta table

# COMMAND ----------
# MAGIC %md ### Step 1: Install Anthropic SDK

# COMMAND ----------

# MAGIC %pip install anthropic
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ### Step 2: Configuration

# COMMAND ----------

import os
os.environ["ANTHROPIC_API_KEY"] = "<YOUR_API_KEY_HERE>"

CURRENT_WELL = "15_9-F-1C"
OFFSET_WELLS = ["15_9-F-11A", "15_9-F-1A", "15_9-F-11B", "15_9-F-1B"]
INTERVAL_SIZE = 50  # meters — correlation window

# Volve-specific formation thresholds (validated against literature)
GR_CLEAN_SAND   = 50   # API — Hugin sandstone threshold
GR_SHALE        = 90   # API — Draupne shale threshold
RHOB_POROUS     = 2.35 # g/cc — porous sand upper bound
RT_HYDROCARBON  = 5.0  # ohm.m — resistivity threshold for HC-bearing sand
DEPTH_SHIFT_MODERATE  = 50   # meters — moderate deviation flag
DEPTH_SHIFT_CRITICAL  = 100  # meters — critical deviation flag

print(f"Current well: {CURRENT_WELL}")
print(f"Offset wells: {OFFSET_WELLS}")
print(f"GR clean sand threshold: {GR_CLEAN_SAND} API")
print(f"GR shale threshold: {GR_SHALE} API")

# COMMAND ----------
# MAGIC %md ### Step 3: Eyes — Compute Formation Signature Statistics
# MAGIC
# MAGIC For each well, compute per-interval formation signatures:
# MAGIC - GR character — sand vs shale classification
# MAGIC - RHOB — porosity indicator
# MAGIC - RT — fluid indicator (hydrocarbon vs brine)
# MAGIC - NPHI — neutron porosity
# MAGIC - PEF — lithology indicator
# MAGIC - Flag intervals that match Draupne or Hugin signatures

# COMMAND ----------

from pyspark.sql.functions import (
    col, floor, mean, stddev, min, max, count,
    when, lit, round as spark_round, lag, abs as spark_abs
)
from pyspark.sql import functions as F

# Load bronze table
df = spark.table("offset_well_crew.bronze_well_logs")

# Create depth interval bins
df = df.withColumn(
    "depth_interval",
    (floor(col("DEPTH") / INTERVAL_SIZE) * INTERVAL_SIZE).cast("int")
)

# Compute formation signature statistics per well per interval
df_sigs = df.groupBy("WELL", "well_role", "depth_interval").agg(
    count("*").alias("sample_count"),
    spark_round(mean("GR"), 2).alias("gr_mean"),
    spark_round(stddev("GR"), 2).alias("gr_std"),
    spark_round(mean("RHOB"), 3).alias("rhob_mean"),
    spark_round(mean("NPHI"), 3).alias("nphi_mean"),
    spark_round(mean("RT"), 3).alias("rt_mean"),
    spark_round(mean("PEF"), 3).alias("pef_mean"),
    spark_round(min("GR"), 2).alias("gr_min"),
    spark_round(max("GR"), 2).alias("gr_max"),
)

# Classify each interval by formation signature
df_sigs = df_sigs.withColumn(
    "formation_class",
    when(col("gr_mean") > GR_SHALE, "DRAUPNE_SHALE")
    .when(col("gr_mean") < GR_CLEAN_SAND, "HUGIN_SAND")
    .when((col("gr_mean") >= GR_CLEAN_SAND) & (col("gr_mean") <= GR_SHALE), "TRANSITION")
    .otherwise("UNKNOWN")
)

# Flag potential reservoir intervals (low GR + elevated RT)
df_sigs = df_sigs.withColumn(
    "potential_reservoir",
    when(
        (col("gr_mean") < GR_CLEAN_SAND) & (col("rt_mean") > RT_HYDROCARBON),
        True
    ).otherwise(False)
)

# Flag Draupne cap rock intervals
df_sigs = df_sigs.withColumn(
    "potential_cap_rock",
    when(col("gr_mean") > GR_SHALE, True).otherwise(False)
)

df_sigs = df_sigs.orderBy("WELL", "depth_interval")

print(f"Formation signature intervals computed: {df_sigs.count()}")

print("\n=== Formation classification by well ===")
df_sigs.groupBy("WELL", "formation_class").count().orderBy("WELL", "formation_class").show()

print("\n=== Potential reservoir intervals ===")
df_sigs.filter(col("potential_reservoir") == True) \
    .select("WELL", "depth_interval", "gr_mean", "rt_mean", "rhob_mean", "formation_class") \
    .orderBy("WELL", "depth_interval") \
    .show()

# COMMAND ----------
# MAGIC %md ### Step 4: Compute offset well formation top summary
# MAGIC
# MAGIC Identify the depth at which key formation transitions occur in each offset well.
# MAGIC This gives Claude the analog pattern to compare against the current well.

# COMMAND ----------

import pandas as pd

# Pull formation signatures to pandas for agent processing
sigs_pd = df_sigs.toPandas()

def get_well_formation_summary(well_name, sigs_pd):
    """Extract key formation intervals for a single well."""
    well_data = sigs_pd[sigs_pd["WELL"] == well_name].sort_values("depth_interval")

    summary = {
        "well_name": well_name,
        "well_role": well_data["well_role"].iloc[0],
        "depth_range_m": {
            "from": int(well_data["depth_interval"].min()),
            "to": int(well_data["depth_interval"].max())
        },
        "total_intervals": len(well_data),
        "formation_distribution": well_data["formation_class"].value_counts().to_dict(),
        "reservoir_intervals": well_data[well_data["potential_reservoir"] == True][
            ["depth_interval", "gr_mean", "rt_mean", "rhob_mean", "nphi_mean"]
        ].to_dict(orient="records"),
        "cap_rock_intervals": well_data[well_data["potential_cap_rock"] == True][
            ["depth_interval", "gr_mean", "rhob_mean"]
        ].to_dict(orient="records"),
        "all_intervals": well_data[
            ["depth_interval", "gr_mean", "rhob_mean", "rt_mean", "nphi_mean",
             "formation_class", "potential_reservoir", "potential_cap_rock"]
        ].to_dict(orient="records")
    }
    return summary

# Build summaries for all wells
all_summaries = {}
for well in OFFSET_WELLS + [CURRENT_WELL]:
    well_data = sigs_pd[sigs_pd["WELL"] == well]
    if not well_data.empty:
        all_summaries[well] = get_well_formation_summary(well, sigs_pd)
        res_count = len(all_summaries[well]["reservoir_intervals"])
        cap_count = len(all_summaries[well]["cap_rock_intervals"])
        print(f"{well}: {res_count} reservoir intervals, {cap_count} cap rock intervals")

# COMMAND ----------
# MAGIC %md ### Step 5: Brain — Claude correlates formation tops
# MAGIC
# MAGIC 3-turn reasoning loop:
# MAGIC - Turn 1: Present offset well formation signatures — establish the analog pattern
# MAGIC - Turn 2: Compare current well against the pattern — identify deviations
# MAGIC - Turn 3: Produce structured output — correlated tops + flagged deviations

# COMMAND ----------

import anthropic
import json

client = anthropic.Anthropic()

def run_formation_tops_agent(current_well_summary, offset_summaries):
    """
    3-turn reasoning loop for Formation Tops Agent.
    Turn 1 (Eyes): Present offset well formation patterns
    Turn 2 (Brain): Compare current well and identify deviations
    Turn 3 (Hands): Produce structured correlated tops output
    """

    # Format offset summaries for prompt
    offset_context = json.dumps(
        {w: s for w, s in offset_summaries.items()},
        indent=2
    )
    current_context = json.dumps(current_well_summary, indent=2)

    # --- Turn 1: Establish offset analog pattern ---
    turn1_prompt = f"""You are a senior formation evaluation and geosteering engineer 
with 20 years of experience in North Sea well log interpretation.

You are performing a formation top correlation analysis for the Volve field, 
Norwegian Continental Shelf (Block 15/9).

KEY VOLVE GEOLOGY:
- Draupne Formation: regional cap rock shale — expect GR > 90 API, low RT, higher RHOB
- Hugin Formation: Middle Jurassic reservoir sandstone — expect GR < 50 API, 
  elevated RT (>5 ohm.m in HC-bearing zones), RHOB < 2.35 g/cc
- Hugin depth range across field: ~2750–3700m
- Field is a 4-way dip closure, western part heavily faulted
- Average porosity 21%, N/G up to 90% in Hugin sands

Here are the formation signature statistics from the OFFSET wells:

{offset_context}

Please analyze the offset wells and establish:
1. Where the Draupne cap rock is present in each offset well (depth range)
2. Where the Hugin reservoir is present in each offset well (depth range)  
3. Which offset wells show hydrocarbon-bearing reservoir (elevated RT + low GR)
4. What the consistent formation top pattern looks like across the offset wells"""

    response1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": turn1_prompt}]
    )
    turn1_response = response1.content[0].text

    # --- Turn 2: Compare current well against offset pattern ---
    turn2_prompt = f"""Good analysis of the offset wells. Now examine the CURRENT well 
({current_well_summary['well_name']}) and compare it against the offset analog pattern.

Current well formation signatures:

{current_context}

For the current well, determine:
1. Does the Draupne cap rock appear at the expected depth compared to offsets?
   - If shifted: how much (meters), direction (deeper/shallower), and significance
2. Does the Hugin reservoir appear at the expected depth?
   - If shifted: structural or stratigraphic explanation?
3. Do the reservoir intervals in the current well show similar RT character to offsets?
   - If lower RT: water-bearing zone? Different facies? 
4. Are there intervals in the current well with no offset analog? Flag these.
5. Are there intervals in the offsets with no equivalent in the current well? Flag these.

Consider that depth shifts >100m in this faulted field may be fault-related, 
not necessarily poor well placement. Distinguish structural from stratigraphic deviations."""

    response2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": turn1_prompt},
            {"role": "assistant", "content": turn1_response},
            {"role": "user", "content": turn2_prompt}
        ]
    )
    turn2_response = response2.content[0].text

    # --- Turn 3: Structured output ---
    turn3_prompt = f"""Based on your formation top correlation analysis, produce a 
structured JSON output.

Return ONLY valid JSON — no preamble, no markdown fences, no explanation outside JSON.

Format:
{{
  "current_well": "{current_well_summary['well_name']}",
  "correlation_summary": "3-4 sentence overall assessment of how current well compares to offsets",
  "offset_pattern": {{
    "draupne_top_avg_depth_m": <number>,
    "hugin_top_avg_depth_m": <number>,
    "hugin_base_avg_depth_m": <number>,
    "hc_bearing_offsets": ["well names with elevated RT in Hugin"]
  }},
  "current_well_tops": [
    {{
      "formation": "DRAUPNE|HUGIN_TOP|HUGIN_BASE",
      "picked_depth_m": <number>,
      "offset_avg_depth_m": <number>,
      "depth_shift_m": <number>,
      "shift_direction": "deeper|shallower|on_prognosis",
      "severity": "CRITICAL|MODERATE|MINOR|ON_PROGNOSIS",
      "interpretation": "<1 sentence — structural, stratigraphic, or data gap explanation>"
    }}
  ],
  "flagged_deviations": [
    {{
      "depth_from_m": <number>,
      "depth_to_m": <number>,
      "deviation_type": "<e.g. Missing Reservoir, Unexpected Shale, Depth Shift>",
      "severity": "CRITICAL|MODERATE|MINOR",
      "offset_analog": "<what offsets show at this depth>",
      "current_well_observation": "<what current well shows instead>",
      "recommendation": "<1 sentence>"
    }}
  ],
  "total_deviations": <number>,
  "deviations_critical": <number>,
  "deviations_moderate": <number>,
  "deviations_minor": <number>
}}"""

    response3 = client.messages.create(
        model="claude-sonnet-4-6",
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
        "turn1": turn1_response,
        "turn2": turn2_response,
        "turn3_json": turn3_response
    }

print("Formation Tops Agent defined.")

# COMMAND ----------
# MAGIC %md ### Step 6: Run the Formation Tops Agent

# COMMAND ----------

current_summary  = all_summaries[CURRENT_WELL]
offset_summaries = {w: all_summaries[w] for w in OFFSET_WELLS if w in all_summaries}

print(f"Running Formation Tops Agent...")
print(f"Current well: {CURRENT_WELL}")
print(f"Offset wells: {list(offset_summaries.keys())}")

result = run_formation_tops_agent(current_summary, offset_summaries)

# Parse and preview
try:
    clean_json = result["turn3_json"].strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(clean_json)
    print(f"\n=== Correlation Summary ===")
    print(parsed["correlation_summary"])
    print(f"\nTotal deviations: {parsed['total_deviations']}")
    print(f"CRITICAL: {parsed['deviations_critical']}, MODERATE: {parsed['deviations_moderate']}, MINOR: {parsed['deviations_minor']}")
except Exception as e:
    print(f"Parse error: {e}")
    print(f"Raw: {result['turn3_json'][:500]}")

# COMMAND ----------
# MAGIC %md ### Step 7: Hands — Write results to Silver Delta tables

# COMMAND ----------

# Write formation tops picks
tops_rows = []
for top in parsed.get("current_well_tops", []):
    tops_rows.append({
        "current_well":        CURRENT_WELL,
        "formation":           top.get("formation", ""),
        "picked_depth_m":      float(top.get("picked_depth_m", 0)),
        "offset_avg_depth_m":  float(top.get("offset_avg_depth_m", 0)),
        "depth_shift_m":       float(top.get("depth_shift_m", 0)),
        "shift_direction":     top.get("shift_direction", ""),
        "severity":            top.get("severity", ""),
        "interpretation":      top.get("interpretation", ""),
        "offset_pattern_json": json.dumps(parsed.get("offset_pattern", {})),
        "correlation_summary": parsed.get("correlation_summary", ""),
    })

if tops_rows:
    df_tops = spark.createDataFrame(pd.DataFrame(tops_rows))
    (df_tops.write.format("delta").mode("overwrite")
        .saveAsTable("offset_well_crew.silver_formation_tops"))
    print(f"Written {len(tops_rows)} formation tops to silver_formation_tops")

# Write flagged deviations
deviation_rows = []
for dev in parsed.get("flagged_deviations", []):
    deviation_rows.append({
        "current_well":              CURRENT_WELL,
        "depth_from_m":              float(dev.get("depth_from_m", 0)),
        "depth_to_m":                float(dev.get("depth_to_m", 0)),
        "deviation_type":            dev.get("deviation_type", ""),
        "severity":                  dev.get("severity", ""),
        "offset_analog":             dev.get("offset_analog", ""),
        "current_well_observation":  dev.get("current_well_observation", ""),
        "recommendation":            dev.get("recommendation", ""),
    })

if deviation_rows:
    df_devs = spark.createDataFrame(pd.DataFrame(deviation_rows))
    (df_devs.write.format("delta").mode("overwrite")
        .saveAsTable("offset_well_crew.silver_formation_deviations"))
    print(f"Written {len(deviation_rows)} deviations to silver_formation_deviations")

# COMMAND ----------
# MAGIC %md ### Step 8: Validate results

# COMMAND ----------

print("=== Formation Tops — Current Well vs Offset Average ===")
spark.table("offset_well_crew.silver_formation_tops") \
    .select("formation", "picked_depth_m", "offset_avg_depth_m", "depth_shift_m", "shift_direction", "severity") \
    .orderBy("picked_depth_m") \
    .show(truncate=60)

print("\n=== Flagged Deviations ===")
spark.table("offset_well_crew.silver_formation_deviations") \
    .select("depth_from_m", "depth_to_m", "deviation_type", "severity", "recommendation") \
    .orderBy("depth_from_m") \
    .show(truncate=80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 3 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `offset_well_crew.silver_formation_tops` | Correlated formation tops — current vs offset average |
# MAGIC | `offset_well_crew.silver_formation_deviations` | Flagged depth intervals where current well deviates |
# MAGIC
# MAGIC **Next:** Phase 4 — Drilling Parameters Agent
