# Databricks notebook source
# MAGIC %md
# MAGIC # Offset Well Intelligence Crew
# MAGIC ## Phase 4: Reservoir Quality + Drillability Agent
# MAGIC
# MAGIC This agent combines petrophysical and mechanical rock characterization
# MAGIC to answer two questions simultaneously:
# MAGIC - **What is the rock?** (GR + RHOB + NPHI + PEF — reservoir quality)
# MAGIC - **How hard is it to drill?** (DT sonic as mechanical strength proxy)
# MAGIC - **How does the current well compare to offsets at equivalent depths?**
# MAGIC
# MAGIC This mirrors the real Schlumberger FE workflow — formation characterization
# MAGIC handed off to drilling engineering with drillability context.
# MAGIC
# MAGIC **Eyes:** Compute reservoir quality + mechanical strength statistics per interval
# MAGIC **Brain:** Claude compares current well vs offset analogs, reasons about differences
# MAGIC **Hands:** Write reservoir quality + drillability flags to Silver Delta table

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

CURRENT_WELL    = "15_9-F-1C"
OFFSET_WELLS    = ["15_9-F-11A", "15_9-F-1A", "15_9-F-11B", "15_9-F-1B"]
INTERVAL_SIZE   = 50  # meters

# Volve petrophysical thresholds
GR_CLEAN_SAND   = 50    # API
GR_SHALE        = 90    # API
RHOB_POROUS     = 2.35  # g/cc — porous sand upper bound
NPHI_POROUS     = 0.20  # v/v — minimum porosity for reservoir quality sand
RT_HC           = 5.0   # ohm.m — HC indicator threshold
PEF_SANDSTONE   = 2.0   # b/e — sandstone PEF range
PEF_LIMESTONE   = 5.0   # b/e — limestone PEF range

# DT mechanical strength proxies (us/ft)
# Higher DT = slower sonic = softer/more porous rock = easier to drill
# Lower DT = faster sonic = harder/denser rock = harder to drill
DT_SOFT         = 90    # us/ft — soft formation threshold
DT_HARD         = 70    # us/ft — hard formation threshold

print("Configuration loaded.")
print(f"Current well: {CURRENT_WELL}")
print(f"Offset wells: {OFFSET_WELLS}")

# COMMAND ----------
# MAGIC %md ### Step 3: Eyes — Compute Reservoir Quality + Mechanical Strength Statistics

# COMMAND ----------

from pyspark.sql.functions import (
    col, floor, mean, stddev, min, max, count,
    when, lit, round as spark_round, isnull
)
from pyspark.sql import functions as F
import pandas as pd

# Load bronze table
df = spark.table("offset_well_crew.bronze_well_logs")

# Depth interval bins
df = df.withColumn(
    "depth_interval",
    (floor(col("DEPTH") / INTERVAL_SIZE) * INTERVAL_SIZE).cast("int")
)

# Compute petrophysical + mechanical stats per well per interval
df_stats = df.groupBy("WELL", "well_role", "depth_interval").agg(
    count("*").alias("sample_count"),
    # Petrophysical — reservoir quality
    spark_round(mean("GR"), 2).alias("gr_mean"),
    spark_round(mean("RHOB"), 3).alias("rhob_mean"),
    spark_round(stddev("RHOB"), 3).alias("rhob_std"),
    spark_round(mean("NPHI"), 3).alias("nphi_mean"),
    spark_round(mean("RT"), 3).alias("rt_mean"),
    spark_round(mean("PEF"), 3).alias("pef_mean"),
    # Mechanical — drillability proxy
    spark_round(mean("DT"), 2).alias("dt_mean"),
    spark_round(stddev("DT"), 2).alias("dt_std"),
    spark_round(min("DT"), 2).alias("dt_min"),
    spark_round(max("DT"), 2).alias("dt_max"),
    # DT availability
    count(when(isnull("DT"), True)).alias("dt_null_count"),
)

# Reservoir quality classification
df_stats = df_stats.withColumn(
    "reservoir_quality",
    when(
        (col("gr_mean") < GR_CLEAN_SAND) &
        (col("rhob_mean") < RHOB_POROUS) &
        (col("nphi_mean") > NPHI_POROUS),
        "GOOD"
    ).when(
        (col("gr_mean") < GR_CLEAN_SAND) &
        (col("rhob_mean") < RHOB_POROUS),
        "MODERATE"
    ).when(
        col("gr_mean") > GR_SHALE, "SHALE"
    ).otherwise("POOR_OR_TIGHT")
)

# HC potential flag
df_stats = df_stats.withColumn(
    "hc_potential",
    when(
        (col("gr_mean") < GR_CLEAN_SAND) & (col("rt_mean") > RT_HC),
        "HIGH"
    ).when(
        (col("gr_mean") < GR_CLEAN_SAND) & (col("rt_mean") > 2.0),
        "MODERATE"
    ).otherwise("LOW")
)

# Lithology from PEF
df_stats = df_stats.withColumn(
    "lithology_pef",
    when(col("pef_mean") < PEF_SANDSTONE + 0.5, "SANDSTONE")
    .when(col("pef_mean") > PEF_LIMESTONE - 0.5, "LIMESTONE")
    .otherwise("MIXED_OR_SHALE")
)

# Drillability from DT (only for wells with sonic)
df_stats = df_stats.withColumn(
    "drillability",
    when(col("dt_null_count") == col("sample_count"), "NO_SONIC")
    .when(col("dt_mean") > DT_SOFT, "SOFT — easy drilling expected")
    .when(col("dt_mean") < DT_HARD, "HARD — reduced ROP expected")
    .otherwise("MODERATE — normal drilling expected")
)

df_stats = df_stats.orderBy("WELL", "depth_interval")

print(f"Total intervals computed: {df_stats.count()}")

print("\n=== Reservoir quality distribution by well ===")
df_stats.groupBy("WELL", "reservoir_quality").count() \
    .orderBy("WELL", "reservoir_quality").show()

print("\n=== HC potential intervals ===")
df_stats.filter(col("hc_potential") == "HIGH") \
    .select("WELL", "depth_interval", "gr_mean", "rt_mean", "rhob_mean", "reservoir_quality") \
    .orderBy("WELL", "depth_interval").show()

# COMMAND ----------
# MAGIC %md ### Step 4: Compute offset average profile for comparison

# COMMAND ----------

# Compute offset average per depth interval for comparison baseline
df_offset_avg = df_stats.filter(col("WELL").isin(OFFSET_WELLS)) \
    .groupBy("depth_interval").agg(
        spark_round(mean("gr_mean"), 2).alias("offset_gr_avg"),
        spark_round(mean("rhob_mean"), 3).alias("offset_rhob_avg"),
        spark_round(mean("nphi_mean"), 3).alias("offset_nphi_avg"),
        spark_round(mean("rt_mean"), 3).alias("offset_rt_avg"),
        spark_round(mean("dt_mean"), 2).alias("offset_dt_avg"),
        count("*").alias("offset_well_count"),
    )

# Join offset averages to current well intervals
df_current = df_stats.filter(col("WELL") == CURRENT_WELL)

df_comparison = df_current.join(df_offset_avg, on="depth_interval", how="left")

# Compute deviations from offset average
df_comparison = df_comparison \
    .withColumn("gr_dev", spark_round(col("gr_mean") - col("offset_gr_avg"), 2)) \
    .withColumn("rhob_dev", spark_round(col("rhob_mean") - col("offset_rhob_avg"), 3)) \
    .withColumn("rt_dev", spark_round(col("rt_mean") - col("offset_rt_avg"), 3))

print(f"Current well intervals with offset comparison: {df_comparison.count()}")
print("\n=== Current well vs offset average (sample) ===")
df_comparison.select(
    "depth_interval", "gr_mean", "offset_gr_avg", "gr_dev",
    "rhob_mean", "offset_rhob_avg", "rt_mean", "offset_rt_avg",
    "reservoir_quality", "hc_potential", "drillability"
).orderBy("depth_interval").show(10)

# COMMAND ----------
# MAGIC %md ### Step 5: Brain — Claude reasons about reservoir quality + drillability

# COMMAND ----------

import anthropic
import json

client = anthropic.Anthropic()

def run_reservoir_drillability_agent(current_well, offset_wells, stats_pd, comparison_pd):
    """
    3-turn reasoning loop for Reservoir Quality + Drillability Agent.
    Turn 1: Present offset well reservoir + drillability profiles
    Turn 2: Compare current well — identify quality and drillability differences
    Turn 3: Structured output — flagged intervals + recommendations
    """

    # Offset well profiles
    offset_profiles = {}
    for well in offset_wells:
        well_data = stats_pd[stats_pd["WELL"] == well]
        if well_data.empty:
            continue
        offset_profiles[well] = {
            "depth_range": {
                "from": int(well_data["depth_interval"].min()),
                "to": int(well_data["depth_interval"].max())
            },
            "good_reservoir_intervals": well_data[
                well_data["reservoir_quality"] == "GOOD"
            ][["depth_interval", "gr_mean", "rhob_mean", "nphi_mean", "rt_mean", "hc_potential"]
            ].to_dict(orient="records"),
            "hc_high_intervals": well_data[
                well_data["hc_potential"] == "HIGH"
            ][["depth_interval", "gr_mean", "rt_mean", "drillability"]
            ].to_dict(orient="records"),
            "drillability_summary": well_data["drillability"].value_counts().to_dict()
        }

    # Current well comparison profile
    current_profile = comparison_pd[[
        "depth_interval", "gr_mean", "rhob_mean", "nphi_mean", "rt_mean",
        "offset_gr_avg", "offset_rhob_avg", "offset_rt_avg",
        "gr_dev", "rhob_dev", "rt_dev",
        "reservoir_quality", "hc_potential", "lithology_pef", "drillability"
    ]].to_dict(orient="records")

    # --- Turn 1: Offset reservoir + drillability profiles ---
    turn1_prompt = f"""You are a senior formation evaluation engineer with 20 years 
of North Sea experience, combining petrophysical interpretation with drilling 
engineering knowledge.

You are analyzing the Volve field (Block 15/9, Norwegian Continental Shelf).

KEY CONTEXT:
- Hugin Formation reservoir: GR < 50 API, RHOB < 2.35 g/cc, NPHI > 0.20 v/v
- HC-bearing zones: RT > 5 ohm.m in clean sand
- DT (sonic) as drillability proxy: DT > 90 us/ft = soft/easy, DT < 70 us/ft = hard
- PEF: ~1.8 b/e = sandstone, ~5.0 b/e = limestone
- Wells WITHOUT sonic (DT): 15_9-F-1C and 15_9-F-11B

Here are the OFFSET well reservoir quality and drillability profiles:

{json.dumps(offset_profiles, indent=2)}

Analyze the offset wells and establish:
1. Which depth intervals show GOOD reservoir quality across multiple offset wells
2. Which intervals are consistently HC-bearing (HIGH RT + clean GR)
3. What drillability character do the offset wells show through the Hugin section
4. Any intervals where offset wells disagree significantly on reservoir quality"""

    response1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": turn1_prompt}]
    )
    turn1_response = response1.content[0].text

    # --- Turn 2: Compare current well ---
    turn2_prompt = f"""Good. Now compare the CURRENT well ({current_well}) against 
the offset analog pattern.

Current well intervals with offset deviations:

{json.dumps(current_profile, indent=2)}

Analyze and determine:
1. RESERVOIR QUALITY: Where does the current well show better/worse reservoir 
   quality than offset average? What is the likely cause?
2. HC POTENTIAL: Does the current well's RT character match offset HC-bearing zones?
   Where does it deviate and what does that mean?
3. DRILLABILITY: Note that {current_well} has NO sonic log — use RHOB and GR 
   character to infer relative hardness compared to offset DT trends.
   Where would you expect harder vs softer drilling based on the formation character?
4. NOTABLE INTERVALS: Flag any depth interval where the current well shows 
   significantly different character from ALL offset wells — these are the 
   highest-priority findings.

Be specific about depths and magnitudes of deviation."""

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
    turn3_prompt = f"""Based on your reservoir quality and drillability analysis,
produce a structured JSON output.

Return ONLY valid JSON — no preamble, no markdown fences.
Keep all string values under 100 characters.

{{
  "current_well": "{current_well}",
  "analysis_summary": "2 sentence max overall assessment",
  "reservoir_quality_summary": {{
    "best_reservoir_depth_from_m": <number>,
    "best_reservoir_depth_to_m": <number>,
    "overall_quality_vs_offsets": "BETTER|SIMILAR|WORSE",
    "hc_bearing_confirmed": true|false
  }},
  "flagged_intervals": [
    {{
      "depth_from_m": <number>,
      "depth_to_m": <number>,
      "flag_type": "RESERVOIR_QUALITY|HC_POTENTIAL|DRILLABILITY|ANOMALY",
      "severity": "CRITICAL|MODERATE|MINOR",
      "current_well_character": "<max 80 chars>",
      "offset_analog_character": "<max 80 chars>",
      "recommendation": "<max 80 chars>"
    }}
  ],
  "drillability_forecast": [
    {{
      "depth_from_m": <number>,
      "depth_to_m": <number>,
      "expected_drillability": "SOFT|MODERATE|HARD",
      "basis": "<max 80 chars>"
    }}
  ],
  "total_flagged": <number>,
  "flags_critical": <number>,
  "flags_moderate": <number>,
  "flags_minor": <number>
}}"""

    response3 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
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

print("Reservoir Quality + Drillability Agent defined.")

# COMMAND ----------
# MAGIC %md ### Step 6: Run the agent

# COMMAND ----------

stats_pd      = df_stats.toPandas()
comparison_pd = df_comparison.toPandas()

print(f"Running Reservoir Quality + Drillability Agent...")
print(f"Current well: {CURRENT_WELL}")
print(f"Offset wells: {OFFSET_WELLS}")

result = run_reservoir_drillability_agent(
    CURRENT_WELL, OFFSET_WELLS, stats_pd, comparison_pd
)

# Parse and preview
try:
    clean_json = result["turn3_json"].strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(clean_json)
    print(f"\n=== Analysis Summary ===")
    print(parsed["analysis_summary"])
    rqs = parsed["reservoir_quality_summary"]
    print(f"\nBest reservoir: {rqs['best_reservoir_depth_from_m']}–{rqs['best_reservoir_depth_to_m']}m")
    print(f"Quality vs offsets: {rqs['overall_quality_vs_offsets']}")
    print(f"HC-bearing confirmed: {rqs['hc_bearing_confirmed']}")
    print(f"\nTotal flagged: {parsed['total_flagged']}")
    print(f"CRITICAL: {parsed['flags_critical']}, MODERATE: {parsed['flags_moderate']}, MINOR: {parsed['flags_minor']}")
except Exception as e:
    print(f"Parse error: {e}")
    print(f"Raw: {result['turn3_json'][:500]}")

# COMMAND ----------
# MAGIC %md ### Step 7: Hands — Write results to Silver Delta tables

# COMMAND ----------

# Write flagged intervals
flag_rows = []
for flag in parsed.get("flagged_intervals", []):
    flag_rows.append({
        "current_well":             CURRENT_WELL,
        "depth_from_m":             float(flag.get("depth_from_m", 0)),
        "depth_to_m":               float(flag.get("depth_to_m", 0)),
        "flag_type":                flag.get("flag_type", ""),
        "severity":                 flag.get("severity", ""),
        "current_well_character":   flag.get("current_well_character", ""),
        "offset_analog_character":  flag.get("offset_analog_character", ""),
        "recommendation":           flag.get("recommendation", ""),
        "analysis_summary":         parsed.get("analysis_summary", ""),
    })

if flag_rows:
    import pandas as pd
    df_flags = spark.createDataFrame(pd.DataFrame(flag_rows))
    (df_flags.write.format("delta").mode("overwrite")
        .saveAsTable("offset_well_crew.silver_reservoir_flags"))
    print(f"Written {len(flag_rows)} flagged intervals to silver_reservoir_flags")

# Write drillability forecast
drill_rows = []
for d in parsed.get("drillability_forecast", []):
    drill_rows.append({
        "current_well":          CURRENT_WELL,
        "depth_from_m":          float(d.get("depth_from_m", 0)),
        "depth_to_m":            float(d.get("depth_to_m", 0)),
        "expected_drillability": d.get("expected_drillability", ""),
        "basis":                 d.get("basis", ""),
    })

if drill_rows:
    df_drill = spark.createDataFrame(pd.DataFrame(drill_rows))
    (df_drill.write.format("delta").mode("overwrite")
        .saveAsTable("offset_well_crew.silver_drillability_forecast"))
    print(f"Written {len(drill_rows)} drillability intervals to silver_drillability_forecast")

# COMMAND ----------
# MAGIC %md ### Step 8: Validate results

# COMMAND ----------

print("=== Reservoir Quality Flags ===")
spark.table("offset_well_crew.silver_reservoir_flags") \
    .select("depth_from_m", "depth_to_m", "flag_type", "severity",
            "current_well_character", "recommendation") \
    .orderBy("depth_from_m") \
    .show(truncate=70)

print("\n=== Drillability Forecast ===")
spark.table("offset_well_crew.silver_drillability_forecast") \
    .select("depth_from_m", "depth_to_m", "expected_drillability", "basis") \
    .orderBy("depth_from_m") \
    .show(truncate=70)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 4 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `offset_well_crew.silver_reservoir_flags` | Reservoir quality + HC potential flags vs offset analog |
# MAGIC | `offset_well_crew.silver_drillability_forecast` | Expected drillability by depth interval |
# MAGIC
# MAGIC **Next:** Phase 5 — Orchestrator + Synthesis
