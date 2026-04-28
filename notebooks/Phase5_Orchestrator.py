# Databricks notebook source
# MAGIC %md
# MAGIC # Offset Well Intelligence Crew
# MAGIC ## Phase 5: Orchestrator + Synthesis
# MAGIC
# MAGIC The Orchestrator wires all three specialist agents together and produces
# MAGIC a unified pre-drill intelligence report for well 15_9-F-1C.
# MAGIC
# MAGIC **Two modes:**
# MAGIC - **Question mode:** User asks a specific question — Orchestrator routes to relevant agents
# MAGIC - **Full report mode:** No question — synthesize everything into a complete pre-drill report
# MAGIC
# MAGIC **Agent Registry:**
# MAGIC | Agent | Handles |
# MAGIC |-------|---------|
# MAGIC | log_qc_agent | data quality, bad hole, washout, missing curves |
# MAGIC | formation_tops_agent | formation tops, depth shifts, Draupne, Hugin correlation |
# MAGIC | reservoir_drillability_agent | reservoir quality, HC potential, drillability forecast |

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
REPORT_PATH  = "/Volumes/workspace/offset_well_crew/volve_data/well_report_15_9_F_1C.md"

print(f"Current well: {CURRENT_WELL}")
print(f"Report output path: {REPORT_PATH}")

# COMMAND ----------
# MAGIC %md ### Step 3: Build Agent Registry

# COMMAND ----------

from pyspark.sql import Row

# Agent registry — same pattern as customer-behavior-crew
agent_registry = [
    Row(
        agent_name="log_qc_agent",
        description="Handles data quality questions — bad hole, washout, missing curves, unreliable intervals",
        keywords="qc,quality,bad hole,washout,caliper,missing,sonic,unreliable,trust,usable",
        silver_table="offset_well_crew.silver_log_qc_flags"
    ),
    Row(
        agent_name="formation_tops_agent",
        description="Handles formation top correlation — Draupne cap rock, Hugin reservoir tops, depth shifts vs offsets",
        keywords="formation,top,tops,Draupne,Hugin,depth,shift,correlation,cap rock,reservoir top,structural",
        silver_table="offset_well_crew.silver_formation_tops,offset_well_crew.silver_formation_deviations"
    ),
    Row(
        agent_name="reservoir_drillability_agent",
        description="Handles reservoir quality, HC potential, and drillability — porosity, resistivity, completion targets, drilling risk",
        keywords="reservoir,HC,hydrocarbon,porosity,resistivity,completion,perforate,drillability,ROP,hard,soft,risk",
        silver_table="offset_well_crew.silver_reservoir_flags,offset_well_crew.silver_drillability_forecast"
    ),
]

df_registry = spark.createDataFrame(agent_registry)
(df_registry.write.format("delta").mode("overwrite")
    .saveAsTable("offset_well_crew.agent_registry"))

print("Agent registry written:")
df_registry.show(truncate=60)

# COMMAND ----------
# MAGIC %md ### Step 4: Load all Silver tables

# COMMAND ----------

import pandas as pd
import json

# Load all silver tables into pandas for Orchestrator context
silver_tables = {
    "log_qc_flags":           spark.table("offset_well_crew.silver_log_qc_flags").toPandas(),
    "formation_tops":         spark.table("offset_well_crew.silver_formation_tops").toPandas(),
    "formation_deviations":   spark.table("offset_well_crew.silver_formation_deviations").toPandas(),
    "reservoir_flags":        spark.table("offset_well_crew.silver_reservoir_flags").toPandas(),
    "drillability_forecast":  spark.table("offset_well_crew.silver_drillability_forecast").toPandas(),
}

for name, df in silver_tables.items():
    print(f"{name}: {len(df)} rows")

# COMMAND ----------
# MAGIC %md ### Step 5: Define Orchestrator

# COMMAND ----------

import anthropic
client = anthropic.Anthropic()

def get_registry_context():
    """Return agent registry as formatted string for Orchestrator prompt."""
    registry_pd = spark.table("offset_well_crew.agent_registry").toPandas()
    agents = []
    for _, row in registry_pd.iterrows():
        agents.append({
            "agent_name": row["agent_name"],
            "description": row["description"],
            "keywords": row["keywords"]
        })
    return json.dumps(agents, indent=2)

def get_agent_data(agent_names, silver_tables):
    """Pull relevant silver table data for selected agents."""
    data = {}
    agent_table_map = {
        "log_qc_agent": ["log_qc_flags"],
        "formation_tops_agent": ["formation_tops", "formation_deviations"],
        "reservoir_drillability_agent": ["reservoir_flags", "drillability_forecast"],
    }
    for agent in agent_names:
        tables = agent_table_map.get(agent, [])
        for table in tables:
            if table in silver_tables:
                df = silver_tables[table]
                # Limit columns to avoid token overflow
                data[table] = df.to_dict(orient="records")
    return data

def run_orchestrator(question=None, silver_tables=None):
    """
    Orchestrator with two modes:
    - Question mode: routes to relevant agents based on question
    - Full report mode: synthesizes all agents into pre-drill report
    """
    registry_context = get_registry_context()
    is_full_report = question is None

    if is_full_report:
        print("Mode: FULL REPORT")
        selected_agents = ["log_qc_agent", "formation_tops_agent", "reservoir_drillability_agent"]
        routing_reasoning = "Full report mode — all agents selected."
    else:
        print(f"Mode: QUESTION — '{question}'")
        # --- Routing turn: Claude decides which agents to call ---
        routing_prompt = f"""You are the Orchestrator for an Offset Well Intelligence Crew
analyzing well {CURRENT_WELL} in the Volve field, Norwegian Continental Shelf.

You have three specialist agents available:

{registry_context}

The user has asked: "{question}"

Based on this question, decide which agents are needed to answer it.
Return ONLY a JSON object — no preamble, no markdown:
{{
  "selected_agents": ["agent_name_1", "agent_name_2"],
  "routing_reasoning": "1 sentence explaining why these agents were selected"
}}"""

        routing_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": routing_prompt}]
        )
        routing_raw = routing_response.content[0].text.strip()
        routing_raw = routing_raw.removeprefix("```json").removesuffix("```").strip()
        routing_parsed = json.loads(routing_raw)
        selected_agents = routing_parsed["selected_agents"]
        routing_reasoning = routing_parsed["routing_reasoning"]
        print(f"Routed to: {selected_agents}")
        print(f"Reasoning: {routing_reasoning}")

    # --- Pull relevant agent data ---
    agent_data = get_agent_data(selected_agents, silver_tables)

    # --- Synthesis turn: Claude produces unified output ---
    if is_full_report:
        synthesis_instruction = f"""Produce a complete PRE-DRILL INTELLIGENCE REPORT for well {CURRENT_WELL}.

The report should be structured as follows:

# Pre-Drill Intelligence Report — {CURRENT_WELL}
## Volve Field, Block 15/9, Norwegian Continental Shelf

### 1. Executive Summary
3-4 sentences covering the key findings across all three domains.

### 2. Data Quality Assessment
Summary of log quality — which intervals are reliable, which are suspect.

### 3. Formation Top Correlation
How {CURRENT_WELL} compares to offset wells — Draupne and Hugin tops, depth shifts.

### 4. Reservoir Quality & HC Potential
Best reservoir intervals, HC-bearing zones, water contacts, completion targets.

### 5. Drillability Forecast
Expected drilling conditions by depth — hard/moderate/soft sections with basis.

### 6. Flagged Depth Intervals — Priority List
A consolidated table of ALL flagged intervals across all agents, ranked by severity:
CRITICAL first, then MODERATE, then MINOR.
Format each as: [Depth] | [Severity] | [Issue] | [Recommendation]

### 7. Recommended Actions
Top 5 prioritized recommendations for the drilling/completion team.

Write this as a professional engineering report — concise, specific, depth-referenced."""

    else:
        synthesis_instruction = f"""Answer this specific question about well {CURRENT_WELL}:

"{question}"

Use only the agent data provided. Be specific — reference depths, curve values, 
and offset comparisons where relevant. Keep your answer concise and actionable.
Format as a short professional response — 3-8 sentences or a brief list."""

    synthesis_prompt = f"""You are the Orchestrator for an Offset Well Intelligence Crew.
You have received findings from the following specialist agents: {selected_agents}

Agent findings data:
{json.dumps(agent_data, indent=2, default=str)}

{synthesis_instruction}"""

    synthesis_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )

    report_text = synthesis_response.content[0].text

    return {
        "mode": "full_report" if is_full_report else "question",
        "question": question,
        "selected_agents": selected_agents,
        "routing_reasoning": routing_reasoning,
        "report": report_text
    }

print("Orchestrator defined.")

# COMMAND ----------
# MAGIC %md ### Step 6: Run — Full Report Mode

# COMMAND ----------

print("=" * 60)
print("RUNNING: Full Report Mode")
print("=" * 60)

full_report_result = run_orchestrator(question=None, silver_tables=silver_tables)
print(full_report_result["report"])

# COMMAND ----------
# MAGIC %md ### Step 7: Run — Question Mode (examples)

# COMMAND ----------

questions = [
    "What are the main drilling risks below 3500m?",
    "Which intervals should be prioritized for completion and perforation?",
    "Are there any data quality issues that affect interpretation confidence?",
]

question_results = []
for q in questions:
    print(f"\n{'='*60}")
    print(f"QUESTION: {q}")
    print("="*60)
    result = run_orchestrator(question=q, silver_tables=silver_tables)
    print(result["report"])
    question_results.append(result)

# COMMAND ----------
# MAGIC %md ### Step 8: Hands — Write Gold report to Delta table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType
from datetime import datetime

report_rows = []

# Full report
report_rows.append({
    "well_name":        CURRENT_WELL,
    "report_type":      "full_report",
    "question":         None,
    "selected_agents":  json.dumps(full_report_result["selected_agents"]),
    "routing_reasoning": full_report_result["routing_reasoning"],
    "report_text":      full_report_result["report"],
    "generated_at":     datetime.now().isoformat(),
})

# Question mode results
for result in question_results:
    report_rows.append({
        "well_name":        CURRENT_WELL,
        "report_type":      "question",
        "question":         result["question"],
        "selected_agents":  json.dumps(result["selected_agents"]),
        "routing_reasoning": result["routing_reasoning"],
        "report_text":      result["report"],
        "generated_at":     datetime.now().isoformat(),
    })

df_reports = spark.createDataFrame(pd.DataFrame(report_rows))
(df_reports.write.format("delta").mode("overwrite")
    .saveAsTable("offset_well_crew.gold_well_reports"))

print(f"Written {len(report_rows)} reports to gold_well_reports")

# COMMAND ----------
# MAGIC %md ### Step 9: Save full report as markdown file

# COMMAND ----------

# Write the full report as a markdown file to the Volume
report_md = full_report_result["report"]

# Write to Volume
dbutils.fs.put(REPORT_PATH, report_md, overwrite=True)
print(f"Report saved to: {REPORT_PATH}")

# Preview first 50 lines
lines = report_md.split("\n")[:50]
print("\n--- REPORT PREVIEW (first 50 lines) ---\n")
print("\n".join(lines))

# COMMAND ----------
# MAGIC %md ### Step 10: Validate Gold table

# COMMAND ----------

print("=== Gold Reports Table ===")
df_gold = spark.table("offset_well_crew.gold_well_reports")
df_gold.select("well_name", "report_type", "question", "selected_agents", "generated_at") \
    .show(truncate=60)

print(f"\nTotal reports: {df_gold.count()}")
print(f"Full reports: {df_gold.filter(col('report_type') == 'full_report').count()}")
print(f"Question reports: {df_gold.filter(col('report_type') == 'question').count()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 5 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `offset_well_crew.agent_registry` | Registry of specialist agents + routing keywords |
# MAGIC | `offset_well_crew.gold_well_reports` | Full report + Q&A responses — Gold layer |
# MAGIC
# MAGIC | File | Description |
# MAGIC |------|-------------|
# MAGIC | `well_report_15_9_F_1C.md` | Human-readable pre-drill intelligence report |
# MAGIC
# MAGIC **Next:** Phase 6 — README + LinkedIn Card
