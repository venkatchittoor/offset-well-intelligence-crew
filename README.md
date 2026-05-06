# Offset Well Intelligence Crew

A multi-agent AI crew that delivers well intelligence by reasoning across offset well log data — correlating formation tops, assessing reservoir quality, forecasting drillability, and flagging anomalies that have no offset analog.

Built with **Claude API** + **Databricks**. Powered by real-world wireline log interpretation expertise from the Norwegian Continental Shelf.

> *"What do the offset wells tell us about this well — and what should we do about it?"*
> This crew answers that question — systematically, at depth, with reasoning you can audit.

## Business Impact

> Formation evaluation from offset wells typically takes a senior engineer 2–3 days manually. This crew produces equivalent pre-drill intelligence in minutes — log QC, formation top correlation, reservoir quality assessment, and drillability forecast.

## Use Cases

| Mode | Input | Output |
|---|---|---|
| **Pre-drill** | Offset well logs only | Formation top prognosis, reservoir forecast, drillability prediction, risk flags |
| **Post-drill** | Offset logs + current well logs | Deviation analysis, anomaly flags, completion targets, drillability confirmation |

---

## Dataset

**Volve Field Open Dataset** — released by [Equinor](https://www.equinor.com/energy/volve-data-sharing) under an open data licence for research and development.

| Parameter | Detail |
|---|---|
| Field | Volve, Block 15/9, Norwegian Continental Shelf |
| Wells | 15_9-F-1A, 15_9-F-1B, 15_9-F-11A, 15_9-F-11B (offset) · 15_9-F-1C (current) |
| Curves | GR, RHOB, NPHI, RT, PEF, CALI, DT |
| Depth range | 2,600–4,740m |
| Rows | 49,305 at 0.1m sampling |
| Key formations | Draupne Formation (cap rock shale) · Hugin Formation (Middle Jurassic reservoir) |

---

## The Crew

Four specialist agents coordinated by an Orchestrator — same architecture as a consulting firm, applied to formation evaluation.

```
User question / scheduled run
         ↓
    Orchestrator
    (reads agent_registry, routes via LLM)
         ↓
┌────────────────────┬─────────────────────────┬────────────────────────────┐
│   Log QC Agent     │  Formation Tops Agent   │ Reservoir + Drillability   │
│                    │                         │       Agent                │
│ Data quality flags │ Formation top           │ Reservoir quality,         │
│ Bad hole, washout  │ correlation vs offsets  │ HC potential,              │
│ Missing curves     │ Depth shift analysis    │ Drillability forecast      │
└────────────────────┴─────────────────────────┴────────────────────────────┘
         ↓
    Synthesized pre-drill report
    (Gold Delta table + Markdown file)
```

### Agent Registry

| Agent | Domain | Silver Tables |
|---|---|---|
| `log_qc_agent` | Data quality, bad hole, washout, missing curves | `silver_log_qc_flags` |
| `formation_tops_agent` | Formation top correlation, depth shifts, Draupne/Hugin | `silver_formation_tops`, `silver_formation_deviations` |
| `reservoir_drillability_agent` | Reservoir quality, HC potential, drillability forecast | `silver_reservoir_flags`, `silver_drillability_forecast` |

---

## Architecture

### Medallion Lakehouse

```
Bronze                    Silver                         Gold
──────                    ──────                         ────
bronze_well_logs    →     silver_log_qc_flags      →     gold_well_reports
well_registry       →     silver_formation_tops    →     agent_registry
                    →     silver_formation_deviations
                    →     silver_reservoir_flags
                    →     silver_drillability_forecast
```

### Eyes → Brain → Hands (per agent)

Each specialist agent follows the same 3-turn reasoning loop:

| Turn | Role | What happens |
|---|---|---|
| **Eyes** | Observe | PySpark computes per-interval statistics from raw log curves |
| **Brain** | Reason | Claude receives statistics and reasons about causes, severity, geological meaning |
| **Hands** | Act | Claude produces structured JSON → written to Silver Delta table |

### Two Orchestrator Modes

**Full report mode** — no question required. All three agents are called. A complete pre-drill intelligence report is synthesized and written as both a Gold Delta table record and a Markdown file.

**Question mode** — user asks a specific question. The Orchestrator reads the agent registry and uses Claude to route to only the relevant agents, then synthesizes a focused answer.

```python
# Question mode example
result = run_orchestrator(
    question="What are the main drilling risks below 3500m?",
    silver_tables=silver_tables
)
# Orchestrator routes to: reservoir_drillability_agent + log_qc_agent
# Reasoning: "Drilling risks require drillability assessment and QC validation"
```

---

## Domain Knowledge

This crew encodes formation evaluation expertise directly into the reasoning prompts — not hardcoded rules.

### Key Log Curves

| Curve | Measures | Interpretation |
|---|---|---|
| **GR** | Gamma Ray (API) | Shale vs sand — high GR = shale, low GR = clean sand |
| **RHOB** | Bulk Density (g/cc) | Porosity and lithology — low RHOB = porous, light rock |
| **NPHI** | Neutron Porosity (v/v) | Hydrogen content — proxy for porosity |
| **RT** | True Resistivity (ohm.m) | Fluid type — **high RT = hydrocarbons, low RT = brine** |
| **PEF** | Photoelectric Factor (b/e) | Lithology — ~1.8 = sandstone, ~5.0 = limestone |
| **CALI** | Caliper (inches) | Borehole size — CALI >> bit size = washout |
| **DT** | Sonic Transit Time (us/ft) | Rock hardness — high DT = soft/easy to drill |

### Cross-Parameter Reasoning (not threshold alerts)

The same signal means different things in different contexts:

| GR | RT | RHOB | Interpretation |
|---|---|---|---|
| Low (<50 API) | High (>5 ohm.m) | Low (<2.35 g/cc) | HC-bearing reservoir ← the prize |
| Low | Low (<2 ohm.m) | Low | Water-bearing sand |
| High (>90 API) | High | High | Radioactive shale or pyrite |
| High | Low | High | Normal shale |

### Volve-Specific Geology

- **Draupne Formation:** Regional cap rock shale — GR >90 API, low RT, high RHOB. The seal.
- **Hugin Formation:** Middle Jurassic reservoir sandstone — GR <50 API, elevated RT in HC zones, RHOB <2.35 g/cc. Average porosity 21%, N/G up to 90%.
- **Structural setting:** 4-way dip closure, western flank heavily faulted. Depth shifts >100m between wells may be fault-related, not stratigraphic.

---

## Sample Output

### Pre-Drill Mode — Formation Top Prognosis (planned well 15_9-F-1D)

> *Note: This example is illustrative, based on offset well averages computed during the analysis. The crew is architecturally capable of pre-drill prognosis — running with offset wells only would produce formation top depth ranges and risk flags without deviation analysis. Pre-drill implementation is a planned extension.*

> *Based on four offset wells, the planned well is prognosed to encounter Draupne cap rock at ~3,333m and Hugin reservoir entry at ~3,217m. HC-bearing reservoir is confirmed in three of four offsets at equivalent structural positions. Expect MODERATE–HARD drilling through the Draupne section (offset DT ~65 us/ft) transitioning to MODERATE through the Hugin sand (offset DT ~85 us/ft). Critical risk: no offset analog below 3,683m — fault repeat possible. Recommend full log suite including sonic, absent in two offset wells.*

| Formation | Prognosed Depth | Confidence | Basis |
|---|---|---|---|
| Draupne top | ~3,333m | HIGH | Consistent across all 4 offsets ±80m |
| Hugin top | ~3,217m | HIGH | 3 of 4 offsets agree ±50m |
| Hugin base | ~3,683m | MODERATE | F-11B outlier at 4,740m — fault possible |

---

### Post-Drill Mode — Deviation Analysis (well 15_9-F-1C)

> *Well 15_9-F-1C confirms HC-bearing Hugin reservoir from 3,350–3,950m, with base Hugin 267m deeper than offset average — indicating an extended HC column with no direct analog. Best confirmed reservoir at 3,800–3,900m (RHOB 2.31–2.32 g/cc; RT 911–4,665 ohm·m) — primary completion target. Critical anomaly at 4,050m where offsets show prime HC sand but F-1C encounters shale — possible fault offset requiring structural review before perforation.*

| Depth | Severity | Finding | Recommendation |
|---|---|---|---|
| 3,200m | MINOR | Draupne top 133m shallower than offset avg | Structurally elevated — consistent with fault block |
| 3,700m | CRITICAL | RT 3,989 ohm·m — 100× offset analog | Exceptional HC signal — test interval |
| 3,800–3,900m | — | Best reservoir — RHOB 0.3 g/cc lighter than offsets | Primary completion target |
| 4,050m | CRITICAL | Shale where offsets show HC sand | No offset analog — fault repeat or facies change |

See [`sample_output/well_report_15_9_F_1C.md`](sample_output/well_report_15_9_F_1C.md) for the full post-drill report.

---

### Question Mode Routing

```
Question: "Are there any data quality issues that affect interpretation confidence?"
→ Routed to: log_qc_agent only
→ Reasoning: "Directly maps to log QC domain — no reservoir or top data needed"

Question: "Which intervals should be prioritized for completion?"
→ Routed to: reservoir_drillability_agent + log_qc_agent
→ Reasoning: "Completion targets need reservoir quality + QC validation of data reliability"

Question: "What is the expected reservoir quality between 3600m and 3900m?"
→ Routed to: reservoir_drillability_agent + log_qc_agent
→ Reasoning: "Reservoir quality assessment requires drillability data and QC confirmation"
```

---

## Phases

| Phase | Notebook | Description |
|---|---|---|
| 1 | `Phase1_DataIngestion.py` | Load Volve CSVs → Bronze Delta tables via Databricks Volumes |
| 2 | `Phase2_LogQC_Agent.py` | Log QC Agent — flags bad hole, washout, missing curves per well |
| 3 | `Phase3_FormationTops_Agent.py` | Formation Tops Agent — correlates Draupne/Hugin tops vs offset average |
| 4 | `Phase4_ReservoirDrillability_Agent.py` | Reservoir + Drillability Agent — HC potential, completion targets, drillability forecast |
| 5 | `Phase5_Orchestrator.py` | Orchestrator — LLM-driven routing, full report + question mode synthesis |

---

## Delta Table Inventory

| Layer | Table | Description |
|---|---|---|
| Bronze | `bronze_well_logs` | All wells, all curves, with well_role tag |
| Bronze | `well_registry` | Well metadata — role, source file, DT availability |
| Silver | `silver_log_qc_flags` | AI-flagged QC intervals with severity and reasoning |
| Silver | `silver_formation_tops` | Correlated formation tops — current well vs offset average |
| Silver | `silver_formation_deviations` | Depth intervals where current well deviates from offset pattern |
| Silver | `silver_reservoir_flags` | Reservoir quality and HC potential flags |
| Silver | `silver_drillability_forecast` | Expected drilling conditions by depth interval |
| Gold | `gold_well_reports` | Full report + Q&A responses — timestamped, queryable |
| Gold | `agent_registry` | Specialist agent registry with routing keywords |

---

## Tech Stack

| Component | Technology |
|---|---|
| AI reasoning | Claude API (claude-sonnet-4-6) |
| Data platform | Databricks (Serverless) |
| Storage | Delta Lake — Bronze / Silver / Gold |
| Data processing | PySpark |
| Data ingestion | Databricks Volumes |
| Source data | Equinor Volve Open Dataset |
| Language | Python 3 |

---

## Setup

**Prerequisites:** Databricks workspace (Free Edition or above), Anthropic API key, Kaggle account

**1. Download Volve data from Kaggle**
```
https://www.kaggle.com/datasets/youcefziat/wells-from-volve-oil-field
```
Download both `wells_for_training.csv` and `wells_for_prediction.csv`

**2. Upload to Databricks Volume**

In Databricks Catalog → create schema `offset_well_crew` → create Volume `volve_data` → upload both CSVs

**3. Run notebooks in order**

Import each `.py` file into your Databricks Workspace and run Phase 1 through Phase 5 sequentially. Add your Anthropic API key in the configuration cell of each notebook (Phases 2–5).

---

## Acknowledgements

Well log data provided by **Equinor** under the Volve Field Open Dataset licence.
Data available at: [https://www.equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing)

---

## Related Projects

| Repo | Description |
|---|---|
| [data-incident-agent](https://github.com/venkatchittoor/data-incident-agent) | Lakehouse monitoring agent — Eyes/Brain/Hands pattern |
| [pricing-decision-agent](https://github.com/venkatchittoor/pricing-decision-agent) | Autonomous pricing with confidence-gated decisions |
| [customer-behavior-crew](https://github.com/venkatchittoor/customer-behavior-crew) | Multi-agent crew with LLM-driven routing |
| [drilling-npt-agent](https://github.com/venkatchittoor/drilling-npt-agent) | NPT early warning for offshore directional drilling |

---

*Built by [VC](https://github.com/venkatchittoor) — Senior Consultant with O&G domain expertise, building at the intersection of data engineering and AI agents.*

## License & Attribution

This project was independently developed by **Venkat Chittoor** on personal time,
using personal resources, and is not affiliated with or owned by any employer
or client organization.

© 2025 Venkat Chittoor. Licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:** view, learn from, share, and adapt this work for
non-commercial purposes with attribution.

**Commercial use** — including use in client demonstrations, sales engagements,
consulting deliverables, or any revenue-generating activity — requires explicit
written permission from the author.

For commercial licensing inquiries: venkat.chittoor24@gmail.com
