# Databricks notebook source
# MAGIC %md
# MAGIC # Offset Well Intelligence Crew
# MAGIC ## Phase 1: Data Ingestion — Bronze Layer
# MAGIC
# MAGIC Loads Volve field well log data into Delta tables.
# MAGIC - **Current well:** 15_9-F-1C (the well being evaluated)
# MAGIC - **Offset wells:** 15_9-F-11A, 15_9-F-1A, 15_9-F-11B (analog wells for comparison)
# MAGIC
# MAGIC **Curves:** NPHI, RHOB, GR, RT, PEF, CALI, DT (offsets only), DEPTH

# COMMAND ----------
# MAGIC %md ### Step 1: Confirm Volume paths
# MAGIC
# MAGIC Files already uploaded to:
# MAGIC - `/Volumes/workspace/offset_well_crew/volve_data/wells_for_training.csv`
# MAGIC - `/Volumes/workspace/offset_well_crew/volve_data/wells_for_prediction.csv`

# COMMAND ----------
# MAGIC %md ### Step 2: Configure database

# COMMAND ----------

# Create a dedicated database for this project
spark.sql("CREATE DATABASE IF NOT EXISTS offset_well_crew")
spark.sql("USE offset_well_crew")
print("Database ready: offset_well_crew")

# COMMAND ----------
# MAGIC %md ### Step 3: Load CSVs into Spark DataFrames

# COMMAND ----------

from pyspark.sql.functions import lit, when, col

# Load training file (offset wells: 15_9-F-11A, 15_9-F-1A, 15_9-F-11B)
df_training = spark.read.csv(
    "/Volumes/workspace/offset_well_crew/volve_data/wells_for_training.csv",
    header=True,
    inferSchema=True
)

# Load prediction file (current well: 15_9-F-1C + 15_9-F-11B)
df_prediction = spark.read.csv(
    "/Volumes/workspace/offset_well_crew/volve_data/wells_for_prediction.csv",
    header=True,
    inferSchema=True
)

print(f"Training rows: {df_training.count()}")
print(f"Prediction rows: {df_prediction.count()}")

# COMMAND ----------
# MAGIC %md ### Step 4: Preview the data

# COMMAND ----------

print("=== Training file (offset wells) ===")
df_training.printSchema()
df_training.show(5)

print("=== Prediction file (current well) ===")
df_prediction.printSchema()
df_prediction.show(5)

# COMMAND ----------
# MAGIC %md ### Step 5: Align schemas
# MAGIC
# MAGIC Training file has DT (sonic) but prediction file does not.
# MAGIC Add a null DT column to prediction file for schema consistency.

# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType

# Add missing DT column to prediction dataframe as null
df_prediction = df_prediction.withColumn("DT", lit(None).cast(DoubleType()))

# Reorder columns to match training file
column_order = ["NPHI", "RHOB", "GR", "RT", "PEF", "CALI", "DT", "WELL", "DEPTH"]
df_training  = df_training.select(column_order)
df_prediction = df_prediction.select(column_order)

print("Schemas aligned.")

# COMMAND ----------
# MAGIC %md ### Step 6: Assign well roles

# COMMAND ----------

# Tag each row with its role: current well vs offset well
df_training = df_training.withColumn(
    "well_role",
    when(col("WELL") == "15_9-F-1C", "current").otherwise("offset")
)

df_prediction = df_prediction.withColumn(
    "well_role",
    when(col("WELL") == "15_9-F-1C", "current").otherwise("offset")
)

# Confirm role distribution
print("=== Training file well roles ===")
df_training.groupBy("WELL", "well_role").count().show()

print("=== Prediction file well roles ===")
df_prediction.groupBy("WELL", "well_role").count().show()

# COMMAND ----------
# MAGIC %md ### Step 7: Union all wells and write Bronze Delta table

# COMMAND ----------

# Union both files into single dataframe
df_all = df_training.union(df_prediction)

print(f"Total rows across all wells: {df_all.count()}")
df_all.groupBy("WELL", "well_role").count().orderBy("well_role", "WELL").show()

# COMMAND ----------

# Write to Bronze Delta table
(df_all
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("offset_well_crew.bronze_well_logs")
)

print("Bronze table written: offset_well_crew.bronze_well_logs")

# COMMAND ----------
# MAGIC %md ### Step 8: Create well registry table

# COMMAND ----------

from pyspark.sql import Row

# Define the well registry
well_registry_data = [
    Row(well_name="15_9-F-1C",  well_role="current", field="Volve", source_file="wells_for_prediction.csv", has_dt=False),
    Row(well_name="15_9-F-11A", well_role="offset",  field="Volve", source_file="wells_for_training.csv",   has_dt=True),
    Row(well_name="15_9-F-1A",  well_role="offset",  field="Volve", source_file="wells_for_training.csv",   has_dt=True),
    Row(well_name="15_9-F-11B", well_role="offset",  field="Volve", source_file="wells_for_prediction.csv", has_dt=False),
]

df_registry = spark.createDataFrame(well_registry_data)

(df_registry
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("offset_well_crew.well_registry")
)

print("Well registry written: offset_well_crew.well_registry")
df_registry.show()

# COMMAND ----------
# MAGIC %md ### Step 9: Validate Bronze table

# COMMAND ----------

# Read back and validate
df_bronze = spark.table("offset_well_crew.bronze_well_logs")

print(f"Total rows in bronze table: {df_bronze.count()}")
print(f"Total columns: {len(df_bronze.columns)}")
print(f"Columns: {df_bronze.columns}")

print("\n=== Row count per well ===")
df_bronze.groupBy("WELL", "well_role").count().orderBy("well_role", "WELL").show()

print("\n=== Depth range per well ===")
df_bronze.groupBy("WELL").agg(
    {"DEPTH": "min", "DEPTH": "max"}
).orderBy("WELL").show()

print("\n=== Null counts per curve ===")
from pyspark.sql.functions import count, when, isnan, isnull

df_bronze.select([
    count(when(isnull(c), c)).alias(c)
    for c in ["NPHI", "RHOB", "GR", "RT", "PEF", "CALI", "DT"]
]).show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 1 Complete ✅
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | `offset_well_crew.bronze_well_logs` | All wells, all curves, with well_role tag |
# MAGIC | `offset_well_crew.well_registry` | Well metadata — role, source, DT availability |
# MAGIC
# MAGIC **Next:** Phase 2 — Log QC Agent
