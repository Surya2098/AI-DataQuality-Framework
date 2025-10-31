# Databricks notebook source
#All imports

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, count, lit, row_number, when
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType, StringType

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set display options for pandas
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# COMMAND ----------

# To start spark session
spark = SparkSession.builder.appName("AI_DQ_Dashboard").getOrCreate()


# COMMAND ----------

#Load data from DBFS(Source file data)
input_path = "/Volumes/workspace/default/volumne_for_ai-powered_data_quality_&_anomaly_detection_framework/synthetic_sample_data_for_automation_framework.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
display(df.limit(10))


# COMMAND ----------

# Data Cleaning & Business Rules


# 1.Null counts
#from pyspark.sql.functions import col, count, when
from pyspark.sql.types import FloatType, DoubleType

null_counts = df.select([
    count(
        when(
            col(c).isNull() | 
            (isinstance(df.schema[c].dataType, (FloatType, DoubleType)) and isnan(col(c))),
            c
        )
    ).alias(c)
    for c in df.columns
])
null_pdf = null_counts.toPandas().transpose().reset_index()
null_pdf.columns = ["Column", "NullCount"]
display(null_pdf)

# COMMAND ----------

# Plot null counts
plt.figure(figsize=(10,5))
sns.barplot(x="Column", y="NullCount", data=null_pdf)
plt.title("Null Counts per Column")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

# b. Deduplication (based on id)
window_spec = Window.partitionBy("id").orderBy(col("join_date").desc())
df = df.withColumn("_rn", row_number().over(window_spec)).filter(col("_rn") == 1).drop("_rn")

# COMMAND ----------

# c. Business Rules
dq_col = "_dq_failed_rules"
df = df.withColumn(dq_col, lit(""))

# COMMAND ----------

from pyspark.sql.functions import col, when, lit, concat

# 3c. Business Rules
dq_col = "_dq_failed_rules"
df = df.withColumn(dq_col, lit(""))

# Required fields
required_cols = ["id", "user_id"]
for rc in required_cols:
    df = df.withColumn(
        dq_col,
        when(
            col(rc).isNull() | (col(rc) == ""), 
            when(
                col(dq_col) == "", 
                lit(f"{rc}_required")
            ).otherwise(concat(col(dq_col), lit(f",{rc}_required")))
        ).otherwise(col(dq_col))
    )



# COMMAND ----------

# Non-negative numeric columns
from pyspark.sql.functions import col, when, lit, concat

# Non-negative numeric columns
numeric_cols = ["amount", "price", "qty", "fare", "distance", "value"]

for nc in numeric_cols:
    if nc in df.columns:
        df = df.withColumn(
            dq_col,
            when(
                col(nc) < 0,
                when(
                    col(dq_col) == "",
                    lit(f"{nc}_non_negative")
                ).otherwise(concat(col(dq_col), lit(f",{nc}_non_negative")))
            ).otherwise(col(dq_col))
        )




# COMMAND ----------

from pyspark.sql.functions import concat_ws, array, col, when, lit

# Initialize dq_col as array (safer to handle multiple rule appends)
dq_col = "_dq_failed_rules"
df = df.withColumn(dq_col, lit(None).cast("array<string>"))

# Numeric rule checks
numeric_cols = ["amount", "price", "qty", "fare", "distance", "value"]

for nc in numeric_cols:
    if nc in df.columns:
        df = df.withColumn(
            dq_col,
            when(col(nc) < 0,
                 when(col(dq_col).isNull(), array(lit(f"{nc}_non_negative")))
                 .otherwise(concat(col(dq_col), array(lit(f"{nc}_non_negative"))))
                ).otherwise(col(dq_col))
        )

# Finally, join rule names into a single comma-separated string
df = df.withColumn(dq_col, concat_ws(",", col(dq_col)))

# Separate clean and failed
clean_df = df.filter(col(dq_col).isNull() | (col(dq_col) == ""))
failed_df = df.filter(col(dq_col).isNotNull() & (col(dq_col) != ""))

display(failed_df.limit(10))

print(f"Total Rows: {df.count()}, Failing Business Rules: {failed_df.count()}")


# COMMAND ----------

# Plot number of failing rules
fail_count = failed_df.count()
total_count = df.count()
print(f"Total Rows: {total_count}, Failing Business Rules: {fail_count}")

# COMMAND ----------

# DBTITLE 1,AI Anomaly Detection (IsolationForest)
# Convert clean Spark DF to Pandas for AI
feature_cols = [c for c in numeric_cols if c in clean_df.columns]
pdf = clean_df.select(feature_cols).na.fill(0).toPandas()

# COMMAND ----------

# Train IsolationForest
clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
clf.fit(pdf)

# COMMAND ----------

# Compute anomaly scores
pdf["__anomaly_score"] = clf.decision_function(pdf[feature_cols])
pdf["__is_anomaly"] = clf.predict(pdf[feature_cols]) == -1  # True = anomaly


# COMMAND ----------

# Add anomaly scores back to Spark DataFrame
anomaly_df = spark.createDataFrame(pdf)
display(anomaly_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Histogram of anomaly scores
plt.figure(figsize=(10,5))
sns.histplot(pdf["__anomaly_score"], bins=50, kde=True)
plt.title("Distribution of Anomaly Scores")
plt.show()

# COMMAND ----------

#Boxplots for numeric columns with anomalies highlighted
for col_name in feature_cols:
    plt.figure(figsize=(10,4))
    sns.boxplot(x="__is_anomaly", y=col_name, data=pdf)
    plt.title(f"Boxplot of {col_name} (Anomalies Highlighted)")
    plt.show()

# COMMAND ----------

# Top 10 most anomalous records
top_anomalies = pdf.sort_values("__anomaly_score").head(10)
display(top_anomalies)

# COMMAND ----------

# DBTITLE 1,Save outputs to DBFS
#Save outputs to DBFS

output_dir = "/Volumes/workspace/default/volumne_for_ai-powered_data_quality_&_anomaly_detection_framework/"

os.makedirs(output_dir, exist_ok=True)
clean_df.write.mode("overwrite").parquet(output_dir + "clean")
failed_df.write.mode("overwrite").parquet(output_dir + "dq_failed")
anomaly_df.write.mode("overwrite").parquet(output_dir + "scored")

print("All outputs saved under /FileStore/tables/outputs_dashboard/")

# COMMAND ----------

