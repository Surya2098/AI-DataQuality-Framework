# AI-Powered Data Quality & Anomaly Detection Framework

Overview

This project provides an end-to-end data quality and anomaly detection
pipeline using PySpark and Isolation Forest. It includes null checks,
deduplication, business rule validations, and AI-driven anomaly
detection.

Features

-   Null & Missing Value Checks
-   Duplicate Removal
-   Business Rule & Range Validation
-   Schema Enforcement
-   AI-based Anomaly Detection (Isolation Forest)

AI Implementation

Isolation Forest isolates records using random partitions.
Anomalies are isolated quicker → higher anomaly score.

Benefits:
- Detect hidden anomalies
- Reduce manual validation
- Improve data reliability

How to Run

Databricks

1.  Upload script + dataset
2.  Update paths
3.  Run the notebook

Local (PyCharm)

    pip install pyspark pandas scikit-learn
    python ai_data_quality_pipeline.py

Tech Stack

-   PySpark
-   Databricks
-   Scikit-learn
-   Python 3.x
