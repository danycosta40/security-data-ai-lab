# Databricks notebook source

# MAGIC %md
# MAGIC # Ingestão GUIDE — Camada Bronze
# MAGIC
# MAGIC Este notebook realiza a ingestão do dataset público
# MAGIC Microsoft Security Incident Prediction (GUIDE).
# MAGIC
# MAGIC O arquivo CSV armazenado no Volume é lido com Apache Spark
# MAGIC e persistido como uma tabela Delta na camada Bronze.
# MAGIC
# MAGIC **Origem:** `GUIDE_Train.csv`
# MAGIC
# MAGIC **Destino:** `security_data_ai.bronze.guide_train`

# COMMAND ----------

file_path = "/Volumes/security_data_ai/bronze/raw_volume/GUIDE_Train.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leitura do arquivo CSV

# COMMAND ----------

df_guide = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(file_path)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validação inicial do schema

# COMMAND ----------

df_guide.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validação da variável IncidentGrade
# MAGIC
# MAGIC Verificação inicial da distribuição das classificações presentes
# MAGIC no dataset antes da persistência na camada Bronze.

# COMMAND ----------

df_guide \
    .groupBy("IncidentGrade") \
    .count() \
    .orderBy("IncidentGrade") \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validação da quantidade de registros

# COMMAND ----------

total_registros = df_guide.count()

print(f"Total de registros: {total_registros:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persistência na camada Bronze
# MAGIC
# MAGIC Os dados são armazenados como uma tabela Delta gerenciada pelo
# MAGIC Databricks, preservando a granularidade original do dataset.

# COMMAND ----------

df_guide.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("security_data_ai.bronze.guide_train")