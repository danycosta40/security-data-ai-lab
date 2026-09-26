# Databricks notebook source
from pyspark.sql import functions as F

df_bronze = spark.table("security_data_ai.bronze.guide_train")

# COMMAND ----------

# MAGIC %md
# MAGIC # Construção da Silver

# COMMAND ----------

df_silver = (
    df_bronze
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.first("IncidentGrade", ignorenulls=True).alias("IncidentGrade"),
        F.min("Timestamp").alias("primeiro_evento"),
        F.max("Timestamp").alias("ultimo_evento"),
        F.countDistinct("AlertId").alias("qtd_alertas")
    )
    .withColumn(
        "duracao_horas",
        F.round(
            (F.unix_timestamp("ultimo_evento") -
             F.unix_timestamp("primeiro_evento")) / 3600,
            2
        )
    )
)

# COMMAND ----------

display(df_silver.limit(20))

# COMMAND ----------

categorias_incidente = (
    df_bronze
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.sort_array(
            F.collect_set("Category")
        ).alias("categorias")
    )
    .withColumn(
        "qtd_categorias",
        F.size("categorias")
    )
)

df_silver = (
    df_silver
    .join(
        categorias_incidente,
        ["OrgId", "IncidentId"],
        "left"
    )
)

# COMMAND ----------

display(
    df_silver
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "qtd_alertas",
        "qtd_categorias",
        "categorias"
    )
    .limit(20)
)

# COMMAND ----------

mitre_incidente = (
    df_bronze
    .filter(F.col("MitreTechniques").isNotNull())
    .select(
        "OrgId",
        "IncidentId",
        F.explode(
            F.split("MitreTechniques", ";")
        ).alias("MitreTechnique")
    )
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.sort_array(
            F.collect_set("MitreTechnique")
        ).alias("tecnicas_mitre")
    )
    .withColumn(
        "qtd_tecnicas_mitre",
        F.size("tecnicas_mitre")
    )
)

df_silver = (
    df_silver
    .join(
        mitre_incidente,
        ["OrgId", "IncidentId"],
        "left"
    )
)

# COMMAND ----------

display(
    df_silver
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "qtd_alertas",
        "qtd_categorias",
        "qtd_tecnicas_mitre",
        "tecnicas_mitre",
        "categorias"
    )
    .limit(20)
)

# COMMAND ----------

entidades_incidente = (
    df_bronze
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.sort_array(
            F.collect_set("EntityType")
        ).alias("tipos_entidade")
    )
    .withColumn(
        "qtd_tipos_entidade",
        F.size("tipos_entidade")
    )
)

df_silver = (
    df_silver
    .join(
        entidades_incidente,
        ["OrgId", "IncidentId"],
        "left"
    )
)

# COMMAND ----------

display(
    df_silver
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "qtd_alertas",
        "qtd_categorias",
        "qtd_tecnicas_mitre",
        "qtd_tipos_entidade",
        "tipos_entidade"
    )
    .limit(20)
)

# COMMAND ----------

evidence_incidente = (
    df_bronze
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.sum(
            F.when(F.col("EvidenceRole") == "Related", 1).otherwise(0)
        ).alias("qtd_related"),

        F.sum(
            F.when(F.col("EvidenceRole") == "Impacted", 1).otherwise(0)
        ).alias("qtd_impacted"),

        F.count("*").alias("qtd_evidencias")
    )
    .withColumn(
        "pct_impacted",
        F.round(
            F.col("qtd_impacted") / F.col("qtd_evidencias") * 100,
            2
        )
    )
)

df_silver = (
    df_silver
    .join(
        evidence_incidente,
        ["OrgId", "IncidentId"],
        "left"
    )
)

# COMMAND ----------

display(
    df_silver
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "qtd_alertas",
        "qtd_evidencias",
        "qtd_related",
        "qtd_impacted",
        "pct_impacted"
    )
    .limit(20)
)

# COMMAND ----------

threat_incidente = (
    df_bronze
    .filter(F.col("ThreatFamily").isNotNull())
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.sort_array(
            F.collect_set("ThreatFamily")
        ).alias("threat_families")
    )
    .withColumn(
        "qtd_threat_families",
        F.size("threat_families")
    )
)

df_silver = (
    df_silver
    .join(
        threat_incidente,
        ["OrgId", "IncidentId"],
        "left"
    )
)
# COMMAND ----------

display(
    df_silver
    .filter(F.col("threat_families").isNotNull())
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "threat_families",
        "qtd_threat_families"
    )
    .limit(20)
)

# COMMAND ----------

total_silver = df_silver.count()

chaves_unicas = (
    df_silver
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

duplicados = total_silver - chaves_unicas

print("Total de linhas Silver:", total_silver)
print("Chaves únicas:", chaves_unicas)
print("Duplicados:", duplicados)

# COMMAND ----------

df_silver.printSchema()

# COMMAND ----------

nulos_silver = df_silver.select(
    *[
        F.sum(
            F.when(F.col(c).isNull(), 1).otherwise(0)
        ).alias(c)
        for c in df_silver.columns
    ]
)

display(nulos_silver)

# COMMAND ----------

spark.sql("""
CREATE SCHEMA IF NOT EXISTS security_data_ai.silver
COMMENT 'Curated incident-level security data prepared for analytics and machine learning'
""")

# COMMAND ----------

df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("security_data_ai.silver.guide_incidents")

# COMMAND ----------

df_silver_check = spark.table(
    "security_data_ai.silver.guide_incidents"
)

print(
    "Total persistido na Silver:",
    df_silver_check.count()
)