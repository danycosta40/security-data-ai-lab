# Databricks notebook source
# MAGIC %md
# MAGIC ## Gold ML
# MAGIC
# MAGIC **Objetivo:** construir o dataset da camada Gold preparado para Machine Learning,
# MAGIC utilizando apenas incidentes rotulados e features selecionadas após análise de
# MAGIC qualidade, cardinalidade e risco de data leakage.

# COMMAND ----------

from pyspark.sql import functions as F

df_silver = spark.table(
    "security_data_ai.silver.guide_incidents"
)

print(f"Total de incidentes na Silver: {df_silver.count():,}")

# COMMAND ----------

# Mantendo apenas incidentes rotulados

df_ml_base = (
    df_silver
    .filter(F.col("IncidentGrade").isNotNull())
)

print(
    f"Incidentes disponíveis para ML: "
    f"{df_ml_base.count():,}"
)

# COMMAND ----------

df_gold_ml = (
    df_ml_base
    .select(
        # Identificadores para rastreabilidade
        "OrgId",
        "IncidentId",

        # Target
        "IncidentGrade",

        # Usada temporariamente para criar features de tempo
        "primeiro_evento",

        # Features numéricas
        "qtd_alertas",
        "qtd_categorias",
        "qtd_tipos_entidade",
        "qtd_evidencias",
        "qtd_tecnicas_mitre",
        "qtd_threat_families",

        # Features multivaloradas
        "categorias",
        "tipos_entidade",
        "tecnicas_mitre"
    )

    .withColumn(
        "hora_evento",
        F.hour("primeiro_evento")
    )

    .withColumn(
        "dia_semana",
        F.dayofweek("primeiro_evento")
    )

    .withColumn(
        "possui_mitre",
        F.when(
            F.col("qtd_tecnicas_mitre").isNotNull() &
            (F.col("qtd_tecnicas_mitre") > 0),
            1
        ).otherwise(0)
    )

    .withColumn(
        "possui_threat_family",
        F.when(
            F.col("qtd_threat_families").isNotNull() &
            (F.col("qtd_threat_families") > 0),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

df_gold_ml = df_gold_ml.drop("primeiro_evento")

# COMMAND ----------

df_gold_ml.printSchema()

# COMMAND ----------

# Validação de nulos na Gold ML

for coluna in df_gold_ml.columns:
    qtd_nulos = (
        df_gold_ml
        .filter(F.col(coluna).isNull())
        .count()
    )

    print(f"{coluna}: {qtd_nulos:,} nulos")

# COMMAND ----------

# Para a Gold ML, ausência de informação MITRE/ThreatFamily
# é representada como 0 ou array vazio para permitir o processamento pelos modelos.

df_gold_ml = (
    df_gold_ml
    .withColumn(
        "qtd_tecnicas_mitre",
        F.coalesce(F.col("qtd_tecnicas_mitre"), F.lit(0))
    )
    .withColumn(
        "qtd_threat_families",
        F.coalesce(F.col("qtd_threat_families"), F.lit(0))
    )
    .withColumn(
        "tecnicas_mitre",
        F.when(
            F.col("tecnicas_mitre").isNull(),
            F.array().cast("array<string>")
        ).otherwise(F.col("tecnicas_mitre"))
    )
)

# COMMAND ----------

for coluna in [
    "qtd_tecnicas_mitre",
    "qtd_threat_families",
    "tecnicas_mitre"
]:
    qtd_nulos = (
        df_gold_ml
        .filter(F.col(coluna).isNull())
        .count()
    )

    print(f"{coluna}: {qtd_nulos:,} nulos")

# COMMAND ----------

top20_mitre = [
    "T1078",
    "T1078.004",
    "T1566",
    "T1566.002",
    "T1110",
    "T1566.001",
    "T1110.003",
    "T1110.001",
    "T1003",
    "T1087",
    "T1106",
    "T1059.005",
    "T1559",
    "T1485",
    "T1568",
    "T1008",
    "T1098",
    "T1071",
    "T1087.002",
    "T1012"
]

# COMMAND ----------

df_gold_ml = (
    df_gold_ml
    .withColumn(
        "tecnicas_mitre_top20",
        F.array_intersect(
            F.col("tecnicas_mitre"),
            F.array(*[F.lit(x) for x in top20_mitre])
        )
    )
)

# COMMAND ----------

df_gold_ml.select(
    "tecnicas_mitre",
    "tecnicas_mitre_top20"
).show(20, truncate=False)

# COMMAND ----------

df_gold_ml = (
    df_gold_ml
    .withColumn(
        "qtd_mitre_top20",
        F.size("tecnicas_mitre_top20")
    )
    .withColumn(
        "possui_mitre_top20",
        F.when(
            F.size("tecnicas_mitre_top20") > 0,
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

df_gold_ml.select(
    "tecnicas_mitre",
    "tecnicas_mitre_top20",
    "qtd_mitre_top20",
    "possui_mitre_top20"
).show(20, truncate=False)

# COMMAND ----------

# Validação final da Gold ML

total_gold_ml = df_gold_ml.count()

chaves_unicas_gold_ml = (
    df_gold_ml
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

duplicados_gold_ml = total_gold_ml - chaves_unicas_gold_ml

print(f"Total de registros: {total_gold_ml:,}")
print(f"Chaves únicas OrgId + IncidentId: {chaves_unicas_gold_ml:,}")
print(f"Duplicados: {duplicados_gold_ml:,}")
print(f"Total de colunas: {len(df_gold_ml.columns)}")

# COMMAND ----------

# Validação de nulos

for coluna in df_gold_ml.columns:
    qtd_nulos = (
        df_gold_ml
        .filter(F.col(coluna).isNull())
        .count()
    )

    if qtd_nulos > 0:
        print(f"{coluna}: {qtd_nulos:,} nulos")

# COMMAND ----------

# Distribuição do target

display(
    df_gold_ml
    .groupBy("IncidentGrade")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# Persistência da Gold ML

df_gold_ml.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "security_data_ai.gold.guide_incidents_ml"
    )

# COMMAND ----------

df_gold_ml_persisted = spark.table(
    "security_data_ai.gold.guide_incidents_ml"
)

print(
    f"Registros persistidos na Gold ML: "
    f"{df_gold_ml_persisted.count():,}"
)