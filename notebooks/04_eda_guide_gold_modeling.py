# Databricks notebook source
# MAGIC %md
# MAGIC **Objetivo: **realizar análise exploratória dos incidentes consolidados na camada Silver para identificar padrões de classificação, categorias, técnicas MITRE ATT&CK, Threat Families, organizações e comportamento temporal. Os resultados desta análise serão utilizados para definir a modelagem da camada Gold Analytics.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
df_silver = spark.table("security_data_ai.silver.guide_incidents")
print(f"Total de incidentes na Silver: {df_silver.count():,}")

# COMMAND ----------

# Distribuição dos incidentes por classificação

df_classificacao = (
    df_silver
    .groupBy("IncidentGrade")
    .agg(
        F.count("*").alias("qtd_incidentes")
    )
    .orderBy(F.desc("qtd_incidentes"))
)

display(df_classificacao)

# COMMAND ----------

# Percentual dos incidentes por classificação

total_incidentes = df_silver.count()

df_classificacao = (
    df_silver
    .groupBy("IncidentGrade")
    .agg(
        F.count("*").alias("qtd_incidentes")
    )
    .withColumn(
        "percentual",
        F.round(
            F.col("qtd_incidentes") / F.lit(total_incidentes) * 100,
            2
        )
    )
    .orderBy(F.desc("qtd_incidentes"))
)

display(df_classificacao)

# COMMAND ----------

# Categoria x classificação do incidente

df_categoria_classificacao = (
    df_silver
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        F.explode("categorias").alias("categoria")
    )
    .groupBy(
        "categoria",
        "IncidentGrade"
    )
    .agg(
        F.countDistinct(
            F.struct("OrgId", "IncidentId")
        ).alias("qtd_incidentes")
    )
    .orderBy(
        "categoria",
        F.desc("qtd_incidentes")
    )
)

display(df_categoria_classificacao)

# COMMAND ----------

# Percentual de TP / FP / BP dentro de cada categoria

window_categoria = Window.partitionBy("categoria")

df_categoria_classificacao_pct = (
    df_categoria_classificacao
    .withColumn(
        "total_categoria",
        F.sum("qtd_incidentes").over(window_categoria)
    )
    .withColumn(
        "percentual",
        F.round(
            F.col("qtd_incidentes") /
            F.col("total_categoria") * 100,
            2
        )
    )
    .orderBy(
        "categoria",
        F.desc("qtd_incidentes")
    )
)

display(df_categoria_classificacao_pct)

# COMMAND ----------

# MITRE ATT&CK x classificação do incidente

df_mitre_classificacao = (
    df_silver
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        F.explode("tecnicas_mitre").alias("mitre_technique")
    )
    .groupBy(
        "mitre_technique",
        "IncidentGrade"
    )
    .agg(
        F.countDistinct(
            F.struct("OrgId", "IncidentId")
        ).alias("qtd_incidentes")
    )
    .orderBy(
        F.desc("qtd_incidentes")
    )
)

display(df_mitre_classificacao)

# COMMAND ----------

# ThreatFamily x classificação do incidente

df_threatfamily_classificacao = (
    df_silver
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        F.explode("threat_families").alias("threat_family")
    )
    .groupBy(
        "threat_family",
        "IncidentGrade"
    )
    .agg(
        F.countDistinct(
            F.struct("OrgId", "IncidentId")
        ).alias("qtd_incidentes")
    )
    .orderBy(
        F.desc("qtd_incidentes")
    )
)

display(df_threatfamily_classificacao)

# COMMAND ----------

# Organização x classificação dos incidentes

df_organizacao_classificacao = (
    df_silver
    .groupBy(
        "OrgId",
        "IncidentGrade"
    )
    .agg(
        F.countDistinct("IncidentId").alias("qtd_incidentes")
    )
    .orderBy(
        F.desc("qtd_incidentes")
    )
)

display(df_organizacao_classificacao)

# COMMAND ----------

# Quantidade de organizações distintas

total_organizacoes = (
    df_silver
    .select("OrgId")
    .distinct()
    .count()
)

print(f"Total de organizações: {total_organizacoes:,}")

# COMMAND ----------

# Evolução mensal dos incidentes por classificação

df_evolucao_mensal = (
    df_silver
    .withColumn(
        "ano_mes",
        F.date_format("primeiro_evento", "yyyy-MM")
    )
    .groupBy(
        "ano_mes",
        "IncidentGrade"
    )
    .agg(
        F.count("*").alias("qtd_incidentes")
    )
    .orderBy(
        "ano_mes",
        "IncidentGrade"
    )
)

display(df_evolucao_mensal)

# COMMAND ----------

# Período coberto pelos incidentes

df_silver.select(
    F.min("primeiro_evento").alias("primeiro_incidente"),
    F.max("primeiro_evento").alias("ultimo_incidente")
).display()