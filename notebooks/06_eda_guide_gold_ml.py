# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC ## EDA para construção da Gold ML
# MAGIC
# MAGIC **Objetivo:** avaliar qualidade, cardinalidade, cobertura, distribuição do target
# MAGIC e risco de data leakage das features disponíveis na camada Silver, definindo
# MAGIC quais atributos serão utilizados na construção da Gold ML.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df_silver = spark.table(
    "security_data_ai.silver.guide_incidents"
)

print(f"Total de incidentes na Silver: {df_silver.count():,}")
print(df_silver.columns)

# COMMAND ----------

# Auditoria inicial de nulos e cardinalidade

colunas_auditoria = [
    "IncidentGrade",
    "primeiro_evento",
    "qtd_alertas",
    "categorias",
    "qtd_categorias",
    "tecnicas_mitre",
    "qtd_tecnicas_mitre",
    "tipos_entidade",
    "qtd_tipos_entidade",
    "qtd_evidencias",
    "threat_families",
    "qtd_threat_families"
]

for coluna in colunas_auditoria:
    qtd_nulos = (
        df_silver
        .filter(F.col(coluna).isNull())
        .count()
    )

    print(f"{coluna}: {qtd_nulos:,} nulos")

# COMMAND ----------

df_ml_base = (
    df_silver
    .filter(F.col("IncidentGrade").isNotNull())
)

print(
    f"Incidentes disponíveis para ML: "
    f"{df_ml_base.count():,}"
)

# COMMAND ----------

df_distribuicao_target = (
    df_ml_base
    .groupBy("IncidentGrade")
    .agg(
        F.count("*").alias("qtd_incidentes")
    )
    .withColumn(
        "percentual",
        F.round(
            F.col("qtd_incidentes") /
            F.lit(df_ml_base.count()) * 100,
            2
        )
    )
    .orderBy(F.desc("qtd_incidentes"))
)

display(df_distribuicao_target)

# COMMAND ----------

# Cardinalidade das principais features categóricas

cardinalidades = {
    "categorias": (
        df_ml_base
        .select(F.explode("categorias").alias("valor"))
        .select("valor")
        .distinct()
        .count()
    ),

    "tecnicas_mitre": (
        df_ml_base
        .select(F.explode("tecnicas_mitre").alias("valor"))
        .select("valor")
        .distinct()
        .count()
    ),

    "tipos_entidade": (
        df_ml_base
        .select(F.explode("tipos_entidade").alias("valor"))
        .select("valor")
        .distinct()
        .count()
    ),

    "threat_families": (
        df_ml_base
        .select(F.explode("threat_families").alias("valor"))
        .select("valor")
        .distinct()
        .count()
    )
}

for feature, qtd in cardinalidades.items():
    print(f"{feature}: {qtd:,} valores distintos")

# COMMAND ----------

df_top_mitre = (
    df_ml_base
    .select(
        "OrgId",
        "IncidentId",
        F.explode("tecnicas_mitre").alias("mitre_technique")
    )
    .groupBy("mitre_technique")
    .agg(
        F.countDistinct(
            F.struct("OrgId", "IncidentId")
        ).alias("qtd_incidentes")
    )
    .orderBy(F.desc("qtd_incidentes"))
)

display(df_top_mitre.limit(20))

# COMMAND ----------

df_top_threatfamily = (
    df_ml_base
    .select(
        "OrgId",
        "IncidentId",
        F.explode("threat_families").alias("threat_family")
    )
    .groupBy("threat_family")
    .agg(
        F.countDistinct(
            F.struct("OrgId", "IncidentId")
        ).alias("qtd_incidentes")
    )
    .orderBy(F.desc("qtd_incidentes"))
)

display(df_top_threatfamily.limit(20))

# COMMAND ----------

# Cobertura acumulada das Threat Families mais frequentes

df_threatfamily_rank = (
    df_top_threatfamily
    .withColumn(
        "rank",
        F.row_number().over(
            Window.orderBy(F.desc("qtd_incidentes"))
        )
    )
)

total_incidentes_threatfamily = (
    df_ml_base
    .filter(F.col("threat_families").isNotNull())
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

for top_n in [10, 20, 50]:
    incidentes_top_n = (
        df_threatfamily_rank
        .filter(F.col("rank") <= top_n)
        .agg(F.sum("qtd_incidentes").alias("total"))
        .collect()[0]["total"]
    )

    percentual = round(
        incidentes_top_n / total_incidentes_threatfamily * 100,
        2
    )

    print(
        f"Top {top_n}: {incidentes_top_n:,} ocorrências "
        f"({percentual}% em relação aos incidentes com ThreatFamily)"
    )

# COMMAND ----------

# Cobertura real por incidente das Threat Families Top N

for top_n in [10, 20, 50]:

    top_families = (
        df_top_threatfamily
        .limit(top_n)
        .select("threat_family")
    )

    incidentes_top_n = (
        df_ml_base
        .select(
            "OrgId",
            "IncidentId",
            F.explode("threat_families").alias("threat_family")
        )
        .join(
            top_families,
            on="threat_family",
            how="inner"
        )
        .select(
            "OrgId",
            "IncidentId"
        )
        .distinct()
        .count()
    )

    percentual = round(
        incidentes_top_n /
        total_incidentes_threatfamily * 100,
        2
    )

    print(
        f"Top {top_n}: "
        f"{incidentes_top_n:,} incidentes únicos "
        f"({percentual}% de cobertura)"
    )

# COMMAND ----------

# Cobertura real por incidente das técnicas MITRE Top N

total_incidentes_mitre = (
    df_ml_base
    .filter(F.col("tecnicas_mitre").isNotNull())
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

for top_n in [10, 20, 50]:

    top_mitre = (
        df_top_mitre
        .limit(top_n)
        .select("mitre_technique")
    )

    incidentes_top_n = (
        df_ml_base
        .select(
            "OrgId",
            "IncidentId",
            F.explode("tecnicas_mitre").alias("mitre_technique")
        )
        .join(
            top_mitre,
            on="mitre_technique",
            how="inner"
        )
        .select(
            "OrgId",
            "IncidentId"
        )
        .distinct()
        .count()
    )

    percentual = round(
        incidentes_top_n /
        total_incidentes_mitre * 100,
        2
    )

    print(
        f"Top {top_n}: "
        f"{incidentes_top_n:,} incidentes únicos "
        f"({percentual}% de cobertura)"
    )

# COMMAND ----------

# Análise temporal dos incidentes para definição das features

df_temporal_ml = (
    df_ml_base
    .select(
        "IncidentGrade",
        F.hour("primeiro_evento").alias("hora_evento"),
        F.dayofweek("primeiro_evento").alias("dia_semana"),
        F.month("primeiro_evento").alias("mes_evento")
    )
)

display(
    df_temporal_ml
    .groupBy(
        "IncidentGrade",
        "hora_evento"
    )
    .count()
    .orderBy(
        "IncidentGrade",
        "hora_evento"
    )
)

# COMMAND ----------

window_classe = Window.partitionBy("IncidentGrade")

df_hora_pct = (
    df_temporal_ml
    .groupBy(
        "IncidentGrade",
        "hora_evento"
    )
    .count()
    .withColumn(
        "total_classe",
        F.sum("count").over(window_classe)
    )
    .withColumn(
        "percentual",
        F.round(
            F.col("count") / F.col("total_classe") * 100,
            2
        )
    )
    .orderBy(
        "IncidentGrade",
        "hora_evento"
    )
)

display(df_hora_pct)

# COMMAND ----------

# Distribuição percentual por dia da semana

df_dia_semana = (
    df_temporal_ml
    .groupBy(
        "IncidentGrade",
        "dia_semana"
    )
    .count()
)

window_classe = Window.partitionBy("IncidentGrade")

df_dia_semana_pct = (
    df_dia_semana
    .withColumn(
        "total_classe",
        F.sum("count").over(window_classe)
    )
    .withColumn(
        "percentual",
        F.round(
            F.col("count") / F.col("total_classe") * 100,
            2
        )
    )
    .orderBy(
        "IncidentGrade",
        "dia_semana"
    )
)

display(df_dia_semana_pct)

# COMMAND ----------

# Distribuição percentual por mês

df_mes = (
    df_temporal_ml
    .groupBy(
        "IncidentGrade",
        "mes_evento"
    )
    .count()
)

window_classe = Window.partitionBy("IncidentGrade")

df_mes_pct = (
    df_mes
    .withColumn(
        "total_classe",
        F.sum("count").over(window_classe)
    )
    .withColumn(
        "percentual",
        F.round(
            F.col("count") / F.col("total_classe") * 100,
            2
        )
    )
    .orderBy(
        "IncidentGrade",
        "mes_evento"
    )
)

display(df_mes_pct)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ### Decisões para a Gold ML
# MAGIC
# MAGIC - `IncidentGrade` será utilizado como target.
# MAGIC - `OrgId` e `IncidentId` serão mantidos apenas para rastreabilidade.
# MAGIC - `hora_evento` e `dia_semana` serão utilizados como features temporais.
# MAGIC - `mes_evento` não será utilizado no baseline devido à forte concentração temporal.
# MAGIC - Category e EntityType serão mantidos como features multivaloradas.
# MAGIC - MITRE será representado inicialmente pelas 20 técnicas mais frequentes.
# MAGIC - ThreatFamily detalhada não será utilizada no baseline devido à baixa cobertura e alta cardinalidade.
# MAGIC - Serão mantidos indicadores de presença de MITRE e ThreatFamily.
# MAGIC - Features com possível risco de data leakage, como `ultimo_evento`, `duracao_horas`,
# MAGIC   `qtd_related`, `qtd_impacted` e `pct_impacted`, não serão utilizadas no baseline.