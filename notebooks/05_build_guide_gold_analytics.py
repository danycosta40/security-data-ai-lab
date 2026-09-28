# Databricks notebook source
from pyspark.sql import functions as F
df_silver = spark.table("security_data_ai.silver.guide_incidents")
print(f"Total de incidentes na Silver: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Analytics
# MAGIC
# MAGIC **Objetivo:** construir a camada Gold Analytics a partir dos incidentes
# MAGIC consolidados na camada Silver, disponibilizando estruturas otimizadas para
# MAGIC análises de classificação, categorias, técnicas MITRE ATT&CK e Threat Families.

# COMMAND ----------

# Construção da Gold Analytics
# Seleção explícita das colunas provenientes da Silver

df_gold_analytics = (
    df_silver
    .select(
        # Identificação do incidente
        "OrgId",
        "IncidentId",

        # Classificação
        "IncidentGrade",

        # Informações temporais
        "primeiro_evento",
        "ultimo_evento",

        # Alertas e duração
        "qtd_alertas",
        "duracao_horas",

        # Categorias
        "categorias",
        "qtd_categorias",

        # MITRE ATT&CK
        "tecnicas_mitre",
        "qtd_tecnicas_mitre",

        # Entidades
        "tipos_entidade",
        "qtd_tipos_entidade",

        # Evidências
        "qtd_related",
        "qtd_impacted",
        "qtd_evidencias",
        "pct_impacted",

        # Threat Family
        "threat_families",
        "qtd_threat_families"
    )

    # Features temporais para análise
    .withColumn("ano", F.year("primeiro_evento"))
    .withColumn("mes", F.month("primeiro_evento"))
    .withColumn("ano_mes", F.date_format("primeiro_evento", "yyyy-MM"))

    # Indicadores de cobertura
    .withColumn(
        "possui_mitre",
        F.when(
            F.col("qtd_tecnicas_mitre") > 0,
            True
        ).otherwise(False)
    )
    .withColumn(
        "possui_threat_family",
        F.when(
            F.col("qtd_threat_families") > 0,
            True
        ).otherwise(False)
    )
)

# COMMAND ----------

df_gold_analytics.printSchema()

# COMMAND ----------

df_gold_analytics.groupBy(
    "possui_mitre",
    "possui_threat_family"
).count().orderBy(
    "possui_mitre",
    "possui_threat_family"
).display()

# COMMAND ----------

spark.sql("""
CREATE SCHEMA IF NOT EXISTS security_data_ai.gold
COMMENT 'Analytics and machine learning datasets prepared for security use cases'
""")

# COMMAND ----------

display(spark.sql("SHOW SCHEMAS IN security_data_ai"))

# COMMAND ----------

#salvando dados na tabela Delta
df_gold_analytics.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("security_data_ai.gold.guide_incidents_analytics")

# COMMAND ----------

df_gold_persisted = spark.table(
    "security_data_ai.gold.guide_incidents_analytics"
)

print(f"Registros persistidos na Gold: {df_gold_persisted.count():,}")

# COMMAND ----------

# Validação da granularidade da Gold


total_gold = df_gold_persisted.count()

chaves_unicas_gold = (
    df_gold_persisted
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

duplicados_gold = total_gold - chaves_unicas_gold

print(f"Total de registros: {total_gold:,}")
print(f"Chaves únicas OrgId + IncidentId: {chaves_unicas_gold:,}")
print(f"Duplicados: {duplicados_gold:,}")
print(f"Total de colunas: {len(df_gold_persisted.columns)}")

# COMMAND ----------

# Gold Analytics - Incidente x MITRE ATT&CK

df_gold_mitre = (
    df_gold_persisted
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "ano",
        "mes",
        "ano_mes",
        F.explode("tecnicas_mitre").alias("mitre_technique")
    )
)

display(df_gold_mitre.limit(20))

# COMMAND ----------

# Validação da granularidade Incidente x MITRE

total_mitre = df_gold_mitre.count()

combinacoes_unicas_mitre = (
    df_gold_mitre
    .select(
        "OrgId",
        "IncidentId",
        "mitre_technique"
    )
    .distinct()
    .count()
)

duplicados_mitre = total_mitre - combinacoes_unicas_mitre

incidentes_com_mitre = (
    df_gold_mitre
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

print(f"Total de relações Incidente x MITRE: {total_mitre:,}")
print(f"Combinações únicas: {combinacoes_unicas_mitre:,}")
print(f"Duplicados: {duplicados_mitre:,}")
print(f"Incidentes com MITRE: {incidentes_com_mitre:,}")

# COMMAND ----------

# Persistência da Gold - Incidente x MITRE ATT&CK

df_gold_mitre.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "security_data_ai.gold.guide_incidents_mitre"
    )

# COMMAND ----------

df_gold_mitre_persisted = spark.table(
    "security_data_ai.gold.guide_incidents_mitre"
)

print(
    f"Relações Incidente x MITRE persistidas: "
    f"{df_gold_mitre_persisted.count():,}"
)

# COMMAND ----------

# Gold Analytics - Incidente x Category

df_gold_category = (
    df_gold_persisted
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "ano",
        "mes",
        "ano_mes",
        F.explode("categorias").alias("categoria")
    )
)

# COMMAND ----------

# Validação da granularidade Incidente x Category

total_category = df_gold_category.count()

combinacoes_unicas_category = (
    df_gold_category
    .select(
        "OrgId",
        "IncidentId",
        "categoria"
    )
    .distinct()
    .count()
)

duplicados_category = (
    total_category - combinacoes_unicas_category
)

incidentes_com_category = (
    df_gold_category
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

print(
    f"Total de relações Incidente x Category: "
    f"{total_category:,}"
)

print(
    f"Combinações únicas: "
    f"{combinacoes_unicas_category:,}"
)

print(
    f"Duplicados: "
    f"{duplicados_category:,}"
)

print(
    f"Incidentes com Category: "
    f"{incidentes_com_category:,}"
)

# COMMAND ----------

# Persistência da Gold - Incidente x Category

df_gold_category.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "security_data_ai.gold.guide_incidents_category"
    )

# COMMAND ----------

df_gold_category_persisted = spark.table(
    "security_data_ai.gold.guide_incidents_category"
)

print(
    f"Relações Incidente x Category persistidas: "
    f"{df_gold_category_persisted.count():,}"
)

# COMMAND ----------

# Gold Analytics - Incidente x ThreatFamily

df_gold_threatfamily = (
    df_gold_persisted
    .select(
        "OrgId",
        "IncidentId",
        "IncidentGrade",
        "ano",
        "mes",
        "ano_mes",
        F.explode("threat_families").alias("threat_family")
    )
)

# COMMAND ----------

# Validação da granularidade Incidente x ThreatFamily

total_threatfamily = df_gold_threatfamily.count()

combinacoes_unicas_threatfamily = (
    df_gold_threatfamily
    .select(
        "OrgId",
        "IncidentId",
        "threat_family"
    )
    .distinct()
    .count()
)

duplicados_threatfamily = (
    total_threatfamily - combinacoes_unicas_threatfamily
)

incidentes_com_threatfamily = (
    df_gold_threatfamily
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

print(
    f"Total de relações Incidente x ThreatFamily: "
    f"{total_threatfamily:,}"
)

print(
    f"Combinações únicas: "
    f"{combinacoes_unicas_threatfamily:,}"
)

print(
    f"Duplicados: "
    f"{duplicados_threatfamily:,}"
)

print(
    f"Incidentes com ThreatFamily: "
    f"{incidentes_com_threatfamily:,}"
)

# COMMAND ----------

# Persistência da Gold - Incidente x ThreatFamily

df_gold_threatfamily.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(
        "security_data_ai.gold.guide_incidents_threatfamily"
    )

# COMMAND ----------

df_gold_threatfamily_persisted = spark.table(
    "security_data_ai.gold.guide_incidents_threatfamily"
)

print(
    f"Relações Incidente x ThreatFamily persistidas: "
    f"{df_gold_threatfamily_persisted.count():,}"
)

# COMMAND ----------

# Validação final das tabelas Gold Analytics

tabelas_gold = {
    "guide_incidents_analytics":
        "security_data_ai.gold.guide_incidents_analytics",

    "guide_incidents_category":
        "security_data_ai.gold.guide_incidents_category",

    "guide_incidents_mitre":
        "security_data_ai.gold.guide_incidents_mitre",

    "guide_incidents_threatfamily":
        "security_data_ai.gold.guide_incidents_threatfamily"
}

for nome, tabela in tabelas_gold.items():
    total = spark.table(tabela).count()
    print(f"{nome}: {total:,} registros")