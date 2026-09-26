# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

df_bronze = spark.table("security_data_ai.bronze.guide_train")

# COMMAND ----------

# MAGIC %md
# MAGIC # Análise Exploratória de Dados — GUIDE
# MAGIC
# MAGIC Este notebook realiza a análise exploratória do dataset GUIDE,
# MAGIC com foco na compreensão da granularidade dos dados, distribuição
# MAGIC da variável alvo e avaliação inicial das características disponíveis
# MAGIC para futuras etapas de análise e modelagem.
# MAGIC
# MAGIC A análise parte da camada Bronze:
# MAGIC `security_data_ai.bronze.guide_train`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Granularidade e identificação dos incidentes
# MAGIC
# MAGIC O primeiro objetivo é determinar qual conjunto de campos representa
# MAGIC corretamente um incidente.
# MAGIC
# MAGIC Inicialmente, avaliamos `IncidentId`. Em seguida, verificamos se o mesmo
# MAGIC identificador pode aparecer em organizações diferentes.

# COMMAND ----------

total_incident_ids = (
    df_bronze
    .select("IncidentId")
    .distinct()
    .count()
)

print(f"IncidentIds distintos: {total_incident_ids:,}")

# COMMAND ----------

incidentes_por_org = (
    df_bronze
    .select("OrgId", "IncidentId")
    .distinct()
    .groupBy("IncidentId")
    .agg(
        F.countDistinct("OrgId").alias("qtd_organizacoes")
    )
    .filter(F.col("qtd_organizacoes") > 1)
    .count()
)

print(
    f"IncidentIds presentes em mais de uma organização: "
    f"{incidentes_por_org:,}"
)

# COMMAND ----------

total_incidentes = (
    df_bronze
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

print(
    f"Incidentes únicos considerando OrgId + IncidentId: "
    f"{total_incidentes:,}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validação da chave composta
# MAGIC
# MAGIC Como `IncidentId` pode aparecer em organizações diferentes,
# MAGIC adotamos `OrgId + IncidentId` como chave do incidente.
# MAGIC
# MAGIC A seguir verificamos se, utilizando essa chave, existem incidentes
# MAGIC associados a mais de uma classificação (`IncidentGrade`).

# COMMAND ----------

conflitos_classificacao = (
    df_bronze
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.countDistinct("IncidentGrade").alias("qtd_grades")
    )
    .filter(F.col("qtd_grades") > 1)
    .count()
)

print(
    f"Incidentes com classificações conflitantes "
    f"(OrgId + IncidentId): {conflitos_classificacao:,}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Relação entre incidentes e alertas
# MAGIC
# MAGIC Um incidente pode ser composto por um ou mais alertas.
# MAGIC Para evitar colisões de identificadores entre organizações,
# MAGIC os alertas também são avaliados utilizando `OrgId + AlertId`.

# COMMAND ----------

total_alertas = (
    df_bronze
    .select("OrgId", "AlertId")
    .distinct()
    .count()
)

print(f"Alertas únicos (OrgId + AlertId): {total_alertas:,}")

# COMMAND ----------

alertas_por_incidente = (
    df_bronze
    .select("OrgId", "IncidentId", "AlertId")
    .distinct()
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.countDistinct("AlertId").alias("qtd_alertas")
    )
)

display(
    alertas_por_incidente.describe("qtd_alertas")
)

# COMMAND ----------

percentis_alertas = (
    alertas_por_incidente
    .selectExpr(
        "percentile_approx(qtd_alertas, 0.50) AS mediana",
        "percentile_approx(qtd_alertas, 0.75) AS p75",
        "percentile_approx(qtd_alertas, 0.90) AS p90",
        "percentile_approx(qtd_alertas, 0.95) AS p95",
        "percentile_approx(qtd_alertas, 0.99) AS p99"
    )
)

display(percentis_alertas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Construção da visão exploratória por incidente
# MAGIC
# MAGIC Para as análises seguintes, consolidamos os registros da Bronze
# MAGIC no nível de `OrgId + IncidentId`.
# MAGIC
# MAGIC Esta estrutura é utilizada somente para exploração dos dados.
# MAGIC A transformação definitiva da camada Silver é realizada em
# MAGIC `03_transform_guide_silver`.

# COMMAND ----------

df_incidentes = (
    df_bronze
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.min("Timestamp").alias("primeiro_evento"),
        F.max("Timestamp").alias("ultimo_evento"),
        F.first(
            "IncidentGrade",
            ignorenulls=True
        ).alias("IncidentGrade"),
        F.countDistinct("AlertId").alias("qtd_alertas")
    )
)

# COMMAND ----------

total_incidentes_eda = df_incidentes.count()

print(
    f"Total de incidentes na visão exploratória: "
    f"{total_incidentes_eda:,}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distribuição da variável alvo — IncidentGrade
# MAGIC
# MAGIC `IncidentGrade` representa a classificação atribuída ao incidente.
# MAGIC As classes presentes no dataset são analisadas separadamente,
# MAGIC preservando a distinção entre TruePositive, FalsePositive e
# MAGIC BenignPositive.

# COMMAND ----------

incidentes_sem_grade = (
    df_incidentes
    .filter(F.col("IncidentGrade").isNull())
    .count()
)

print(
    f"Incidentes sem classificação: "
    f"{incidentes_sem_grade:,}"
)

# COMMAND ----------

distribuicao_grade = (
    df_incidentes
    .groupBy("IncidentGrade")
    .count()
    .orderBy(F.desc("count"))
)

display(distribuicao_grade)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Exploração das colunas disponíveis
# MAGIC
# MAGIC A seguir listamos as variáveis existentes na camada Bronze para
# MAGIC apoiar a seleção inicial de características relevantes para
# MAGIC análise e futura modelagem.

# COMMAND ----------

for coluna in df_bronze.columns:
    print(coluna)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Análise de Category
# MAGIC
# MAGIC `Category` representa a categoria associada ao registro de segurança.
# MAGIC Como um incidente pode possuir múltiplos registros, a análise utiliza
# MAGIC combinações distintas de organização, incidente, categoria e
# MAGIC classificação.

# COMMAND ----------

categoria_incidente = (
    df_bronze
    .select(
        "OrgId",
        "IncidentId",
        "Category",
        "IncidentGrade"
    )
    .distinct()
)

# COMMAND ----------

categoria_grade = (
    categoria_incidente
    .filter(F.col("IncidentGrade").isNotNull())
    .groupBy("Category", "IncidentGrade")
    .count()
)

window_categoria = Window.partitionBy("Category")

categoria_percentual = (
    categoria_grade
    .withColumn(
        "total_categoria",
        F.sum("count").over(window_categoria)
    )
    .withColumn(
        "percentual",
        F.round(
            F.col("count") /
            F.col("total_categoria") * 100,
            2
        )
    )
)

display(
    categoria_percentual
    .orderBy(
        "Category",
        F.desc("percentual")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Análise de MITRE ATT&CK
# MAGIC
# MAGIC O campo `MitreTechniques` pode conter uma ou mais técnicas,
# MAGIC separadas por `;`.
# MAGIC
# MAGIC Primeiro avaliamos a cobertura da variável e, posteriormente,
# MAGIC separamos as técnicas para analisar sua diversidade e frequência.

# COMMAND ----------

cobertura_mitre = (
    df_bronze
    .select(
        F.count("*").alias("total"),
        F.count("MitreTechniques").alias("com_mitre"),
        F.sum(
            F.when(
                F.col("MitreTechniques").isNull(),
                1
            ).otherwise(0)
        ).alias("sem_mitre")
    )
)

display(cobertura_mitre)

# COMMAND ----------

mitre_distintas = (
    df_bronze
    .filter(F.col("MitreTechniques").isNotNull())
    .select(
        F.explode(
            F.split(
                F.col("MitreTechniques"),
                ";"
            )
        ).alias("MitreTechnique")
    )
    .select("MitreTechnique")
    .distinct()
)

print(
    f"Técnicas MITRE distintas: "
    f"{mitre_distintas.count():,}"
)

# COMMAND ----------

mitre_frequencia = (
    df_bronze
    .filter(F.col("MitreTechniques").isNotNull())
    .select(
        F.explode(
            F.split(
                F.col("MitreTechniques"),
                ";"
            )
        ).alias("MitreTechnique")
    )
    .groupBy("MitreTechnique")
    .count()
    .orderBy(F.desc("count"))
)

display(
    mitre_frequencia.limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MITRE no nível do incidente
# MAGIC
# MAGIC Como uma técnica pode aparecer repetidamente em diferentes
# MAGIC evidências do mesmo incidente, removemos duplicidades antes
# MAGIC de calcular sua frequência entre incidentes.

# COMMAND ----------

mitre_por_incidente = (
    df_bronze
    .filter(F.col("MitreTechniques").isNotNull())
    .select(
        "OrgId",
        "IncidentId",
        F.explode(
            F.split(
                F.col("MitreTechniques"),
                ";"
            )
        ).alias("MitreTechnique")
    )
    .distinct()
)

mitre_incidente_frequencia = (
    mitre_por_incidente
    .groupBy("MitreTechnique")
    .count()
    .orderBy(F.desc("count"))
)

display(
    mitre_incidente_frequencia.limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Análise de EntityType
# MAGIC
# MAGIC `EntityType` identifica os tipos de entidades associados aos
# MAGIC registros de segurança, como usuários, máquinas, endereços IP,
# MAGIC arquivos e outras entidades.

# COMMAND ----------

entity_type_frequencia = (
    df_bronze
    .groupBy("EntityType")
    .count()
    .orderBy(F.desc("count"))
)

display(
    entity_type_frequencia.limit(50)
)

# COMMAND ----------

entidades_por_incidente = (
    df_bronze
    .select(
        "OrgId",
        "IncidentId",
        "EntityType"
    )
    .distinct()
    .groupBy("OrgId", "IncidentId")
    .agg(
        F.countDistinct(
            "EntityType"
        ).alias("qtd_tipos_entidade")
    )
)

display(
    entidades_por_incidente
    .select("qtd_tipos_entidade")
    .describe()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Análise de EvidenceRole
# MAGIC
# MAGIC Avaliamos a distribuição dos papéis associados às evidências
# MAGIC presentes nos incidentes.

# COMMAND ----------

evidence_role_frequencia = (
    df_bronze
    .groupBy("EvidenceRole")
    .count()
    .orderBy(F.desc("count"))
)

display(evidence_role_frequencia)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Análise de ThreatFamily
# MAGIC
# MAGIC Por fim, avaliamos a disponibilidade de `ThreatFamily`.
# MAGIC A cobertura é analisada tanto no nível dos registros da Bronze
# MAGIC quanto no nível dos incidentes.

# COMMAND ----------

cobertura_threat_family = (
    df_bronze
    .select(
        F.count("*").alias("total"),
        F.count("ThreatFamily").alias("com_threat_family"),
        F.sum(
            F.when(
                F.col("ThreatFamily").isNull(),
                1
            ).otherwise(0)
        ).alias("sem_threat_family")
    )
)

display(cobertura_threat_family)

# COMMAND ----------

incidentes_com_threat_family = (
    df_bronze
    .filter(F.col("ThreatFamily").isNotNull())
    .select("OrgId", "IncidentId")
    .distinct()
    .count()
)

print(
    f"Incidentes com ThreatFamily: "
    f"{incidentes_com_threat_family:,}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusão da análise exploratória
# MAGIC
# MAGIC A exploração permitiu definir `OrgId + IncidentId` como chave
# MAGIC para análise no nível de incidente e avaliar características
# MAGIC relacionadas a alertas, categorias, técnicas MITRE, entidades,
# MAGIC evidências e famílias de ameaça.
# MAGIC
# MAGIC Essas descobertas orientam a transformação Bronze → Silver e
# MAGIC servirão de base para as próximas etapas de análise, preparação
# MAGIC de features e modelagem.