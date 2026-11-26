"""
gold.py — sexto estagio (quinto e ultimo job Glue): `silver/contract-final/` vira os tres
conjuntos de ouro (`supplier_month`, `agency_month` e `quality`).
"""
from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

from domsp import MODELO_AUSENTE, VERSAO_REGRAS
from domsp.contrato import esquema, lago
from domsp.pipeline import gold


def main() -> int:
    argumentos = getResolvedOptions(
        sys.argv, ["bucket", "database", "run_id", "model_version"]
    )
    bucket = argumentos["bucket"]
    run_id = argumentos["run_id"]
    modelo = argumentos["model_version"] or MODELO_AUSENTE

    contexto_spark = SparkContext.getOrCreate()
    contexto_glue = GlueContext(contexto_spark)
    spark = contexto_glue.spark_session

    spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    # Le a particao `dt` como STRING (nao infere `date`) — o ouro agrega por mes a partir de
    # `dt`, e um objeto date re-inferido quebraria o corte de mes.
    spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")

    raiz = f"s3://{bucket}"
    contrato_df = spark.read.parquet(lago.CONTRATO_FINAL.caminho(raiz))
    linhas = [registro.asDict() for registro in contrato_df.toLocalIterator()]
    print(f"contract-final lido: {len(linhas)} atos")

    # A procedencia que vai em cada linha de agregado. `source_hash` vazio: o agregado nao vem
    # de um HTML unico.
    procedencia = {
        "rule_version": VERSAO_REGRAS,
        "model_version": modelo,
        "run_id": run_id,
        "source_hash": "",
        "processed_at": _agora(),
    }

    fornecedor = gold.por_fornecedor(linhas, **procedencia)
    orgao = gold.por_orgao(linhas, **procedencia)
    qualidade = [m.como_dicionario() for m in gold.qualidade(linhas, run_id=run_id, modelo=modelo)]
    print(f"supplier_month={len(fornecedor)} agency_month={len(orgao)} quality={len(qualidade)}")

    _gravar(spark, raiz, lago.FORNECEDOR_MES, fornecedor)
    _gravar(spark, raiz, lago.ORGAO_MES, orgao)
    _gravar(spark, raiz, lago.QUALIDADE, qualidade)
    return 0


def _agora():
    import datetime as dt

    return dt.datetime.now(dt.UTC)


def _gravar(spark, raiz: str, conjunto: lago.Conjunto, linhas: list[dict]) -> None:
    """Grava um conjunto de ouro como Parquet, sobrescrevendo a tabela inteira."""
    if not linhas:
        print(f"{conjunto.tabela}: vazio, nada a gravar")
        return
    # Paraleliza em fatias antes de criar o DataFrame — evita estourar spark.rpc.message.maxSize.
    fatias = max(8, (len(linhas) + 1999) // 2000)
    rdd = spark.sparkContext.parallelize(linhas, fatias)
    quadro = spark.createDataFrame(rdd, schema=esquema.struct_type(conjunto))
    destino = conjunto.caminho(raiz)
    quadro.write.mode("overwrite").parquet(destino)
    print(f"{conjunto.tabela}: {len(linhas)} linhas em {destino} (fatias={fatias})")


if __name__ == "__main__":
    # Sem `raise SystemExit(main())`: o Glue trata qualquer SystemExit como erro.
    _codigo = main()
    if _codigo:
        raise SystemExit(_codigo)
