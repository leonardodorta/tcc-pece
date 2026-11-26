"""
apply_inference.py — quinto estagio: aplica a saida do modelo (`model-output/`) sobre `contract`.

Grava `inference-raw/` (todas as inferencias), `inference-accepted/` (as aceitas) e
`contract-final/` (o contrato consolidado, sem sobrescrever o que as camadas deterministicas
já preencheram).
"""
from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

from domsp import MODELO_AUSENTE
from domsp.contrato import esquema, lago
from domsp.pipeline import inferir


def _dt_iso(valor) -> str:
    """
    Normaliza a particao `dt` para string ISO `AAAA-MM-DD`, venha como date, datetime ou str.

    O Spark infere a coluna de particao ao ler o Parquet e a devolve como `date`; regrava-la
    como valor de particao sem formatar produz o dump de um GregorianCalendar na chave, que
    quebra a partition projection do Athena.
    """
    import datetime as _dt

    if valor is None or valor == "":
        return ""
    if isinstance(valor, _dt.datetime):
        return valor.date().isoformat()
    if isinstance(valor, _dt.date):
        return valor.isoformat()
    return str(valor)[:10]


def main() -> int:
    argumentos = getResolvedOptions(
        sys.argv, ["bucket", "database", "run_id", "model_version"]
    )
    bucket = argumentos["bucket"]
    run_id = argumentos["run_id"]
    # O nome em `getResolvedOptions` usa underscore, igual ao `--model_version` do template.
    modelo = argumentos["model_version"] or MODELO_AUSENTE

    import boto3

    contexto_spark = SparkContext.getOrCreate()
    contexto_glue = GlueContext(contexto_spark)
    spark = contexto_glue.spark_session

    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    # Le a coluna de particao `dt` como STRING (nao infere `date`), senao regrava-la produz o
    # dump de um GregorianCalendar na chave e quebra a partition projection do Athena.
    spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")

    s3 = boto3.client("s3")
    raiz = f"s3://{bucket}"

    # Os mapas vem do residuo, e nao da saida do modelo: a particao e o hash sao propriedade
    # do ato, nao da inferencia.
    particao_por_ato: dict[str, str] = {}
    hash_por_ato: dict[str, str] = {}
    try:
        residuo_df = spark.read.parquet(lago.RESIDUO.caminho(raiz))
        for linha in residuo_df.select(
            "contract_id", "dt", "source_hash"
        ).toLocalIterator():
            # Formatar para ISO `AAAA-MM-DD` garante a string, venha `dt` como date, datetime
            # ou str — ver `_dt_iso`.
            particao_por_ato[linha["contract_id"]] = _dt_iso(linha["dt"])
            hash_por_ato[linha["contract_id"]] = linha["source_hash"] or ""
    except Exception as erro:  # noqa: BLE001 — residuo ausente e caso legitimo
        print(f"residuo ausente ou vazio ({erro}); sem inferencia a aplicar")

    # A saida do modelo (model-output), se houver.
    saida = _ler_jsonl(s3, bucket, lago.SAIDA_MODELO)
    inferencias = inferir.da_saida_do_modelo(
        saida,
        modelo=modelo,
        particao_por_ato=particao_por_ato,
        hash_por_ato=hash_por_ato,
        run_id=run_id,
    )
    aceitas = inferir.aceitas(inferencias)
    print(f"inferencias={len(inferencias)} aceitas={len(aceitas)}")

    _gravar(spark, raiz, lago.INFERENCIA_BRUTA, [i.como_dicionario() for i in inferencias])
    _gravar(spark, raiz, lago.INFERENCIA_ACEITA, [i.como_dicionario() for i in aceitas])

    # A consolidacao: contract + aceita, sem sobrescrever.
    contrato_df = spark.read.parquet(lago.CONTRATO.caminho(raiz))
    atos = [linha.asDict() for linha in contrato_df.toLocalIterator()]
    melhores = inferir.melhor_por_ato(aceitas)
    finais = inferir.consolidar(atos, melhores, camada_modelo="model")
    print(f"contract-final={len(finais)} (com inferencia em {len(melhores)} atos)")

    _gravar(spark, raiz, lago.CONTRATO_FINAL, finais)
    return 0


def _ler_jsonl(s3, bucket: str, prefixo: str) -> list[str]:
    """
    Le todas as linhas de todos os objetos sob um prefixo do S3.

    O Batch Transform escreve um `.out` por arquivo de entrada, e pode fragmentar. Ler o
    prefixo inteiro e concatenar as linhas e o que reconstroi a saida completa.
    """
    paginador = s3.get_paginator("list_objects_v2")
    linhas: list[str] = []
    for pagina in paginador.paginate(Bucket=bucket, Prefix=f"{prefixo}/"):
        for objeto in pagina.get("Contents", []):
            corpo = s3.get_object(Bucket=bucket, Key=objeto["Key"])["Body"].read()
            for linha in corpo.decode("utf-8").splitlines():
                if linha.strip():
                    linhas.append(linha)
    print(f"{prefixo}: {len(linhas)} linhas de saida do modelo")
    return linhas


def _gravar(spark, raiz: str, conjunto: lago.Conjunto, linhas: list[dict]) -> None:
    """Grava um conjunto como Parquet particionado, ou nada se vazio."""
    if not linhas:
        print(f"{conjunto.tabela}: vazio, nada a gravar")
        return
    # Paraleliza em fatias antes de criar o DataFrame — evita estourar spark.rpc.message.maxSize.
    fatias = max(8, (len(linhas) + 1999) // 2000)
    rdd = spark.sparkContext.parallelize(linhas, fatias)
    quadro = spark.createDataFrame(rdd, schema=esquema.struct_type(conjunto))
    destino = conjunto.caminho(raiz)
    escrita = quadro.write.mode("overwrite")
    if conjunto.particionado:
        escrita = escrita.partitionBy(lago.PARTICAO)
    escrita.parquet(destino)
    print(f"{conjunto.tabela}: {len(linhas)} linhas em {destino} (fatias={fatias})")


if __name__ == "__main__":
    # Sem `raise SystemExit(main())`: o Glue trata qualquer SystemExit como erro.
    _codigo = main()
    if _codigo:
        raise SystemExit(_codigo)
