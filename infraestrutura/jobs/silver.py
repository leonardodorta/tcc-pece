"""
silver.py — terceiro estagio: `bronze/publication/` vira as saidas da prata.

Grava tres conjuntos em Parquet (`contract`, `discard` e `residual`) e, a partir do residuo, os
objetos JSONL de `model-input/` que o Batch Transform le.
"""
from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

from domsp.contrato import esquema, lago
from domsp.pipeline import silver


def main() -> int:
    argumentos = getResolvedOptions(
        sys.argv, ["bucket", "database", "run_id", "desde", "ate", "fatias_modelo"]
    )
    bucket = argumentos["bucket"]
    run_id = argumentos["run_id"]
    # Mesma resolucao do bronze, pela mesma funcao.
    desde, ate = lago.resolver_janela(argumentos["desde"], argumentos["ate"])

    contexto_spark = SparkContext.getOrCreate()
    contexto_glue = GlueContext(contexto_spark)
    spark = contexto_glue.spark_session

    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    # A coluna de particao `dt` e STRING no esquema. Sem esta config, o Spark infere o tipo da
    # particao ao ler (`dt=2026-03-09` vira `date`), e regrava-la particionando produz o dump de
    # um GregorianCalendar na chave, que quebra a partition projection do Athena.
    spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")

    origem = lago.PUBLICACAO.caminho(f"s3://{bucket}")
    bronze_df = spark.read.parquet(origem)

    # A JANELA: recorta por particao `dt`, o que vira partition pruning (o Spark nem abre os
    # Parquet de fora do intervalo). Comparacao de string porque `AAAA-MM-DD` ordena igual em
    # texto e em tempo.
    if desde != lago.SEM_LIMITE:
        bronze_df = bronze_df.filter(bronze_df[lago.PARTICAO] >= desde)
    if ate != lago.SEM_LIMITE:
        bronze_df = bronze_df.filter(bronze_df[lago.PARTICAO] <= ate)
    print(f"janela=[{desde} .. {ate}]")

    atos: list = []
    descartes: list = []
    for registro in bronze_df.toLocalIterator():
        resultado = silver.de_publicacao(registro.asDict(), run_id=run_id)
        if isinstance(resultado, silver.Ato):
            atos.append(resultado)
        else:
            descartes.append(resultado)

    residuos = silver.residuo_de(atos)
    print(
        f"atos={len(atos)} descartes={len(descartes)} residuo={len(residuos)}"
    )

    _gravar(spark, bucket, lago.CONTRATO, [a.como_dicionario() for a in atos])
    _gravar(spark, bucket, lago.DESCARTE, [d.como_dicionario() for d in descartes])
    _gravar(spark, bucket, lago.RESIDUO, [r.como_dicionario() for r in residuos])

    # A entrada do modelo: o residuo como JSONL que o Batch Transform le. `fatias_modelo` e o
    # numero de objetos em que o residuo e repartido.
    _gravar_jsonl(s3_do_spark(spark), bucket, residuos, int(argumentos["fatias_modelo"]))
    return 0


def s3_do_spark(_spark):
    """Devolve um cliente boto3 do S3. O argumento `_spark` nao e usado."""
    import boto3

    return boto3.client("s3")


def _limpar_prefixo(s3, bucket: str, prefixo: str) -> None:
    """Apaga todos os objetos de um prefixo descartavel, em lotes de 1.000 (limite da API)."""
    paginador = s3.get_paginator("list_objects_v2")
    for pagina in paginador.paginate(Bucket=bucket, Prefix=f"{prefixo}/"):
        chaves = [{"Key": o["Key"]} for o in pagina.get("Contents", [])]
        if chaves:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": chaves})


def _gravar_jsonl(s3, bucket: str, residuos: list, fatias: int = 1) -> None:
    """
    Escreve o residuo em `model-input/`, repartido em `fatias` objetos.

    O Batch Transform distribui por OBJETO S3, nao por linha: um `input.jsonl` unico so seria
    processado por uma instancia, mesmo com `InstanceCount` maior. Repartir em mais fatias que
    instancias reduz o desequilibrio quando uma fatia concentra atos longos. A distribuicao usa
    `i % fatias` (alternada) em vez de blocos contiguos, pois os atos chegam agrupados por
    edicao e o tamanho do texto e correlacionado dentro de uma edicao.
    """
    from domsp.pipeline import inferir

    linhas = list(inferir.para_jsonl(residuos))
    if not linhas:
        print("residuo vazio — nenhuma entrada para o modelo")
        return

    # Apaga o que havia antes: senao uma execucao com menos fatias que a anterior deixaria
    # objetos excedentes no prefixo, que o Batch Transform (que le o prefixo inteiro) os
    # processaria junto.
    _limpar_prefixo(s3, bucket, lago.ENTRADA_MODELO)
    # E tambem a saida, pelo mesmo motivo: sem limpar, cada execucao somaria a saida as
    # anteriores e produziria linhas duplicadas na prata final.
    _limpar_prefixo(s3, bucket, lago.SAIDA_MODELO)

    fatias = max(1, min(fatias, len(linhas)))
    baldes: list[list[str]] = [[] for _ in range(fatias)]
    for i, linha in enumerate(linhas):
        baldes[i % fatias].append(linha)

    for n, balde in enumerate(baldes):
        if not balde:
            continue
        # Largura fixa: a ordem lexicografica das chaves fica igual a numerica.
        chave = f"{lago.ENTRADA_MODELO}/parte-{n:03d}.jsonl"
        s3.put_object(Bucket=bucket, Key=chave, Body=("\n".join(balde) + "\n").encode("utf-8"))
    print(f"model-input: {len(linhas)} atos em {fatias} fatia(s) sob s3://{bucket}/"
          f"{lago.ENTRADA_MODELO}/")


def _gravar(spark, bucket: str, conjunto: lago.Conjunto, linhas: list[dict]) -> None:
    """
    Grava um conjunto da prata como Parquet particionado, ou nao faz nada se vazio.

    Vazio nao e erro: uma execucao pode nao produzir nenhum descarte de um tipo, ou nenhum
    residuo. Criar um DataFrame vazio com esquema e grava-lo apagaria a particao anterior sem
    ganho — pular preserva o historico.
    """
    if not linhas:
        print(f"{conjunto.tabela}: vazio, nada a gravar")
        return
    # Paraleliza em fatias antes de criar o DataFrame: evita estourar `spark.rpc.message.maxSize`
    # (128 MB) quando a lista e grande.
    fatias = max(8, (len(linhas) + 1999) // 2000)
    rdd = spark.sparkContext.parallelize(linhas, fatias)
    quadro = spark.createDataFrame(rdd, schema=esquema.struct_type(conjunto))
    destino = conjunto.caminho(f"s3://{bucket}")
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
