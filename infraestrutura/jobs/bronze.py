"""
bronze.py — segundo estagio: o JSON de `raw/` vira `bronze/publication/` (Parquet tipado).

O HTML de cada publicacao e gravado em `body_html` sem conversao; a conversao para texto
acontece na prata.
"""
from __future__ import annotations

import sys

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext

from domsp.contrato import esquema, lago
from domsp.fonte import edicao as _edicao
from domsp.pipeline import bronze

RAW = "raw"


def main() -> int:
    argumentos = getResolvedOptions(sys.argv, ["bucket", "database", "run_id", "desde", "ate"])
    bucket = argumentos["bucket"]
    run_id = argumentos["run_id"]
    # Resolve a especificacao da janela (`tudo`, data ISO, ou `ultimos:N`) em valores absolutos.
    desde, ate = lago.resolver_janela(argumentos["desde"], argumentos["ate"])

    import boto3

    contexto_spark = SparkContext.getOrCreate()
    contexto_glue = GlueContext(contexto_spark)
    spark = contexto_glue.spark_session

    # Reescreve so a particao tocada; nao mexe no resto do historico.
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")

    s3 = boto3.client("s3")
    paginador = s3.get_paginator("list_objects_v2")

    linhas: list[dict] = []
    edicoes = 0
    fora_da_janela = 0
    repetidas = 0
    # O conjunto vive fora do laco das paginas: a duplicacao ocorre ENTRE arquivos da mesma
    # particao.
    vistas: set[str] = set()
    for pagina in paginador.paginate(Bucket=bucket, Prefix=f"{RAW}/"):
        for objeto in pagina.get("Contents", []):
            chave = objeto["Key"]
            if not chave.endswith(".json"):
                continue
            # A data vem do nome do diretorio `dt=AAAA-MM-DD`: o JSON da fonte nao a traz.
            data = _data_da_chave(chave)
            if data is None:
                continue
            if not lago.na_janela(data, desde, ate):
                fora_da_janela += 1
                continue
            bruto = s3.get_object(Bucket=bucket, Key=chave)["Body"].read()
            ed = _edicao.carregar_bytes(bruto, data)
            for publicacao in bronze.da_edicao(ed, run_id=run_id):
                # Deduplicacao por `publication_id`: o campo `documento` da fonte ou, na falta
                # dele, os 32 primeiros caracteres do SHA-256 do corpo (ver
                # `domsp.pipeline.bronze.da_edicao`). Uma particao de `raw/` pode conter mais de
                # um arquivo para a mesma data, e a mesma publicacao entraria mais de uma vez.
                if publicacao.publication_id in vistas:
                    repetidas += 1
                    continue
                vistas.add(publicacao.publication_id)
                linhas.append(publicacao.como_dicionario())
            edicoes += 1

    print(
        f"janela=[{desde} .. {ate}] edicoes lidas={edicoes} "
        f"fora da janela={fora_da_janela} publicacoes={len(linhas)} "
        f"repetidas descartadas={repetidas}"
    )
    if not linhas:
        print("nenhuma publicacao — nada a gravar no bronze")
        return 0

    # `createDataFrame` de uma lista Python serializa tudo numa unica tarefa RPC, que estoura
    # `spark.rpc.message.maxSize` (128 MB) com o corpus inteiro. Paralelizar em fatias antes de
    # criar o DataFrame evita isso.
    fatias = max(8, (len(linhas) + 1999) // 2000)
    rdd = contexto_spark.parallelize(linhas, fatias)
    quadro = spark.createDataFrame(rdd, schema=esquema.struct_type(lago.PUBLICACAO))
    destino = lago.PUBLICACAO.caminho(f"s3://{bucket}")
    quadro.write.mode("overwrite").partitionBy(lago.PARTICAO).parquet(destino)
    print(f"gravado em {destino} (fatias={fatias})")
    return 0


def _data_da_chave(chave: str) -> str | None:
    """Extrai `AAAA-MM-DD` de uma chave `raw/dt=AAAA-MM-DD/arquivo.json`."""
    for parte in chave.split("/"):
        if parte.startswith("dt="):
            return parte[3:]
    return None



if __name__ == "__main__":
    # Sem `raise SystemExit(main())`: o Glue trata qualquer SystemExit como erro.
    _codigo = main()
    if _codigo:
        raise SystemExit(_codigo)
