#!/usr/bin/env python3
"""
gerar_catalogo.py — gera `03-catalog.yaml` a partir da fonte unica de esquema.

Roda com o `domsp` importavel (PYTHONPATH=src). Sem argumentos, reescreve o template inteiro.
As colunas saem de `esquema.colunas_catalogo`, para que uma coluna acrescentada ao esquema nao
fique esquecida no template.

    PYTHONPATH=src python infraestrutura/scripts/gerar_catalogo.py
    PYTHONPATH=src python infraestrutura/scripts/gerar_catalogo.py --conferir

`--conferir` nao escreve: compara o template existente com o que o esquema produz agora e
devolve 1 se divergirem.
"""
from __future__ import annotations

import difflib
import pathlib
import sys

from domsp.contrato import esquema, lago

DESTINO = (
    pathlib.Path(__file__).resolve().parents[1] / "cloudformation" / "03-catalog.yaml"
)


def _esc(s: str) -> str:
    """Escapa um escalar YAML entre aspas duplas: barra invertida e aspas."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _cabecalho() -> list[str]:
    return [
        'AWSTemplateFormatVersion: "2010-09-09"',
        'Description: "DOM-SP pipeline - Glue database and the external tables,'
        ' one per lake dataset"',
        "",
        "# O catalogo Glue: um database e uma tabela por conjunto de dados do lago (bronze, prata e",
        "# ouro), com particionamento por data via projecao. Gerado por scripts/gerar_catalogo.py a",
        "# partir do esquema unico (esquema.colunas_catalogo) - nao editar a mao.",
        "",
        "Parameters:",
        "  DataLakeBucketName:",
        "    Type: String",
        "  DatabaseName:",
        "    Type: String",
        "  ProjectionStart:",
        "    Type: String",
        '    Default: "2024-01-01"',
        '    Description: "First edition date the partition projection covers, format yyyy-MM-dd"',
        "",
        "Resources:",
        "",
        "  GlueDatabase:",
        "    Type: AWS::Glue::Database",
        "    Properties:",
        "      CatalogId: !Ref AWS::AccountId",
        "      DatabaseInput:",
        "        Name: !Ref DatabaseName",
        '        Description: "DOM-SP contract extraction - bronze, silver and gold layers"',
        '        LocationUri: !Sub "s3://${DataLakeBucketName}/"',
        "",
    ]


def _recurso(conjunto: lago.Conjunto) -> list[str]:
    nome = "".join(p.capitalize() for p in conjunto.tabela.split("_")) + "Table"
    linhas = [
        f"  {nome}:",
        "    Type: AWS::Glue::Table",
        "    Properties:",
        "      CatalogId: !Ref AWS::AccountId",
        "      DatabaseName: !Ref GlueDatabase",
        "      TableInput:",
        f"        Name: {conjunto.tabela}",
        f'        Description: "{_esc(conjunto.descricao)}"',
        "        TableType: EXTERNAL_TABLE",
    ]
    if conjunto.particionado:
        linhas.append("        PartitionKeys:")
        for pk in esquema.particao_catalogo():
            linhas += [
                f'          - Name: {pk["Name"]}',
                f'            Type: {pk["Type"]}',
                f'            Comment: "{_esc(pk["Comment"])}"',
            ]
    linhas += [
        "        Parameters:",
        "          classification: parquet",
        '          "parquet.compression": SNAPPY',
    ]
    if conjunto.particionado:
        for chave, valor in esquema.propriedades_projecao("PROJSTART").items():
            if valor == "PROJSTART,NOW":
                linhas.append(f'          "{chave}": !Sub "${{ProjectionStart}},NOW"')
            else:
                linhas.append(f'          "{chave}": "{valor}"')
    linhas += [
        "        StorageDescriptor:",
        f'          Location: !Sub "s3://${{DataLakeBucketName}}/{conjunto.prefixo}/"',
        "          InputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
        "          OutputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
        "          SerdeInfo:",
        "            SerializationLibrary:"
        " org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
        "          Columns:",
    ]
    for col in esquema.colunas_catalogo(conjunto):
        linhas += [
            f'            - Name: {col["Name"]}',
            f'              Type: {col["Type"]}',
            f'              Comment: "{_esc(col["Comment"])}"',
        ]
    linhas.append("")
    return linhas


def _render() -> str:
    """Monta o YAML do catálogo sem tocar em disco."""
    saida = _cabecalho()
    for conjunto in lago.CONJUNTOS:
        saida += _recurso(conjunto)
    return "\n".join(saida) + "\n"


def main(argv: list[str] | None = None) -> int:
    """
    Gera o catálogo, ou confere que o commitado bate com o esquema.

    `--conferir` não escreve: compara e devolve 1 se divergir, para servir de porta em CI.
    """
    conferir = bool(argv is not None and "--conferir" in argv) or (
        argv is None and "--conferir" in sys.argv[1:]
    )
    gerado = _render()
    if conferir:
        if not DESTINO.exists():
            print(f"FALHA: {DESTINO} não existe; rode sem --conferir para gerar")
            return 1
        atual = DESTINO.read_text(encoding="utf-8")
        if atual == gerado:
            print(f"ok: {DESTINO.name} bate com contrato/esquema.py "
                  f"({len(lago.CONJUNTOS)} tabelas)")
            return 0
        print(f"FALHA: {DESTINO.name} DIVERGE de contrato/esquema.py. "
              f"Rode `gerar_catalogo.py` para regerar.")
        for linha in difflib.unified_diff(
            atual.splitlines(), gerado.splitlines(),
            fromfile=f"{DESTINO.name} (commitado)", tofile="gerado de esquema.py", lineterm="",
        ):
            print(f"  {linha}")
        return 1
    DESTINO.write_text(gerado, encoding="utf-8")
    print(f"escrito {DESTINO} com {len(lago.CONJUNTOS)} tabelas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
