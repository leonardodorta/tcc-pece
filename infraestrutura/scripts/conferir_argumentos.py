#!/usr/bin/env python3
"""
conferir_argumentos.py — confere se o template passa o que cada script Glue exige.

Roda antes do deploy (chamado por validate.sh). Sai com codigo 1 se algum job divergir.

Para cada `AWS::Glue::Job` do 04-processing.yaml: todo nome pedido em `getResolvedOptions`
tem de estar nos `DefaultArguments` com o prefixo `--`. O contrario (um DefaultArgument que o
script nao pede) tambem e divergencia, porque quase sempre e hifen contra underscore.
"""
from __future__ import annotations

import pathlib
import re

import yaml

RAIZ = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = RAIZ / "cloudformation" / "04-processing.yaml"
SCRIPTS = RAIZ / "jobs"

# Chaves dos DefaultArguments que o script nao pede via `getResolvedOptions` e que ficam fora
# da comparacao. `--additional-python-modules`, `--enable-metrics`,
# `--enable-continuous-cloudwatch-log`, `--job-language`, `--job-bookmark-option` e `--TempDir`
# sao argumentos que o proprio Glue interpreta. `--TRANSFORMERS_CACHE` e `--HF_HOME` nao sao
# argumentos do Glue: estao na lista apenas para serem toleradas caso o template as passe.
RESERVADOS = {
    "--additional-python-modules",
    "--enable-metrics",
    "--enable-continuous-cloudwatch-log",
    "--job-language",
    "--job-bookmark-option",
    "--TempDir",
    "--TRANSFORMERS_CACHE",
    "--HF_HOME",
}


class _SemTags(yaml.SafeLoader):
    """Le o YAML do CloudFormation sem resolver `!Ref`, `!Sub` e afins."""


def _tag(carregador, sufixo, no):
    if isinstance(no, yaml.ScalarNode):
        return f"!{sufixo} {no.value}"
    if isinstance(no, yaml.SequenceNode):
        return carregador.construct_sequence(no)
    return carregador.construct_mapping(no)


_SemTags.add_multi_constructor("!", _tag)


def conferir() -> list[str]:
    """Devolve a lista de divergencias. Vazia significa que o contrato fecha."""
    documento = yaml.load(TEMPLATE.read_text(encoding="utf-8"), Loader=_SemTags)
    problemas: list[str] = []

    jobs = 0
    for recurso in documento["Resources"].values():
        if recurso.get("Type") != "AWS::Glue::Job":
            continue
        jobs += 1

        localizacao = recurso["Properties"]["Command"]["ScriptLocation"]
        nome_do_script = str(localizacao).split("/")[-1]
        arquivo = SCRIPTS / nome_do_script
        if not arquivo.exists():
            problemas.append(f"{nome_do_script}: o template aponta para um script que nao existe")
            continue

        passados = {
            chave
            for chave in recurso["Properties"].get("DefaultArguments", {})
            if chave not in RESERVADOS
        }

        texto = arquivo.read_text(encoding="utf-8")
        achado = re.search(
            r"getResolvedOptions\(\s*sys\.argv,\s*\[([^\]]*)\]", texto
        )
        if achado is None:
            problemas.append(f"{nome_do_script}: nao chama getResolvedOptions")
            continue

        exigidos = {
            f"--{termo.strip().strip(chr(34)).strip(chr(39))}"
            for termo in achado.group(1).split(",")
            if termo.strip()
        }

        for faltando in sorted(exigidos - passados):
            problemas.append(
                f"{nome_do_script}: exige `{faltando}` e o template nao passa. "
                f"O job falha na primeira linha com GlueArgumentError"
            )
        for sobrando in sorted(passados - exigidos):
            problemas.append(
                f"{nome_do_script}: o template passa `{sobrando}`, que o script nao pede. "
                f"Provavelmente hifen contra underscore"
            )

    if jobs == 0:
        problemas.append(
            "nenhum AWS::Glue::Job encontrado no template — o conferidor nao verificou nada"
        )
    return problemas


def main() -> int:
    problemas = conferir()
    if problemas:
        print("── divergencias no contrato de argumentos ──")
        for problema in problemas:
            print(f"  {problema}")
        return 1
    print("  contrato de argumentos: os jobs do template fecham com os scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
