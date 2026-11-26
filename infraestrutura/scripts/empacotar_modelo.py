#!/usr/bin/env python3
"""
empacotar_modelo.py — monta o `model.tar.gz` de serving a partir do checkpoint treinado.

Recusa empacotar um modelo não aprovado pelo critério de aprovação, a menos que seja autorizado
explicitamente com `DOMSP_EM_OBSERVACAO=sim`; nesse caso grava a não aprovação dentro do
próprio artefato (`metadata.json`: `avaliacao.aprovado=false`, `situacao="em observacao..."`),
para que o pacote não possa ser confundido com um aprovado quando consumido depois.

Entradas esperadas, produzidas fora deste repositório (o script falha se faltarem):
    modelos/checkpoints/large/         checkpoint treinado (com o metadata do treino)
    resultados/metricas_modelo.json    métricas do treino, com o bloco `avaliacao`
    resultados/manifest_corpus.json    manifesto do corpus
    dados/gabarito/manifest.json       manifesto do gabarito
A saída é gravada em `modelos/pacote/`.

Uso:
    .venv/bin/python infraestrutura/scripts/empacotar_modelo.py            # confere e para
    DOMSP_EM_OBSERVACAO=sim .venv/bin/python infraestrutura/scripts/empacotar_modelo.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from domsp.modelo import empacotar as emp  # noqa: E402

CHECKPOINT = RAIZ / "modelos" / "checkpoints" / "large"
PACOTE = RAIZ / "modelos" / "pacote"
RESULTADOS = RAIZ / "resultados"
GABARITO = RAIZ / "dados" / "gabarito"

SITUACAO_OBSERVACAO = "em observacao - nao aprovado pelo criterio de aprovacao"
SITUACAO_APROVADO = "aprovado pelo criterio de aprovacao"


def _ler(caminho: pathlib.Path) -> dict:
    if not caminho.exists():
        raise SystemExit(f"ERRO: nao achei {caminho}")
    return json.loads(caminho.read_text(encoding="utf-8"))


def principal() -> int:
    metricas = _ler(RESULTADOS / "metricas_modelo.json")
    manifesto_corpus = _ler(RESULTADOS / "manifest_corpus.json")
    manifesto_gabarito = _ler(GABARITO / "manifest.json")

    avaliacao = dict(metricas.get("avaliacao") or {})
    if not avaliacao:
        raise SystemExit(
            "ERRO: metricas_modelo.json nao tem o bloco `avaliacao`. Rode a avaliacao antes: "
            "empacotar sem avaliacao produziria um pacote sem procedencia de qualidade"
        )

    aprovado = bool(avaliacao.get("aprovado"))
    reprovados = [
        item
        for item in avaliacao.get("itens_do_criterio", [])
        if item.get("resultado") is False
    ]

    print("== estado da avaliacao ==")
    print(f"  checkpoint ......... {CHECKPOINT}")
    print(f"  aprovado ........... {aprovado}")
    print(f"  teto da rotulagem .. {avaliacao.get('teto_da_rotulagem')}")
    for item in avaliacao.get("itens_do_criterio", []):
        marca = {True: "ok  ", False: "FALHA", None: "n/a "}[item.get("resultado")]
        print(f"  {marca} {item.get('item')}  {item.get('detalhe')}")

    if not aprovado and os.environ.get("DOMSP_EM_OBSERVACAO") != "sim":
        print("")
        print("PARADO. O modelo NAO passou pelo criterio de aprovacao:")
        for item in reprovados:
            print(f"  - {item.get('item')}: {item.get('detalhe')}")
        print("")
        print("Nada foi empacotado. Para empacotar como NAO APROVADO, em observacao:")
        print("  DOMSP_EM_OBSERVACAO=sim .venv/bin/python "
              "infraestrutura/scripts/empacotar_modelo.py")
        return 0

    metadados = emp.montar_metadados(
        CHECKPOINT,
        manifesto_corpus=manifesto_corpus,
        manifesto_gabarito=manifesto_gabarito,
        metricas_modelo=metricas,
        avaliacao=avaliacao,
    )
    metadados["situacao"] = SITUACAO_APROVADO if aprovado else SITUACAO_OBSERVACAO
    metadados["itens_reprovados"] = reprovados

    print("")
    print(f"== montando o pacote ({metadados['situacao']}) ==")
    pacote = emp.empacotar(CHECKPOINT, PACOTE, metadados=metadados)
    print(f"  caminho .... {pacote.caminho}")
    print(f"  tamanho .... {pacote.bytes / 2**20:.1f} MiB")
    print(f"  sha256 ..... {pacote.sha256}")

    print("")
    print("== conferindo o tar ==")
    conferencia = emp.conferir(pacote.caminho)
    for chave, valor in conferencia.items():
        if chave != "arquivos":
            print(f"  {chave:24} {valor}")
    for nome in conferencia["arquivos"]:
        print(f"    {nome}")
    if not conferencia["valido"]:
        raise SystemExit("ERRO: o pacote nao satisfaz o contrato de serving")

    print("")
    print("ok. O pacote carrega a propria situacao em metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
