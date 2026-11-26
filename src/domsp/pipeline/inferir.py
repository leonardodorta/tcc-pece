"""
inferir — a fronteira entre o pipeline e o modelo, nos dois sentidos.

Este módulo não importa o modelo: conhece apenas o formato do JSONL que atravessa a
fronteira. Assim, a ausência ou falha do modelo não pode impedir as camadas
determinísticas.

## Os dois sentidos

    ida     resíduo → JSONL que o Batch Transform lê
    volta   JSONL que o Batch Transform escreveu → linhas de inference-raw e -accepted

## A porta é a mesma

Na volta, cada span passa por `dominio.parte.porta_de_nome` — a mesma função das três
camadas determinísticas. Sem ela, o modelo poderia gravar um nome que nenhuma regra
aceitaria, e as quatro camadas deixariam de ser comparáveis.

Guardar o que a porta recusou (`inference-raw` com `accepted=false` e a ressalva) é a
única forma de distinguir "o modelo não achou nada" de "o modelo achou e a porta
recusou" — e as duas situações pedem correções opostas.
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .. import VERSAO_REGRAS
from ..dominio import parte

CLASSE_CONTRATADA = "CONTRATADA"
CLASSE_CONTRATANTE = "CONTRATANTE"

# IDA — o resíduo como JSONL
#
# JSONL e não JSON: o Batch Transform divide a entrada por linha para paralelizar, e um JSON
# único de milhares de atos iria inteiro para uma instância só.


def para_jsonl(residuos: Iterable[Any]) -> Iterator[str]:
    """
    Cada resíduo como uma linha de JSON com `contract_id` e `text`.

    `ensure_ascii=False` de propósito: o texto é português, e escapar os acentos aumentaria o
    tamanho da entrada sem ganho.

    `contract_id` viaja junto porque a saída do Batch Transform não preserva a ordem entre
    instâncias. Sem o identificador na linha, um `zip` entre entrada e saída poderia atribuir
    nomes ao ato errado sem dar erro.
    """
    for r in residuos:
        identificador = getattr(r, "contract_id", None) or r["contract_id"]
        texto = getattr(r, "target_text", None)
        if texto is None:
            texto = r["target_text"]
        yield json.dumps(
            {"contract_id": identificador, "text": texto},
            ensure_ascii=False,
        )


# VOLTA — o JSONL do modelo como linhas da prata


@dataclass(frozen=True)
class Inferencia:
    """Uma linha de `silver_inference_raw` ou `_accepted`."""

    contract_id: str
    entity_class: str
    span_text: str
    span_start: int | None
    span_end: int | None
    score: float | None
    accepted: bool
    rejection: str | None
    truncated: bool
    rule_version: str
    model_version: str
    run_id: str | None
    source_hash: str
    processed_at: dt.datetime
    dt: str

    def como_dicionario(self) -> dict[str, Any]:
        return dict(self.__dict__)


def da_saida_do_modelo(
    linhas: Iterable[str | dict[str, Any]],
    *,
    modelo: str,
    particao_por_ato: dict[str, str],
    hash_por_ato: dict[str, str] | None = None,
    run_id: str | None = None,
    agora: dt.datetime | None = None,
) -> list[Inferencia]:
    """
    Converte a saída do modelo em linhas de inferência, aplicando a porta de validação.

    Espera, por linha, `{"contract_id": ..., "spans": [{"class","text","start","end","score"}],
    "truncated": bool}` — o formato que `predict_fn`, em `modelo/inferencia_sagemaker.py`,
    produz.

    `particao_por_ato` vem do resíduo, e não da saída do modelo: a partição é propriedade do
    ato e não da inferência, e o container não tem como conhecê-la.

    Um `contract_id` que não esteja em `particao_por_ato` é descartado silenciosamente (não
    entra no resultado): significa que o modelo devolveu um ato que não estava na entrada, e
    gravá-lo criaria linha órfã na prata.
    """
    instante = agora or dt.datetime.now(dt.UTC)
    hashes = hash_por_ato or {}
    resultado: list[Inferencia] = []

    for linha in linhas:
        registro = json.loads(linha) if isinstance(linha, str) else linha
        identificador = registro.get("contract_id")
        if not identificador or identificador not in particao_por_ato:
            continue

        truncado = bool(registro.get("truncated"))
        for span in registro.get("spans") or []:
            bruto = (span.get("text") or "").strip()
            if not bruto:
                continue

            classe = span.get("class") or CLASSE_CONTRATADA
            # CONTRATANTE nunca é resposta de parte contratada: a recusa é estrutural, igual à
            # de `servir.predict_fn`. A `porta_de_nome` não recebe a classe do span, então a
            # recusa é feita aqui, antes dela.
            if classe != CLASSE_CONTRATADA:
                resultado.append(Inferencia(
                    contract_id=identificador,
                    entity_class=classe,
                    span_text=bruto,
                    span_start=_inteiro(span.get("start")),
                    span_end=_inteiro(span.get("end")),
                    score=_real(span.get("score")),
                    accepted=False,
                    rejection="polo_publico_nao_e_contratada",
                    truncated=truncado,
                    rule_version=VERSAO_REGRAS,
                    model_version=modelo,
                    run_id=run_id,
                    source_hash=hashes.get(identificador, ""),
                    processed_at=instante,
                    dt=particao_por_ato[identificador],
                ))
                continue

            # A mesma porta das três camadas determinísticas.
            aceito = parte.porta_de_nome(bruto)
            ressalva = ", ".join(aceito.ressalvas) or None

            resultado.append(Inferencia(
                contract_id=identificador,
                entity_class=classe,
                # Grava o nome limpo quando a porta aceitou, e o bruto quando recusou: o bruto
                # na linha recusada mostra o que o modelo produziu de errado; o limpo na
                # aceita é o que vai para a prata final.
                span_text=aceito.nome if aceito.nome else bruto,
                span_start=_inteiro(span.get("start")),
                span_end=_inteiro(span.get("end")),
                score=_real(span.get("score")),
                accepted=aceito.nome is not None,
                rejection=ressalva,
                truncated=truncado,
                rule_version=VERSAO_REGRAS,
                model_version=modelo,
                run_id=run_id,
                source_hash=hashes.get(identificador, ""),
                processed_at=instante,
                dt=particao_por_ato[identificador],
            ))
    return resultado


def aceitas(inferencias: Iterable[Inferencia]) -> list[Inferencia]:
    """As que passaram pela porta. É o que `silver/inference-accepted/` recebe."""
    return [i for i in inferencias if i.accepted]


def melhor_por_ato(
    inferencias: Iterable[Inferencia],
    *,
    classe: str = CLASSE_CONTRATADA,
    escore_minimo: float = 0.0,
) -> dict[str, Inferencia]:
    """
    Uma inferência por ato: a de maior escore da classe pedida, entre as aceitas.

    Um ato pode receber vários spans — o modelo às vezes acha duas organizações na mesma prosa.
    Escolher por escore, e não a primeira, é o que torna a escolha independente da ordem em que
    o container devolveu os spans.

    `escore_minimo` é a alavanca de precisão sobre cobertura: subir o limiar recusa mais e erra
    menos. O valor de produção sai da avaliação no conjunto de teste, não daqui — este módulo
    não decide o limiar, só o aplica.
    """
    melhor: dict[str, Inferencia] = {}
    for i in inferencias:
        if not i.accepted or i.entity_class != classe:
            continue
        if (i.score or 0.0) < escore_minimo:
            continue
        atual = melhor.get(i.contract_id)
        if atual is None or (i.score or 0.0) > (atual.score or 0.0):
            melhor[i.contract_id] = i
    return melhor


# A CONSOLIDAÇÃO


def consolidar(
    atos: Iterable[Any],
    melhores: dict[str, Inferencia],
    *,
    camada_modelo: str = "model",
) -> list[dict[str, Any]]:
    """
    A prata final: determinístico mais inferência aceita, sem sobrescrever.

    A regra em uma linha: o modelo só preenche onde havia nulo.

    Devolve dicionários e não `Ato` porque a linha final carrega `model_version` diferente da
    do ato, e um `Ato` congelado teria de ser reconstruído inteiro.
    """
    finais: list[dict[str, Any]] = []
    for a in atos:
        linha = a.como_dicionario() if hasattr(a, "como_dicionario") else dict(a)
        inferida = melhores.get(linha["contract_id"])

        # A ordem desta condição é a garantia: só entra no `if` quando o determinístico deixou
        # nulo. Inverter para `if inferida is not None` sobrescreveria dado declarado pela fonte.
        if linha.get("contractor_name") is None and inferida is not None:
            linha["contractor_name"] = inferida.span_text
            linha["model_version"] = inferida.model_version
            proveniencia = dict(linha.get("field_provenance") or {})
            proveniencia["contractor_name"] = camada_modelo
            linha["field_provenance"] = proveniencia
        finais.append(linha)
    return finais


def _inteiro(valor: object) -> int | None:
    try:
        return int(valor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _real(valor: object) -> float | None:
    try:
        return float(valor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
