"""
gold — agregados com significado declarado, e as métricas do próprio pipeline.

As perguntas que a fonte publica e não permite fazer:

    gold_supplier_month   quanto cada fornecedor recebeu por mês, em quantos órgãos
    gold_agency_month     quanto cada órgão contratou por mês, de quantos fornecedores
    gold_quality          a cobertura de cada camada e de cada campo, com intervalo

## A regra financeira é aplicada aqui, e vem de um lugar só

`total_amount_brl` soma apenas as linhas em que `is_summable` é verdadeiro, e `is_summable` foi
decidido na prata por `contrato.financeiro.avaliar`. O ouro não reimplementa a regra: se o
fizesse, uma mudança de vocabulário atualizaria um lado e deixaria o outro, e o número do painel
divergiria do da prata sem sintoma.

`summable_count` ao lado de `contract_count` é o que torna a diferença visível: quando os dois
divergem muito, o total não representa os atos.

## Por que gold_quality é uma TABELA e não um log

A cobertura por camada precisa ser consultável na mesma linguagem que o resto:
`SELECT proportion FROM gold_quality WHERE metric_name = ...`.

Um número em log de execução não é verificável depois e não permite acompanhar deriva: se a
cobertura de uma camada mudar entre execuções, esta tabela mostra quando a mudança começou.
"""
from __future__ import annotations

import datetime as dt
import decimal
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .. import MODELO_AUSENTE, VERSAO_REGRAS
from ..contrato import vocabulario as V
from ..medicao import incerteza
from .silver import CAMADA_MODELO, CAMADAS_DETERMINISTICAS

ZERO = decimal.Decimal("0.00")


# OS AGREGADOS


@dataclass(frozen=True)
class Medida:
    """Uma linha de `gold_quality`."""

    metric_name: str
    stratum: str
    numerator: int
    denominator: int
    proportion: float
    ci_lower: float
    ci_upper: float
    ci_method: str
    measured_at: dt.datetime
    rule_version: str
    model_version: str
    run_id: str | None
    source_hash: str
    processed_at: dt.datetime

    def como_dicionario(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _mes(linha: dict[str, Any]) -> str:
    """
    O mês da EDIÇÃO, `aaaa-mm`, derivado de `dt`.

    De `dt` e não de `signature_date`: a data de assinatura pode faltar, e um
    agregado mensal com essas linhas fora do mês estaria errado de forma silenciosa. `dt` é a
    data da edição, sempre presente por construção.
    """
    return str(linha.get("dt") or "")[:7]


def por_fornecedor(linhas: Iterable[dict[str, Any]], **procedencia: Any) -> list[dict[str, Any]]:
    """
    Quanto cada fornecedor recebeu por mês.

    Agrupa por documento fiscal e não por nome: a fonte escreve o mesmo fornecedor de várias
    formas ("LTDA", "Ltda.", "LTDA ME"), e agrupar por nome multiplicaria a mesma empresa em
    linhas separadas.

    O nome exibido é a grafia mais frequente entre as do mesmo documento, e não a primeira,
    porque "a primeira" depende da ordem de leitura dos arquivos.
    """
    grupos: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for linha in linhas:
        documento = linha.get("contractor_tax_id")
        if not documento:
            # Sem documento não há chave de agrupamento confiável. A linha continua na prata e
            # não entra neste agregado — e é por isso que somar este agregado dá menos que
            # somar a prata. A medida `summable_reachable_by_supplier` de `gold_quality`
            # declara exatamente quanto.
            continue
        grupos[(_mes(linha), str(documento))].append(linha)

    resultado: list[dict[str, Any]] = []
    for (mes, documento), atos in sorted(grupos.items()):
        somaveis = [a for a in atos if a.get("is_summable")]
        nomes = Counter(a["contractor_name"] for a in atos if a.get("contractor_name"))
        resultado.append({
            "year_month": mes,
            "contractor_tax_id": documento,
            "contractor_name": nomes.most_common(1)[0][0] if nomes else None,
            "contract_count": len(atos),
            "summable_count": len(somaveis),
            "total_amount_brl": sum(
                (a["amount_brl"] for a in somaveis if a.get("amount_brl") is not None), ZERO
            ),
            "agency_count": len({a["agency"] for a in atos if a.get("agency")}),
            **procedencia,
        })
    return resultado


def por_orgao(linhas: Iterable[dict[str, Any]], **procedencia: Any) -> list[dict[str, Any]]:
    """Quanto cada órgão contratou por mês, de quantos fornecedores distintos."""
    grupos: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for linha in linhas:
        orgao = linha.get("agency")
        if not orgao:
            continue
        grupos[(_mes(linha), str(orgao))].append(linha)

    resultado: list[dict[str, Any]] = []
    for (mes, orgao), atos in sorted(grupos.items()):
        somaveis = [a for a in atos if a.get("is_summable")]
        resultado.append({
            "year_month": mes,
            "agency": orgao,
            "contract_count": len(atos),
            "summable_count": len(somaveis),
            "total_amount_brl": sum(
                (a["amount_brl"] for a in somaveis if a.get("amount_brl") is not None), ZERO
            ),
            "supplier_count": len(
                {a["contractor_tax_id"] for a in atos if a.get("contractor_tax_id")}
            ),
            **procedencia,
        })
    return resultado


# AS MÉTRICAS

# Os campos cuja taxa de preenchimento é medida. Nomes de coluna, para que acrescentar uma
# coluna ao esquema e esquecer de medi-la seja visível numa lista.
CAMPOS_MEDIDOS: tuple[str, ...] = (
    "contractor_name",
    "contractor_tax_id",
    "object",
    "amount_brl",
    "signature_date",
    "term_months",
    "contract_number",
    "process_number",
    "legal_basis",
    "act_kind",
    "financial_nature",
)


def qualidade(
    linhas: list[dict[str, Any]],
    *,
    total_publicacoes: int | None = None,
    run_id: str | None = None,
    modelo: str = MODELO_AUSENTE,
    agora: dt.datetime | None = None,
) -> list[Medida]:
    """
    As métricas do pipeline, com intervalo de Wilson a 95%.

    Wilson e não normal: com proporção perto de 0 ou de 1 o intervalo normal produz limite fora
    de [0,1]. Com 0 acertos em `n` casos, o intervalo normal degenera em [0, 0], afirmando
    certeza absoluta com amostra finita.

    `total_publicacoes` habilita a taxa de aceitação da triagem. É opcional porque quem chama
    pode ter apenas a prata em mão, e uma métrica ausente é melhor que uma métrica com
    denominador errado.
    """
    instante = agora or dt.datetime.now(dt.UTC)
    total = len(linhas)
    medidas: list[tuple[str, str, int, int]] = []

    if total_publicacoes:
        medidas.append(("triage_accepted", V.ESTRATO_AGREGADO, total, total_publicacoes))

    if total == 0:
        return []

    # ── cobertura por camada de contratada ────────────────────────────────────
    camadas = Counter(
        (linha.get("field_provenance") or {}).get("contractor_name")
        for linha in linhas
    )
    for camada in (*CAMADAS_DETERMINISTICAS, CAMADA_MODELO):
        medidas.append((f"layer_{camada}", V.ESTRATO_AGREGADO, camadas.get(camada, 0), total))
    deterministico = sum(camadas.get(c, 0) for c in CAMADAS_DETERMINISTICAS)
    medidas.append(("layer_deterministic_total", V.ESTRATO_AGREGADO, deterministico, total))

    # ── preenchimento por campo ───────────────────────────────────────────────
    for campo in CAMPOS_MEDIDOS:
        preenchidos = sum(1 for linha in linhas if linha.get(campo) is not None)
        medidas.append((f"filled_{campo}", V.ESTRATO_AGREGADO, preenchidos, total))

    # ── a regra financeira ────────────────────────────────────────────────────
    com_valor = [linha for linha in linhas if linha.get("amount_brl") is not None]
    somaveis = sum(1 for linha in linhas if linha.get("is_summable"))
    medidas.append(("summable", V.ESTRATO_AGREGADO, somaveis, total))
    if com_valor:
        # Entre os que têm valor, quantos podem somar — diz se somar `amount_brl` sem filtro
        # superestimaria, e por quanto.
        somaveis_com_valor = sum(1 for linha in com_valor if linha.get("is_summable"))
        medidas.append(("summable_given_amount", "com_valor", somaveis_com_valor, len(com_valor)))

    # O que o agregado de fornecedor não alcança.
    #
    # `por_fornecedor` agrupa por documento fiscal e descarta a linha sem documento, então
    # somar `gold_supplier_month` dá menos que somar a prata. A métrica
    # `filled_contractor_tax_id` mede o preenchimento sobre todos os atos; esta mede a
    # cobertura sobre os atos que de fato somam.
    #
    # Não se corrige agrupando o nulo sob um sentinela: isso juntaria fornecedores distintos
    # numa linha só e estragaria a leitura de "maiores fornecedores". A correção certa é
    # declarar a cobertura.
    if somaveis:
        somaveis_com_documento = sum(
            1
            for linha in linhas
            if linha.get("is_summable") and linha.get("contractor_tax_id")
        )
        medidas.append(
            ("summable_reachable_by_supplier", "somavel", somaveis_com_documento, somaveis)
        )

    # as razões de não somar, cada uma como sua própria medida
    razoes = Counter(
        linha.get("summability_reason") for linha in linhas if not linha.get("is_summable")
    )
    for razao, quantos in sorted(razoes.items(), key=lambda kv: str(kv[0])):
        if razao:
            medidas.append((f"not_summable_{razao}", V.ESTRATO_AGREGADO, quantos, total))

    # datas recusadas por implausibilidade: a contagem permite acompanhar, entre execuções, se
    # a recusa vem do parser de data ou da fonte.
    recusadas = sum(
        1 for linha in linhas
        if any(
            str(v).startswith("recusada")
            for v in (linha.get("field_provenance") or {}).values()
        )
    )
    medidas.append(("rejected_implausible_date", V.ESTRATO_AGREGADO, recusadas, total))

    return [
        _medida(nome, estrato, num, den, instante, run_id, modelo)
        for nome, estrato, num, den in medidas
    ]


def _medida(
    nome: str,
    estrato: str,
    numerador: int,
    denominador: int,
    instante: dt.datetime,
    run_id: str | None,
    modelo: str,
) -> Medida:
    intervalo = incerteza.wilson(numerador, denominador)
    return Medida(
        metric_name=nome,
        stratum=estrato,
        numerator=numerador,
        denominator=denominador,
        proportion=intervalo.valor,
        ci_lower=intervalo.limite_inferior,
        ci_upper=intervalo.limite_superior,
        ci_method=V.INTERVALO_WILSON,
        measured_at=instante,
        rule_version=VERSAO_REGRAS,
        model_version=modelo,
        run_id=run_id,
        source_hash="",
        processed_at=instante,
    )
