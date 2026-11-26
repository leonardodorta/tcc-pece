"""
financeiro — quando um valor pode entrar num somatório de gasto.

A pergunta "quanto o município gastou com este fornecedor" não se responde com
`SUM(amount_brl)`.

## Por que a regra existe

Sem ela, `SUM(amount_brl)` sem filtro somaria, no mesmo total:

    valor mensal de um contrato de 12 meses      (soma 1/12 do que deveria)
    valor unitário de uma ata de registro        (soma o preço de uma unidade)
    valor de uma multa aplicada                  (não é gasto com o objeto)
    valor de um contrato depois anulado          (nunca foi gasto)
    valor de um aditivo junto com o contrato original (pode contar o mesmo dinheiro duas vezes)

Nenhum desses casos produz erro visível: o número sai, é plausível, e está errado.

## As três condições

    natureza    o ato é gasto efetivo, e não penalidade, receita ou ato sem dinheiro
    grandeza    o valor lido é o total ou o estimado, e não mensal, unitário ou parcela
    vigência    o ato não foi anulado

As três são necessárias e nenhuma implica outra: um ato de gasto efetivo pode declarar valor
mensal (natureza ok, grandeza não), e um total pode ter sido anulado (natureza e grandeza ok,
vigência não).

## O aditivo

`dominio.ato` classifica aditivo e aditamento (e apostilamento) com a natureza `alteracao`,
que não está em `vocabulario.NATUREZAS_SOMAVEIS`. Por isso `avaliar` devolve
`(False, NATUREZA_NAO_SOMAVEL)` para eles, e `is_summable` fica falso: o valor de um aditivo
não entra no somatório, e o contrato original é o único que conta.

A consequência é que o acréscimo de valor trazido por um aditivo não é somado por esta
regra. Incluí-lo exigiria ligar o aditivo ao contrato original pelo número do contrato, que
a fonte não declara de forma confiável em todo aditivo, para saber se o aditivo reafirma o
valor do original ou acrescenta a ele.

## Por que devolve o motivo

`avaliar` devolve `(somavel, motivo)` e não só o booleano: sem o motivo, "por que este valor
não somou?" só se responde reexecutando o código mentalmente. Com ele, a distribuição dos
motivos é uma consulta, e uma mudança de distribuição é sinal de que a fonte mudou de forma.
"""
from __future__ import annotations

from . import vocabulario as V

# Os motivos de não somar

SOMAVEL = "somavel"
SEM_VALOR = "sem_valor"
NATUREZA_NAO_SOMAVEL = "natureza_nao_somavel"
GRANDEZA_NAO_SOMAVEL = "grandeza_nao_somavel"
ANULADO = "anulado"

MOTIVOS: tuple[str, ...] = (
    SOMAVEL,
    SEM_VALOR,
    NATUREZA_NAO_SOMAVEL,
    GRANDEZA_NAO_SOMAVEL,
    ANULADO,
)


def avaliar(
    *,
    valor: object | None,
    natureza: str | None,
    grandeza: str | None,
    anulado: bool | None,
) -> tuple[bool, str]:
    """
    O valor pode entrar num somatório de gasto? Devolve `(somavel, motivo)`.

    Só por palavra-chave: os quatro argumentos são fáceis de confundir na ordem, e trocar
    `natureza` por `grandeza` numa chamada posicional produziria um resultado plausível e
    errado.

    A ordem das verificações vai do mais barato e mais decisivo para o menos: sem valor não
    há o que somar.
    """
    if valor is None:
        return False, SEM_VALOR

    # `natureza` ausente (None) e natureza não somável recebem o mesmo resultado e o mesmo
    # motivo, `NATUREZA_NAO_SOMAVEL`: sem saber a natureza, somar é presumir.
    if natureza not in V.NATUREZAS_SOMAVEIS:
        return False, NATUREZA_NAO_SOMAVEL

    if grandeza not in V.GRANDEZAS_SOMAVEIS:
        return False, GRANDEZA_NAO_SOMAVEL

    # `anulado` nulo conta como não anulado: a coluna é preenchida por `ato.eh_anulacao`,
    # que devolve booleano sempre, então nulo aqui só acontece em linha antiga.
    if anulado:
        return False, ANULADO

    return True, SOMAVEL


def somavel(
    *,
    valor: object | None,
    natureza: str | None,
    grandeza: str | None,
    anulado: bool | None,
) -> bool:
    """Só o booleano, para quem grava a coluna `is_summable` e não precisa do motivo."""
    resultado, _ = avaliar(
        valor=valor, natureza=natureza, grandeza=grandeza, anulado=anulado
    )
    return resultado


# A mesma regra em SQL
#
# O ouro filtra em Spark e o consumidor filtra em Athena. Gerar a expressão a partir do mesmo
# vocabulário é o que impede a divergência entre as duas.


def _lista_sql(valores: frozenset[str]) -> str:
    """Os valores como lista SQL, em ordem determinística."""
    return ", ".join(f"'{v}'" for v in sorted(valores))


def clausula_sql(prefixo: str = "") -> str:
    """
    A regra como predicado SQL, para o `WHERE` de uma consulta Athena.

    `prefixo` qualifica as colunas quando a consulta junta tabelas (`c.` por exemplo).
    """
    p = f"{prefixo}." if prefixo and not prefixo.endswith(".") else prefixo
    return (
        f"{p}amount_brl IS NOT NULL"
        f" AND {p}financial_nature IN ({_lista_sql(V.NATUREZAS_SOMAVEIS)})"
        f" AND {p}amount_magnitude IN ({_lista_sql(V.GRANDEZAS_SOMAVEIS)})"
        f" AND COALESCE({p}is_annulment, FALSE) = FALSE"
    )
