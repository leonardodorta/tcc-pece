"""ato.py — classifica o tipo canônico do ato e a natureza do dinheiro envolvido, e detecta anulação e republicação."""
from __future__ import annotations

from .. import texto as T
from ..contrato import vocabulario as V
from . import triagem

# TIPO CANÔNICO
# O vocabulário fechado da coluna `act_kind`. Declarado aqui para que este módulo
# possa validar que todo tipo que devolve está no contrato — uma série nova na
# fonte que mapeie para um tipo fora desta tupla é defeito, e a guarda no fim do
# módulo o transforma em erro na importação.
TIPOS_CANONICOS: frozenset[str] = frozenset({
    "contrato", "aditivo", "ata_registro_preco", "apostilamento", "rescisao",
    "penalidade", "convenio", "termo_fomento", "termo_colaboracao",
    "acordo_cooperacao", "doacao", "empenho", "autorizacao_uso",
})

# Refino do tipo de uma série genérica pela palavra-chave que a confirmou. A ordem
# é do mais específico para o mais geral, porque a primeira que casar decide: uma
# prosa que diz "termo de colaboração" e "contrato" é colaboração, não contrato.
_KEYWORD_PARA_TIPO: tuple[tuple[str, str], ...] = (
    ("aditiv", "aditivo"),
    ("aditamento", "aditivo"),
    ("apostilamento", "apostilamento"),
    ("ata de registro", "ata_registro_preco"),
    ("colaboracao", "termo_colaboracao"),
    ("fomento", "termo_fomento"),
    ("cooperacao", "acordo_cooperacao"),
    ("convenio", "convenio"),
    ("doacao", "doacao"),
    ("gestao", "contrato"),        # "contrato de gestão" é contrato
    ("contrat", "contrato"),
)


def tipo_por_keyword(keyword: str | None) -> str:
    """
    Refina o tipo de uma série genérica pela expressão que a confirmou.

    `keyword` vem de `triagem.keyword_confirmadora`. A comparação é sem acento
    porque a keyword pode vir da fonte com grafias variadas. Default `contrato`:
    uma série genérica que passou na triagem por uma keyword sem refino específico
    é, conservadoramente, um contrato — a natureza gasto_efetivo é a hipótese mais
    exposta a controle, e errar para o lado de "é gasto" é preferível a esconder
    despesa.
    """
    k = T.remover_acentos(keyword or "").lower()
    for padrao, tipo in _KEYWORD_PARA_TIPO:
        if padrao in k:
            return tipo
    return "contrato"


def tipo_canonico(
    serie: str | None,
    conteudo_texto: str | None = None,
) -> str | None:
    """
    O tipo canônico do ato, ou `None` se a série não é de contratação.

    Não repete a triagem: pressupõe que a publicação já passou por `triagem.triar`.
    Para série genérica ("Extrato"/"Termo"), refina pela keyword confirmadora do
    texto; para as demais, lê direto de `triagem.SERIES_CONTRATO`.
    """
    s = (serie or "").strip()
    tipo = triagem.SERIES_CONTRATO.get(s)
    if tipo is None:
        return None
    if tipo == triagem.TIPO_GENERICO:
        return tipo_por_keyword(triagem.keyword_confirmadora(conteudo_texto))
    return tipo


# NATUREZA FINANCEIRA
# Os cinco valores da coluna `financial_nature`. gasto_efetivo (contrato, empenho) é
# dinheiro que sai para adquirir bem ou serviço; reserva_preco (ata_registro_preco)
# é preço registrado, não executado; repasse (convenio, doacao, termo_fomento,
# termo_colaboracao, acordo_cooperacao) é transferência ao terceiro setor ou a
# outro ente; alteracao (aditivo, apostilamento) muda um contrato já existente, e
# não deve ser somada ao original para não contar o mesmo objeto duas vezes;
# nao_gasto (rescisao, penalidade, autorizacao_uso) não é despesa. Fechado e
# conferido pela guarda no fim do módulo.

NATUREZAS_CANONICAS: frozenset[str] = frozenset({
    "gasto_efetivo", "reserva_preco", "repasse", "alteracao", "nao_gasto",
})

_NATUREZA_POR_GRUPO: dict[str, tuple[str, ...]] = {
    "gasto_efetivo": ("contrato", "empenho"),
    "reserva_preco": ("ata_registro_preco",),
    "repasse": ("convenio", "doacao", "termo_fomento", "termo_colaboracao",
                "acordo_cooperacao"),
    "alteracao": ("aditivo", "apostilamento"),
    "nao_gasto": ("rescisao", "penalidade", "autorizacao_uso"),
}

TIPO_PARA_NATUREZA: dict[str, str] = {
    tipo: natureza
    for natureza, tipos in _NATUREZA_POR_GRUPO.items()
    for tipo in tipos
}


def natureza(tipo: str | None) -> str | None:
    """
    A natureza financeira de um tipo canônico, ou `None` se o tipo é desconhecido.

    Devolver `None` (e não um default) para tipo desconhecido é deliberado: um tipo
    fora do vocabulário é defeito a expor, não a mascarar com uma natureza
    plausível. A guarda no fim do módulo garante que todo tipo canônico tem
    natureza, então na prática `None` só ocorre para entrada inválida.
    """
    return TIPO_PARA_NATUREZA.get((tipo or "").strip() or None)


# ATO NÃO ONEROSO
# Comodato, cessão gratuita, permissão de uso e contrato declarado "não oneroso"/"sem
# ônus" NÃO movimentam dinheiro — ainda que a fonte declare um "valor" (o valor de
# referência do bem cedido, por exemplo). A natureza correta é `nao_gasto`, que a
# regra financeira não soma.
MARCADORES_NAO_ONEROSO: tuple[str, ...] = (
    "nao oneroso", "nao-oneroso", "sem onus", "a titulo gratuito", "a titulo nao oneroso",
    "cessao gratuita", "cessao de uso gratuita", "comodato", "doacao sem encargo",
    "permissao de uso a titulo gratuito",
)
MARCADORES_NAO_ONEROSO_NORM: tuple[str, ...] = tuple(
    T.remover_acentos(m).lower() for m in MARCADORES_NAO_ONEROSO
)
# Janela: o preâmbulo e o corpo próximo, onde a fonte declara a natureza do ajuste.
JANELA_NAO_ONEROSO = 800


def eh_nao_oneroso(conteudo_texto: str | None) -> bool:
    """
    O ato é não oneroso (comodato, cessão gratuita, sem ônus) e não deve ser somado.

    Detectado no texto, como `eh_anulacao`. Conservador: só marca quando a fonte DECLARA a
    gratuidade com um dos marcadores inequívocos — não infere de ausência de valor.
    """
    if not conteudo_texto:
        return False
    janela = T.remover_acentos(conteudo_texto[:JANELA_NAO_ONEROSO]).lower()
    return any(m in janela for m in MARCADORES_NAO_ONEROSO_NORM)


# ANULAÇÃO
# Marcadores de ato que retira ou corrige publicação anterior. Vários destes
# marcadores aparecem em séries que a triagem já descarta (Despacho, Portaria); o
# valor de `eh_anulacao` está em marcá-los nas publicações de série de CONTRATO
# que são, na verdade, anulações — é ali que o sinal do valor se inverte.
MARCADORES_ANULACAO: tuple[str, ...] = (
    "tornar sem efeito", "torno sem efeito", "fica sem efeito",
    "republicação por incorreção", "republicado por ter saído com incorreção",
    "onde se lê", "leia-se", "anulo", "declaro nulo", "revogo",
)

# Normalizado no mesmo espaço (sem acento, minúsculo) em que o texto é comparado,
# para que marcadores com acento casem.
MARCADORES_ANULACAO_NORM: tuple[str, ...] = tuple(
    T.remover_acentos(m).lower() for m in MARCADORES_ANULACAO
)

# Só o preâmbulo é inspecionado: os marcadores aparecem no início do ato, e
# procurá-los no corpo inteiro geraria falso positivo em cláusula de rescisão que
# cita "tornar sem efeito" como hipótese contratual.
JANELA_ANULACAO = 400


def eh_anulacao(conteudo_texto: str | None) -> bool:
    """
    A publicação retira ou corrige um ato anterior (valor de sinal oposto).

    Uma anulação de empenho não deve ser registrada como contrato novo nem ter o
    valor somado como gasto, pois representa dinheiro devolvido. Comparação no
    preâmbulo normalizado; os dois lados no mesmo espaço.
    """
    if not conteudo_texto:
        return False
    cabeca = T.remover_acentos(conteudo_texto[:JANELA_ANULACAO]).lower()
    return any(m in cabeca for m in MARCADORES_ANULACAO_NORM)


# REPUBLICAÇÃO
# Sinal, no próprio texto, de que o ato é republicação de um ato anterior. Detecta
# só o que é local à publicação — o ato se declarar republicação; a deduplicação
# por pares (comparar publicações entre si) é agregação e não é feita aqui.
#
# Há sobreposição deliberada com MARCADORES_ANULACAO ("republicado",
# "republicação"), porque republicação-por-incorreção é, ao mesmo tempo, correção
# de ato anterior. Um ato pode ser anulação E republicação; os dois campos são
# independentes de propósito.
MARCADORES_REPUBLICACAO: tuple[str, ...] = (
    "republicad",                       # republicado / republicada / republicados
    "republicacao",                     # republicação (sem acento após normalizar)
    "republicado por ter saido",
    "republicacao por incorrecao",
    "publicado novamente",
)

MARCADORES_REPUBLICACAO_NORM: tuple[str, ...] = tuple(
    T.remover_acentos(m).lower() for m in MARCADORES_REPUBLICACAO
)

JANELA_REPUBLICACAO = 400


def eh_republicacao(conteudo_texto: str | None) -> bool:
    """
    A publicação se declara republicação de um ato anterior.

    Marca o fenômeno para que a agregação possa descontá-lo; não deduplica.
    """
    if not conteudo_texto:
        return False
    cabeca = T.remover_acentos(conteudo_texto[:JANELA_REPUBLICACAO]).lower()
    return any(m in cabeca for m in MARCADORES_REPUBLICACAO_NORM)


def classificar(
    serie: str | None,
    conteudo_texto: str | None = None,
) -> dict[str, object]:
    """
    A classificação completa de um ato já triado: tipo, natureza e os dois sinais.

    Conveniência que reúne as decisões numa chamada, na forma que a tabela
    `silver_contract` consome. Pressupõe que a publicação passou por `triagem.triar`
    (motivo `None`). Devolve as chaves `tipo`, `natureza`, `eh_anulacao`,
    `eh_republicacao` e `eh_nao_oneroso`.
    """
    tipo = tipo_canonico(serie, conteudo_texto)
    nat = natureza(tipo)
    nao_oneroso = eh_nao_oneroso(conteudo_texto)
    # Um ato não oneroso declarado não é gasto, mesmo que a série o classifique como
    # contrato e a fonte traga um valor de referência. A natureza vira `nao_gasto`,
    # que a regra financeira não soma. Só rebaixa o que seria `gasto_efetivo`: um
    # repasse ou uma alteração mantêm a sua natureza.
    if nao_oneroso and nat == "gasto_efetivo":
        nat = "nao_gasto"
    return {
        "tipo": tipo,
        "natureza": nat,
        "eh_anulacao": eh_anulacao(conteudo_texto),
        "eh_republicacao": eh_republicacao(conteudo_texto),
        "eh_nao_oneroso": nao_oneroso,
    }


# A GUARDA — executada na importação
# Um tipo que a taxonomia de séries produz e o contrato não conhece, ou um tipo
# sem natureza, é defeito silencioso que deve falhar na importação, não aparecer
# como coluna vazia no Athena.

# Todo tipo que SERIES_CONTRATO produz (exceto a marca genérica) deve ser canônico.
for _serie, _tipo in triagem.SERIES_CONTRATO.items():
    if _tipo == triagem.TIPO_GENERICO:
        continue
    if _tipo not in TIPOS_CANONICOS:
        raise AssertionError(
            f"série {_serie!r} mapeia para tipo {_tipo!r}, que não está no "
            "vocabulário da coluna `act_kind` de silver_contract"
        )

# Todo destino de _KEYWORD_PARA_TIPO também.
for _padrao, _tipo in _KEYWORD_PARA_TIPO:
    if _tipo not in TIPOS_CANONICOS:
        raise AssertionError(
            f"keyword {_padrao!r} mapeia para tipo {_tipo!r} fora do vocabulário"
        )

# Todo tipo canônico tem natureza, e toda natureza é canônica.
for _tipo in TIPOS_CANONICOS:
    _nat = TIPO_PARA_NATUREZA.get(_tipo)
    if _nat is None:
        raise AssertionError(f"tipo canônico {_tipo!r} sem natureza declarada")
    if _nat not in NATUREZAS_CANONICAS:
        raise AssertionError(
            f"tipo {_tipo!r} tem natureza {_nat!r} fora do vocabulário da coluna"
        )

del _serie, _tipo, _padrao, _nat


# A GUARDA CONTRA O VOCABULÁRIO PUBLICADO
# `contrato/vocabulario.py` é a declaração única dos conjuntos fechados. Estas
# guardas fazem com que um tipo ou natureza nova neste módulo, sem entrada no
# vocabulário, falhe na importação — nos dois sentidos: produzir valor não
# declarado, e declarar valor que nenhuma regra produz.

_tipos = set(TIPO_PARA_NATUREZA)
_naturezas = set(TIPO_PARA_NATUREZA.values())

if _tipos - set(V.TIPOS):
    raise AssertionError(
        f"tipo produzido e não declarado em contrato.vocabulario.TIPOS: "
        f"{sorted(_tipos - set(V.TIPOS))}"
    )
if set(V.TIPOS) - _tipos:
    raise AssertionError(
        f"tipo declarado no vocabulário e que nenhuma regra produz: "
        f"{sorted(set(V.TIPOS) - _tipos)}"
    )
if _naturezas != set(V.NATUREZAS):
    raise AssertionError(
        f"naturezas divergem do vocabulário: "
        f"produzidas-declaradas={sorted(_naturezas - set(V.NATUREZAS))} "
        f"declaradas-produzidas={sorted(set(V.NATUREZAS) - _naturezas)}"
    )
del _tipos, _naturezas
