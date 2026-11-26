"""
objeto.py — extrai o objeto contratado (o que o ajuste compra), por campo de
formulário ou por rótulo na prosa ("OBJETO:", "DO OBJETO:" etc.), sem resumir,
classificar ou normalizar o texto para uma taxonomia.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import texto as T

# Rótulos em ordem de especificidade: `OBJETO DO CONTRATO` antes de `OBJETO` porque a
# alternância de regex casa a primeira alternativa que serve, e `OBJETO` casaria o
# prefixo de `OBJETO DO CONTRATO`, deixando ` DO CONTRATO:` dentro do valor extraído.
ROTULOS: tuple[str, ...] = (
    r"OBJETO\s+DO\s+CONTRATO",
    r"OBJETO\s+DO\s+ADITAMENTO",
    r"OBJETO\s+DO\s+AJUSTE",
    r"OBJETO\s+DO\s+TERMO(?:\s+ADITIVO)?",
    r"OBJETO\s+DA\s+ATA",
    r"OBJETO\s+DA\s+PARCERIA",
    r"OBJETO\s+DA\s+CONTRATA[ÇC][ÃA]O",
    r"DO\s+OBJETO",
    r"OBJETO",
)

_ROTULO = re.compile(
    r"(?i)(?:^|[\s;.\n])(" + "|".join(ROTULOS) + r")\s*:?\s*",
)

# Onde o objeto termina: a prosa não tem fim sintático próprio, então quem o encerra é
# o rótulo do campo seguinte (ex.: "VALOR:"). A lista vem dos rótulos usados pela
# fonte; não é exaustiva — o limite MAXIMO abaixo cobre o rótulo que falte aqui.
TERMINADORES: tuple[str, ...] = (
    r"VALOR(?:\s+(?:TOTAL|GLOBAL|MENSAL|ANUAL|ESTIMADO|UNIT[ÁA]RIO))?",
    r"VIG[ÊE]NCIA",
    r"PRAZO(?:\s+DE\s+(?:VIG[ÊE]NCIA|EXECU[ÇC][ÃA]O))?",
    r"DATA\s+(?:DA\s+)?ASSINATURA",
    r"CONTRATAD[AO]",
    r"CONTRATANTE",
    r"DETENTORA",
    r"COMPROMISS[ÁA]RIA",
    r"CNPJ",
    r"CPF",
    r"PROCESSO(?:\s+(?:N[º°]|ADMINISTRATIVO|SEI))?",
    r"FUNDAMENT(?:O|A[ÇC][ÃA]O)\s+LEGAL",
    r"MODALIDADE",
    r"DOTA[ÇC][ÃA]O(?:\s+OR[ÇC]AMENT[ÁA]RIA)?",
    r"NOTA\s+DE\s+(?:EMPENHO|RESERVA)",
    r"AUTORIZA[ÇC][ÃA]O",
    r"JUSTIFICATIVA",
    r"SIGNAT[ÁA]RIOS?",
    r"LOCAL\s+DE\s+EXECU[ÇC][ÃA]O",
    r"UNIDADE\s+REQUISITANTE",
    r"T[ÍI]TULO",
)

_TERMINADOR = re.compile(
    r"(?i)(?:^|[\s;.\n])(?:" + "|".join(TERMINADORES) + r")\s*:",
)

# Abaixo deste tamanho o recorte é tratado como resíduo de rótulo ou pontuação, e não
# como objeto.
MINIMO = 12

# Acima deste tamanho o recorte provavelmente passou do objeto e engoliu cláusula.
# Trunca em vez de recusar porque o começo do recorte continua sendo o objeto.
MAXIMO = 2_000

RECUSA_VAZIO = "objeto_vazio"
RECUSA_CURTO = "objeto_curto_demais"
RECUSA_SO_RUIDO = "objeto_sem_letra"
TRUNCADO = "objeto_truncado"


@dataclass(frozen=True)
class ResultadoObjeto:
    """Resultado da validação: `texto` é None quando o candidato foi recusado, e
    `ressalvas` lista o que foi recusado ou corrigido."""

    texto: str | None
    ressalvas: tuple[str, ...]

    def __iter__(self):
        yield self.texto
        yield tuple(self.ressalvas)


def porta_de_objeto(candidato: str | None) -> ResultadoObjeto:
    """A validação única por onde passa todo objeto candidato, vindo do formulário
    ou da prosa."""
    if candidato is None:
        return ResultadoObjeto(None, (RECUSA_VAZIO,))

    limpo = T.colapsar_espacos(candidato).strip(" \t\r\n;:.,-–—")
    if not limpo:
        return ResultadoObjeto(None, (RECUSA_VAZIO,))

    # Precisa conter letra: um recorte só de número e pontuação é resíduo de outro
    # campo que escapou do terminador, não descrição de objeto.
    if not re.search(r"[A-Za-zÀ-ÿ]", limpo):
        return ResultadoObjeto(None, (RECUSA_SO_RUIDO,))

    if len(limpo) < MINIMO:
        return ResultadoObjeto(None, (RECUSA_CURTO,))

    ressalvas: tuple[str, ...] = ()
    if len(limpo) > MAXIMO:
        # Corta na fronteira de palavra para não partir token no meio.
        corte = limpo.rfind(" ", 0, MAXIMO)
        limpo = limpo[: corte if corte > MINIMO else MAXIMO].rstrip()
        ressalvas = (TRUNCADO,)

    return ResultadoObjeto(limpo, ressalvas)


def do_formulario(valor_do_campo: str | None) -> ResultadoObjeto:
    """
    O objeto declarado no campo `Objeto do Contrato` do Painel de Negócios.

    Recebe o valor do campo já lido, e não o `Formulario`, para não depender de
    `formulario`. Quem chama passa `form.campos.get(formulario.CAMPO_OBJETO)`.
    """
    return porta_de_objeto(valor_do_campo)


def da_prosa(alvo: str | None) -> ResultadoObjeto:
    """
    O objeto nomeado por rótulo na prosa, do rótulo até o campo seguinte.

    O recorte vai do fim do rótulo até o primeiro terminador, ou até `MAXIMO` se
    nenhum terminador aparecer. Procurar o terminador **depois** do rótulo, e não
    no texto todo, evita que um `VALOR:` anterior ao objeto encurte o recorte a nada.
    """
    if not alvo:
        return ResultadoObjeto(None, (RECUSA_VAZIO,))

    achado = _ROTULO.search(alvo)
    if achado is None:
        return ResultadoObjeto(None, (RECUSA_VAZIO,))

    resto = alvo[achado.end() :]
    fim = _TERMINADOR.search(resto)
    bruto = resto[: fim.start()] if fim else resto[: MAXIMO + 1]
    return porta_de_objeto(bruto)


CAMADA_FORMULARIO = "form"
CAMADA_PROSA = "prose_label"


def extrair(
    valor_do_campo: str | None, alvo: str | None
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """
    O objeto pelas duas camadas, na ordem de precisão. Devolve (texto, camada, ressalvas).

    Campo antes de prosa: o campo é declaração da fonte e a prosa é leitura feita pelo pipeline.
    """
    do_campo = do_formulario(valor_do_campo)
    if do_campo.texto is not None:
        return do_campo.texto, CAMADA_FORMULARIO, do_campo.ressalvas

    na_prosa = da_prosa(alvo)
    if na_prosa.texto is not None:
        return na_prosa.texto, CAMADA_PROSA, na_prosa.ressalvas

    # Nenhuma camada alcançou. Devolve as ressalvas da prosa (tentativa mais recente).
    return None, None, na_prosa.ressalvas
