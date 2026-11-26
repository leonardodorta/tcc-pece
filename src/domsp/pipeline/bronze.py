"""
bronze — a publicação bruta como linha tipada, sem interpretar nada.

Promete fidelidade: `body_html` vai inteiro, sem limpeza, sem normalizar espaço, sem
decodificar entidade. Se a fonte publicou com erro de digitação, o erro está aqui.

Não promete esquema semântico, tipo de negócio nem sentido. Isso é o estágio
seguinte. Uma camada que "arruma" o texto antes de guardar destrói a única cópia
fiel que o pipeline teria — e com ela a possibilidade de auditar qualquer extração
contra o que a fonte de fato disse.

## Codificação dos metadados

O bronze não altera o texto. A correção de dupla codificação (UTF-8 gravado e relido como
latin-1, em que "ã" chega como "Ã" seguido de um caractere de controle) é feita antes, em
`fonte.edicao.carregar_bytes`, e só nos sete campos curtos de metadado (`veiculo`,
`orgao`, `unidade`, `serie`, `processo`, `documento`, `link`). O corpo (`conteudo`) não
passa por essa correção.

A decisão é por edição, a partir da proporção de publicações com a assinatura de dupla
codificação. O bronze apenas copia o regime detectado (`Edicao.regime`) para a coluna
`encoding_regime`, para que a decisão fique registrada ao lado do dado.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Any

from .. import MODELO_AUSENTE, VERSAO_REGRAS
from ..fonte import edicao as _edicao


def hash_origem(texto: str | None) -> str:
    """
    SHA-256 do texto de origem, hexadecimal completo.

    SHA-256 e não `hash()` nativo: o `hash()` de `str` no CPython é aleatorizado por
    processo desde a 3.3 (`PYTHONHASHSEED`), então a mesma publicação produziria
    identificadores diferentes em execuções diferentes.
    """
    return hashlib.sha256((texto or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Publicacao:
    """Uma linha de `bronze_publication`. Os nomes são os do esquema."""

    publication_id: str
    edition_date: str | None
    edition_number: str | None
    section: str | None
    agency: str | None
    title: str | None
    body_html: str
    encoding_regime: str | None
    rule_version: str
    model_version: str
    run_id: str | None
    source_hash: str
    processed_at: dt.datetime
    dt: str

    def como_dicionario(self) -> dict[str, Any]:
        return dict(self.__dict__)


def da_edicao(
    ed: _edicao.Edicao,
    *,
    run_id: str | None = None,
    agora: dt.datetime | None = None,
) -> list[Publicacao]:
    """
    As linhas de bronze de uma edição já carregada.

    Recebe a `Edicao` e não o caminho: quem lê o arquivo é `fonte.edicao`, e separar as duas
    coisas é o que permite testar este estágio com uma edição sintética, sem tocar disco.

    Publicação sem corpo é descartada aqui e não gera linha. Não é triagem — a triagem decide
    o que é ato de contratação, e roda no estágio seguinte. Aqui só cai o que não tem texto
    nenhum, que é registro vazio da fonte.
    """
    instante = agora or dt.datetime.now(dt.UTC)
    linhas: list[Publicacao] = []

    for pub in ed.publicacoes:
        corpo = pub.get("conteudo")
        if not isinstance(corpo, str) or not corpo.strip():
            continue

        # O identificador da fonte quando existe; o hash do corpo quando não. O hash é
        # estável entre execuções, que é a única propriedade exigida — ver `hash_origem`.
        digest = hash_origem(corpo)
        identificador = _como_texto(pub.get("documento")) or digest[:32]

        linhas.append(Publicacao(
            publication_id=identificador,
            edition_date=_como_texto(pub.get("data")) or ed.data,
            edition_number=_como_texto(getattr(ed, "numero_edicao", None)) or None,
            # `veiculo` é o caderno. `serie` é o tipo do ato. A triagem usa os dois, e
            # confundi-los faria a triagem recusar o caderno certo pela série errada.
            section=_como_texto(pub.get("veiculo")),
            agency=_como_texto(pub.get("orgao")) or _como_texto(pub.get("unidade")),
            title=_como_texto(pub.get("serie")),
            body_html=corpo,
            encoding_regime=getattr(ed, "regime", None),
            rule_version=VERSAO_REGRAS,
            model_version=MODELO_AUSENTE,
            run_id=run_id,
            source_hash=digest,
            processed_at=instante,
            dt=ed.data,
        ))
    return linhas


def _como_texto(valor: object) -> str | None:
    """
    O valor como texto, ou None quando vazio.

    A fonte às vezes devolve número onde o esquema espera texto (o número da edição vem como
    int em algumas edições e como str noutras). Converter aqui, e não no consumidor, é o que
    impede o Parquet de receber tipos diferentes na mesma coluna em dias diferentes.
    """
    if valor is None:
        return None
    texto = str(valor).strip()
    return texto or None
