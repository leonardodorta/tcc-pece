"""
texto — HTML para texto, e as normalizações que todo o resto assume.

Folha da árvore de dependências: não importa nenhum outro subpacote de `domsp`. É o
único lugar onde a decisão "o que conta como o mesmo caractere" está escrita.

A implementação fica em `_texto.py` e é reexportada aqui.
"""
from __future__ import annotations

from ._texto import (
    colapsar_espacos,
    html_para_texto,
    normalizar,
    normalizar_chave,
    remover_acentos,
    traduzir,
)

__all__ = [
    "colapsar_espacos",
    "html_para_texto",
    "normalizar",
    "normalizar_chave",
    "remover_acentos",
    "traduzir",
]
