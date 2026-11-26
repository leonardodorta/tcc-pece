"""
fonte — o que o Diário Oficial devolve, antes de qualquer interpretação.

    edicao.py   carregar o arquivo bruto de uma edição, e o que dele é confiável
    baixar.py   o endpoint que o site usa, e como ele diz "não há edição"

Trata a dobra de codificação (UTF-8 gravado e relido como latin-1) por edição, e trata
HTTP 500 de corpo vazio ou curto como "não há edição nesta data", não como falha.
"""
from __future__ import annotations

from . import baixar, edicao

__all__ = ["baixar", "edicao"]
