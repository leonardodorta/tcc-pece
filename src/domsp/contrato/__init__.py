"""
contrato — quais colunas existem, de que tipo, e o que pode somar.

    lago         onde cada coisa mora no S3, e qual tabela a lê
    esquema      a declaração autoritativa das colunas — fonte única
    financeiro   quando um valor pode entrar num somatório de gasto
    vocabulario  os valores fechados que as colunas admitem

## A fronteira

`contrato` não importa `dominio`, `medicao`, `modelo` nem `pipeline`. A direção é o
contrário: `dominio` e `pipeline` importam `contrato` (`dominio` só `contrato.vocabulario`).

`esquema.struct_type` importa pyspark dentro da função, de propósito: o módulo tem de
importar num laptop sem Spark, porque é onde ele é mais verificado.
"""
from __future__ import annotations

from . import esquema, financeiro, lago, vocabulario

__all__ = ["esquema", "financeiro", "lago", "vocabulario"]
