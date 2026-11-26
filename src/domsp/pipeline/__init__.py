"""
pipeline — os estágios, em Python puro, sem Spark.

    bronze   publicação bruta → linha tipada
    silver   triagem + as camadas determinísticas → uma linha por ato
    inferir  o resíduo → JSONL de entrada do modelo, e a volta
    gold     agregados sob a regra financeira, e as métricas

Cada função recebe dados e devolve dados: nenhuma abre arquivo ou fala com o S3. Por isso
o mesmo código pode ser executado localmente e nos jobs de `infraestrutura/jobs/`
sem adaptação.

`pipeline` importa `dominio`, `contrato`, `texto`, `fonte` e `medicao`. Não importa
`modelo`: a inferência roda noutro lugar, e `inferir.py` só conhece o formato do JSONL
que atravessa a fronteira, não o modelo que o produz.
"""
from __future__ import annotations

from . import bronze, gold, inferir, silver

__all__ = ["bronze", "gold", "inferir", "silver"]
