"""
dominio — as regras de leitura do Diário Oficial.

Esconde uma decisão só: **como a fonte escreve.** Que séries existem, o que é um ato de
contratação, como um CNPJ se valida, como um valor por extenso confirma um numeral, como
a fonte redige uma data, quando um valor pode somar. Nada aqui sabe onde o código roda nem
onde o dado é gravado.

## Os módulos

    triagem     o que é ato de contratação e o que não é, com o motivo da recusa
    formulario  o formulário do Painel de Negócios: campos, layout, prosa
    parte       quem contratou e quem foi contratado; documento fiscal e seu DV
    dinheiro    valor do ajuste, numeral × extenso, grandeza semântica
    ato         tipo e natureza do ato, anulação e republicação
    tempo       assinatura, vigência e prazo normalizado em meses
    objeto      o objeto contratado — o campo do formulário e o recorte na prosa

## A fronteira

Um módulo daqui não pode importar pyspark, torch,
transformers, boto3, pandas, numpy, awsglue, requests ou duckdb, nem importar
`medicao`, `modelo` ou `pipeline`. De `contrato`, só `contrato.vocabulario` é permitido, e só porque é folha: é a
declaração única dos valores que as colunas admitem, e importá-la é ler o vocabulário, não
depender do esquema.

## O que o dominio decide, e o modelo não

O modelo não decide validade de CNPJ, grandeza monetária, valor somável, natureza
financeira, anulação, normalização de data ou prazo normalizado. Todas essas decisões vivem
aqui, e o modelo entra só onde NER acrescenta: o span de `CONTRATADA` e de `OBJETO` na prosa
livre. O modelo identifica spans; estas regras validam e normalizam.
"""
from __future__ import annotations

from . import ato, dinheiro, formulario, objeto, parte, tempo, triagem

__all__ = ["ato", "dinheiro", "formulario", "objeto", "parte", "tempo", "triagem"]
