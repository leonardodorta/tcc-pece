"""
domsp — extração de atos de contratação do Diário Oficial da Cidade de São Paulo.

## Os sete subpacotes, e a fronteira de cada um

    texto      HTML → texto, normalização, dobra de codificação
    fonte      o que a fonte devolve: edição bruta e o endpoint de download
    dominio    as regras de leitura do Diário
    contrato   quais colunas existem, de que tipo, e o que pode somar
    medicao    o gabarito automático e as métricas com intervalo
    modelo     rótulo BIO, treino do BERTimbau Large e o contrato de serving
    pipeline   os quatro estágios (bronze, silver, inferir, gold), em Python puro, sem Spark

A dependência corre numa direção só. Cada linha diz o que o subpacote importa de outros
subpacotes de `domsp`:

    texto      nada
    fonte      nada
    contrato   nada
    dominio    texto, contrato.vocabulario
    modelo     dominio, texto, fonte
    medicao    dominio, texto, fonte, modelo.rotulo, contrato.vocabulario
    pipeline   texto, fonte, dominio, contrato, medicao.incerteza (não importa modelo)

`dominio`, `contrato`, `texto` e `fonte` não importam pyspark, torch, transformers,
boto3, pandas nem numpy.
"""
from __future__ import annotations

__version__ = "1.0.2"

VERSAO_REGRAS = "1.0.2"
"""
Versão das regras determinísticas, gravada em `rule_version` na prata.

Deve ser incrementada sempre que uma regra de extração mudar o resultado, para que
cada linha indique com qual versão das regras foi produzida.
"""

MODELO_AUSENTE = "nenhum"
"""
Valor de `model_version` quando nenhum modelo participou da linha.

Literal e não nulo, de propósito: nulo em `model_version` seria ambíguo entre
"o modelo não rodou" e "o modelo rodou e não gravou versão".
"""

__all__ = ["MODELO_AUSENTE", "VERSAO_REGRAS", "__version__"]
