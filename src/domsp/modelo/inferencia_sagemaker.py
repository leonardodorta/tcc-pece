"""
inferencia_sagemaker.py — os quatro ganchos que o contêiner do SageMaker chama.

É copiado como `code/inference.py` para dentro do `model.tar.gz` (ver `empacotar.py`), junto com o wheel do `domsp`, que
`code/requirements.txt` manda o contêiner instalar. Por isso este arquivo importa o
algoritmo de `domsp.modelo.servir` em vez de reimplementá-lo — garante que a inferência
do contêiner e a da avaliação sejam exatamente o mesmo código.

Ganchos e ordem de chamada:

    model_fn(dir)                  uma vez, ao subir o contêiner
    input_fn(corpo, tipo)          por requisição — desserializa
    predict_fn(entrada, modelo)    por requisição — infere
    output_fn(saida, tipo)         por requisição — serializa

`model_fn`, `input_fn` e `output_fn` são reexportados de `servir` sem alteração. Só
`predict_fn` é próprio, porque fala o contrato do pipeline (`{contract_id, spans[],
truncated}`) e delega o cálculo inteiro a `servir.inferir_texto`.

Se o import de `servir` falhar, este módulo levanta `ImportError` na subida do contêiner,
com instrução de reempacotar.
"""
from __future__ import annotations

import logging

try:
    from domsp.modelo.servir import (
        TIPO_JSON,
        TIPO_JSONL,
        inferir_texto,
        input_fn,
        model_fn,
        output_fn,
    )
except ImportError as erro:  # pragma: no cover - só ocorre em pacote mal montado
    raise ImportError(
        "code/inference.py não encontrou domsp.modelo.servir. O pacote precisa trazer o "
        "wheel do domsp em code/ e referenciá-lo em code/requirements.txt — é o que faz a "
        "inferência do contêiner ser o mesmo algoritmo da avaliação. Reempacote com "
        "`infraestrutura/scripts/empacotar_modelo.py` depois de `python -m build --wheel`"
    ) from erro

#: O limite de tokens da janela. O mesmo do treino, e o mesmo que a avaliação usa por
#: default em `servir.inferir_texto`.
MAX_TOKENS_JANELA = 512

_log = logging.getLogger(__name__)


def predict_fn(atos: list[dict], modelo) -> list[dict]:
    """
    Para cada ato, os spans que o modelo apontou — no contrato que o pipeline consome.

    Devolve uma linha por ato, com `{contract_id, spans[], truncated}`, que é o que
    `pipeline.inferir.da_saida_do_modelo` lê. Os spans vêm inteiramente de
    `servir.inferir_texto`: janelamento, fusão entre janelas, escore e decodificação são
    o mesmo código da avaliação.

    Não há filtro por escore aqui — a validação é a porta de nome em
    `da_saida_do_modelo`, a mesma das três camadas determinísticas, que grava `aceito` e
    `motivo_recusa` ao lado do dado.

    `truncated` informa que o texto excedeu a janela e foi cortado — o consumidor registra
    isso como ressalva da linha.
    """
    linhas: list[dict] = []
    for ato in atos:
        identificador = ato.get("contract_id") or ato.get("documento")
        texto = ato.get("text") or ato.get("texto") or ""
        spans = [
            {
                "class": sp.rotulo,
                "text": sp.span,
                "start": sp.inicio,
                "end": sp.fim,
                "score": sp.escore,
            }
            for sp in inferir_texto(texto, modelo, max_tokens=MAX_TOKENS_JANELA)
        ]
        linhas.append({
            "contract_id": identificador,
            "spans": spans,
            "truncated": _excedeu_a_janela(texto, modelo),
        })
    return linhas


def _excedeu_a_janela(texto: str, modelo) -> bool:
    """
    Se o texto, tokenizado, passa do limite da janela.

    Medido sobre o tokenizador do próprio modelo, e não por contagem de palavras: a
    pergunta é se o modelo viu o texto inteiro, e isso se decide em token.
    """
    if not texto:
        return False
    try:
        return len(modelo.tokenizador(texto).get("input_ids") or []) > MAX_TOKENS_JANELA
    except Exception:  # noqa: BLE001 - a ressalva não vale interromper o lote
        # A flag `truncated` é informativa e não deve abortar o lote; o aviso no log evita
        # que um tokenizador quebrado passe sem rastro.
        _log.warning("falha ao medir truncamento do texto; assumindo não truncado",
                     exc_info=True)
        return False


__all__ = [
    "MAX_TOKENS_JANELA",
    "TIPO_JSON",
    "TIPO_JSONL",
    "input_fn",
    "model_fn",
    "output_fn",
    "predict_fn",
]
