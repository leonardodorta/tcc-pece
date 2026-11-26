"""
servir.py — a inferência, como entrypoint de SageMaker Batch Transform e como script local

As quatro funções que o contêiner PyTorch do SageMaker chama — `model_fn`, `input_fn`,
`predict_fn`, `output_fn` — estão aqui, cada uma com uma única responsabilidade. Fora do
SageMaker, `inferir_jsonl` compõe as quatro para rodar localmente pelo mesmo caminho.

Entrada: JSONL, uma linha por ato do resíduo, com `documento` e `texto` (a prosa que o
modelo lê). Saída de `predict_fn`: uma linha por span, com as chaves `documento,
data_publicacao, span, rotulo, escore, inicio, fim, modelo, aceito, motivo_recusa`. Os
nomes não são os de `contrato.esquema.INFERENCIA`; o formato que o pipeline consome é o
de `inferencia_sagemaker.predict_fn`, lido por `pipeline.inferir`.

Todo span devolvido pelo modelo passa por `dominio.parte.porta_de_nome`, a mesma porta
das camadas determinísticas (formulário e rótulo em prosa). `aceito` e `motivo_recusa`
são gravados por span.

O modelo pode dar escore alto a um span que ocupa a posição sintática de contratada mas
não é o nome da parte — o PAPEL contratual ('DONATÁRIA', 'DETENTORA') ou o SIGNATÁRIO
('SR. FULANO DE TAL'). Escore não separa esses casos do nome verdadeiro, porque os três
aparecem na mesma posição; quem separa é a validação estrutural da porta, que recusa
papel (RECUSA_PAPEL) e signatário (RECUSA_SIGNATARIO) por regra.

O escore de um span é o mínimo das probabilidades das palavras que o compõem (não a
média): o span só está certo se todas as suas palavras pertencem à entidade, e uma
palavra de baixa probabilidade no meio do nome é evidência de fronteira errada.

Um texto pode passar do limite de 512 tokens do modelo. `rotulo.janelas` produz janelas
com sobreposição (`passo` < `max_tokens`); este módulo infere por janela, projeta os
subtokens de volta à palavra (a grade que `rotulo.py` declara) e reúne as predições entre
janelas palavra a palavra: uma palavra vista por mais de uma janela fica com a predição
de maior escore. Os spans saem da sequência de rótulos por palavra, seguindo a mesma
convenção IOB2 de `rotulo.decodificar`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from domsp.modelo import ROTULOS
from domsp.modelo import rotulo as R
from domsp.modelo.rotulo import palavras

# O contêiner PyTorch do SageMaker negocia estes tipos de conteúdo. JSONL é o formato do
# Batch Transform quando `SplitType=Line`. O literal é `application/jsonlines`, e não
# `application/jsonl`, porque é o valor usado em `ContentType`/`Accept`.
TIPO_JSONL = "application/jsonlines"
TIPO_JSON = "application/json"

# Classes de span. CONTRATANTE existe no esquema para o modelo distinguir o polo público
# da contratada: um span CONTRATANTE é decodificado e registrado, mas não é candidato a
# parte contratada.
CLASSE_CONTRATADA = "CONTRATADA"
CLASSE_CONTRATANTE = "CONTRATANTE"


@dataclass
class Modelo:
    """O checkpoint pronto para inferir, com o que o treino registrou sobre ele."""

    tokenizador: object
    rede: object
    dispositivo: str
    identificador: str
    metadados: dict = field(default_factory=dict)


# AMBIENTE — torch/transformers só dentro de função.


def _exigir_torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "servir exige torch: pip install torch transformers"
        ) from e
    return torch


def _dispositivo() -> str:
    torch = _exigir_torch()
    if torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():  # pragma: no cover
        return "mps"
    return "cpu"


def model_fn(diretorio_modelo: str) -> Modelo:
    """
    Carrega o checkpoint e confere o `id2label` salvo contra `ROTULOS`.

    A conferência impede que uma reordenação de `ROTULOS` posterior ao treino troque
    CONTRATADA por CONTRATANTE sem erro, o que registraria o órgão público como
    contratada.
    """
    _exigir_torch()
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    rede = AutoModelForTokenClassification.from_pretrained(diretorio_modelo)
    salvos = [rede.config.id2label[i] for i in sorted(rede.config.id2label)]
    if tuple(salvos) != tuple(ROTULOS):
        raise RuntimeError(
            "id2label do checkpoint difere de modelo.ROTULOS:\n"
            f"  checkpoint: {salvos}\n  modelo:     {list(ROTULOS)}\n"
            "as classes seriam interpretadas trocadas; retreine ou restaure a ordem"
        )

    identificador = diretorio_modelo
    meta_arq = Path(diretorio_modelo) / "metadata.json"
    metadados: dict = {}
    if meta_arq.exists():
        metadados = json.loads(meta_arq.read_text(encoding="utf-8"))
        identificador = metadados.get("modelo_base", diretorio_modelo)

    dev = _dispositivo()
    rede.to(dev).eval()
    return Modelo(
        tokenizador=AutoTokenizer.from_pretrained(diretorio_modelo),
        rede=rede, dispositivo=dev, identificador=identificador, metadados=metadados,
    )


def input_fn(corpo: str | bytes, tipo_conteudo: str = TIPO_JSONL) -> list[dict]:
    """
    Uma linha JSON por ato → lista de dicionários com pelo menos `documento` e `texto`.

    Aceita JSONL (o modo de lote) e um JSON único (uma requisição de teste). Linhas em
    branco são ignoradas; uma linha malformada erra alto em vez de sumir, para não
    perder um ato em silêncio.
    """
    if isinstance(corpo, bytes):
        corpo = corpo.decode("utf-8")
    corpo = corpo.strip()
    if not corpo:
        return []
    if tipo_conteudo == TIPO_JSON or (corpo[0] == "[" and corpo[-1] == "]"):
        dados = json.loads(corpo)
        return dados if isinstance(dados, list) else [dados]
    atos: list[dict] = []
    for n, linha in enumerate(corpo.splitlines(), start=1):
        linha = linha.strip()
        if not linha:
            continue
        try:
            atos.append(json.loads(linha))
        except json.JSONDecodeError as e:  # pragma: no cover
            raise ValueError(f"linha {n} do JSONL malformada: {e}") from e
    return atos


@dataclass(frozen=True)
class SpanPredito:
    """Um span que o modelo apontou, antes da porta: trecho, classe, escore, posição."""

    span: str
    rotulo: str
    escore: float
    inicio: int
    fim: int


def _rotulo_por_palavra(texto: str, modelo: Modelo, max_tokens: int,
                        passo: int) -> list[tuple[str, float]]:
    """
    Predição por PALAVRA (rótulo IOB2, escore), reunindo as janelas deslizantes.

    Roda o modelo por janela, projeta o subtoken de volta à palavra pelo `ids_palavra`
    que `rotulo.janelas` fornece (rótulo no primeiro subtoken da palavra, os demais
    ignorados — a mesma grade do treino), e reúne entre janelas: uma palavra vista por
    duas janelas fica com a predição de maior escore (e qualquer predição diferente de
    'O' substitui o 'O' padrão).
    """
    torch = _exigir_torch()
    pals = palavras(texto)
    if not pals:
        return []

    # Rótulo default: 'O' com escore 1.0 (palavra que nenhuma janela cobriu não é
    # entidade). Sobrescrito quando uma janela a prediz.
    melhor: list[tuple[str, float]] = [("O", 1.0)] * len(pals)

    janelas = R.janelas(texto, ["O"] * len(pals), modelo.tokenizador,
                        max_tokens=max_tokens, passo=passo)
    with torch.no_grad():
        for jan in janelas:
            ids = torch.tensor([jan.ids]).to(modelo.dispositivo)
            att = torch.tensor([jan.mascara]).to(modelo.dispositivo)
            logits = modelo.rede(input_ids=ids, attention_mask=att).logits[0]
            probs = torch.softmax(logits, dim=-1)
            escores, indices = probs.max(dim=-1)
            escores = escores.cpu().tolist()
            indices = indices.cpu().tolist()
            vistos: set[int] = set()
            for pos, idp in enumerate(jan.ids_palavra):
                if idp is None or idp in vistos:
                    continue
                vistos.add(idp)
                classe = ROTULOS[indices[pos]]
                esc = float(escores[pos])
                if esc > melhor[idp][1] or melhor[idp][0] == "O" and classe != "O":
                    melhor[idp] = (classe, esc)
    return melhor


def _spans_de(texto: str, rotulo_palavra: Sequence[tuple[str, float]],
              classe: str) -> list[SpanPredito]:
    """
    Decodifica os spans de uma classe e agrega o escore por mínimo das palavras.

    A segmentação segue a mesma convenção de `rotulo.decodificar` (`B-` abre, `I-` da mesma
    classe continua, qualquer outra coisa fecha), mas aqui também são devolvidas as
    posições em caractere e o escore, então a grade de palavras é percorrida em paralelo
    com os rótulos. O escore do span é o mínimo dos escores das suas palavras, porque o
    span é uma conjunção (ver docstring do módulo).
    """
    pals = palavras(texto)
    rotulos = [r for r, _ in rotulo_palavra]
    _ = R.decodificar(texto, rotulos, classe)

    spans: list[SpanPredito] = []
    atual: list[int] = []  # índices de palavra do span em construção
    abre, dentro = f"B-{classe}", f"I-{classe}"

    def fechar():
        if not atual:
            return
        ini, fim = pals[atual[0]].inicio, pals[atual[-1]].fim
        esc = min(rotulo_palavra[i][1] for i in atual)
        spans.append(SpanPredito(texto[ini:fim], classe, round(esc, 4), ini, fim))

    for i, r in enumerate(rotulos):
        if r == abre:
            fechar()
            atual = [i]
        elif r == dentro and atual:
            atual.append(i)
        else:
            fechar()
            atual = []
    fechar()
    return spans


def inferir_texto(texto: str, modelo: Modelo, *, max_tokens: int = 512,
                  passo: int = 128) -> list[SpanPredito]:
    """Todos os spans (das duas classes) que o modelo aponta num texto, antes da porta."""
    rot = _rotulo_por_palavra(texto, modelo, max_tokens, passo)
    return (_spans_de(texto, rot, CLASSE_CONTRATADA)
            + _spans_de(texto, rot, CLASSE_CONTRATANTE))


def predict_fn(atos: list[dict], modelo: Modelo) -> list[dict]:
    """
    Para cada ato, infere os spans e passa cada um pela porta de `dominio.parte`.

    A porta é `porta_de_nome`, a mesma das camadas determinísticas. Para os spans de
    CONTRATADA, o polo público conhecido (quando o ato o traz em `contratante_conhecido`)
    é passado à porta, que assim recusa a Administração devolvida como contratada.
    Spans de CONTRATANTE são registrados com `aceito=False` e um motivo próprio: eles não
    são candidatos a parte contratada por definição, e essa recusa é estrutural, não do
    escore.

    Cada span vira uma linha com as chaves listadas no docstring do módulo, incluindo
    `aceito` e `motivo_recusa`.
    """
    from ..dominio import parte

    linhas: list[dict] = []
    for ato in atos:
        documento = ato.get("documento")
        data = ato.get("data_publicacao")
        texto = ato.get("texto") or ""
        lado_publico = ato.get("contratante_conhecido")
        for sp in inferir_texto(texto, modelo):
            if sp.rotulo == CLASSE_CONTRATADA:
                resultado = parte.porta_de_nome(
                    sp.span, lado_publico_conhecido=lado_publico)
                aceito = resultado.nome is not None
                motivo = "" if aceito else (resultado.ressalvas[-1]
                                            if resultado.ressalvas else "recusado")
            else:
                # CONTRATANTE nunca é resposta de parte contratada — recusa estrutural.
                aceito = False
                motivo = "polo_publico_nao_e_contratada"
            linhas.append({
                "documento": documento,
                "data_publicacao": data,
                "span": sp.span,
                "rotulo": sp.rotulo,
                "escore": sp.escore,
                "inicio": sp.inicio,
                "fim": sp.fim,
                "modelo": modelo.identificador,
                "aceito": aceito,
                "motivo_recusa": motivo,
            })
    return linhas


def output_fn(linhas: list[dict], tipo_aceito: str = TIPO_JSONL) -> str:
    """Linhas → JSONL (uma por span), a saída que o Batch Transform grava no S3."""
    if tipo_aceito == TIPO_JSON:
        return json.dumps(linhas, ensure_ascii=False)
    return "\n".join(json.dumps(linha, ensure_ascii=False) for linha in linhas)


def inferir_jsonl(diretorio_modelo: str, entrada: Path, saida: Path) -> int:
    """
    Roda a inferência de ponta a ponta sobre um JSONL local, compondo as quatro funções.

    É o mesmo caminho de código do Batch Transform — `model_fn` → `input_fn` →
    `predict_fn` → `output_fn` — só que amarrado aqui em vez de pelo contêiner. Devolve
    o número de linhas (spans) escritas.
    """
    modelo = model_fn(diretorio_modelo)
    atos = input_fn(Path(entrada).read_text(encoding="utf-8"))
    linhas = predict_fn(atos, modelo)
    Path(saida).write_text(output_fn(linhas), encoding="utf-8")
    return len(linhas)


def _argumentos(argv: Sequence[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(
        description="Inferência do BERTimbau sobre o resíduo (Batch Transform ou local)")
    p.add_argument("--modelo", required=True, help="diretório do checkpoint")
    p.add_argument("--entrada", required=True, help="JSONL de atos do resíduo")
    p.add_argument("--saida", required=True, help="JSONL de saída (spans)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _argumentos(argv)
    n = inferir_jsonl(args.modelo, Path(args.entrada), Path(args.saida))
    print(f"  → {n} spans escritos em {args.saida}")


if __name__ == "__main__":
    main()
