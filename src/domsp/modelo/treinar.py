"""
treinar.py — ajuste fino do classificador de tokens (BERTimbau), como entrypoint de
SageMaker Training Job e como script local.

Ajusta o classificador de tokens sobre o rótulo BIO que `rotulo.py` deriva da
redundância da fonte. Treina a variante escolhida por argumento (`--modelo`), grava o
checkpoint e um `metadata.json`.

## O corte por predicado

O conjunto de treino é o complemento exato do resíduo: os casos em que a camada 2 (o
rótulo determinístico na prosa) responde. O modelo, em produção, só vê o resíduo — os
casos em que ela não responde. A separação é estrutural: o mesmo predicado
determinístico não pode responder e não responder para o mesmo texto, então nenhum
caso de avaliação pode vazar para o treino.

O conjunto de avaliação é por isso enviesado por construção: é a parte difícil da
tarefa, não uma amostra aleatória dela. `separar_por_predicado` faz essa separação; a
CLI deste módulo (`main`) não a aplica e treina com os exemplos que recebe. A métrica de
qualidade é de `medicao/`, não deste módulo.

O predicado da camada 2 é recebido por parâmetro, como uma função `str -> bool` que diz
se o caso pertence ao treino, para que este módulo não precise importar o pacote
`pipeline`. No modo autônomo, `exemplos_do_corpus` lê os pares do corpus local.

## Torch puro, laço explícito, sem Trainer/datasets/accelerate

Cada dependência a mais é uma a menos de reprodutibilidade. `Trainer` traria
`accelerate`; um `Dataset` traria `datasets`; nenhum dos dois paga o seu peso aqui, e
ambos esconderiam a ordem dos lotes e a máscara de perda, que este módulo precisa
declarar.

## Lotes homogêneos em comprimento

Padding dinâmico por lote só economiza se o lote for homogêneo em comprimento. Com
lotes aleatórios, o custo de cada lote é o máximo dos comprimentos sorteados. A
economia vem de agrupar comprimentos parecidos.

A receita preserva a aleatoriedade entre épocas sem perder a economia: (1) embaralhar
os exemplos; (2) agrupar em blocos de `BLOCO_ORDENACAO` lotes; (3) ordenar por
comprimento **dentro** do bloco; (4) formar os lotes; (5) embaralhar a **ordem dos
lotes** de novo. A cada época a semente muda, então a ordem muda, mas continua função
da semente base — a execução inteira permanece reproduzível. `diagnostico_padding`
mede o desperdício das duas formas.

## Contrato do SageMaker, e também local

`SM_CHANNEL_TREINO` (entrada, lida por `_carregar_exemplos`) e `SM_MODEL_DIR` (saída)
são lidos do ambiente quando existem; senão, valem os caminhos por argumento. O entry
point do Training Job, `entrada_treino.py`, usa `SM_CHANNEL_TRAIN` e
`SM_CHANNEL_VALIDATION`.

## Referência

SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian
Portuguese. In: *Intelligent Systems (BRACIS 2020)*. Cham: Springer, 2020. p. 403–417.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# ── por que o import é ABSOLUTO ───────────────────────────────────────────────
# O arquivo pode rodar como script solto, isto é, como `__main__` sem pacote pai. Import
# relativo falha nesse caso com "attempted relative import with no known parent package".
from domsp.modelo import IGNORAR, INDICE, MODELO_BASE, MODELO_LARGE, ROTULOS
from domsp.modelo import rotulo as R

# Quantos lotes formam o bloco dentro do qual se ordena por comprimento. Bloco
# grande o bastante para a ordenação achar comprimentos parecidos, pequeno o
# bastante para não virar ordenação global (que mataria a aleatoriedade entre épocas).
BLOCO_ORDENACAO = 32

# Hiperparâmetros default, na faixa da literatura de fine-tune de BERT (Devlin et
# al., 2019; Souza et al., 2020 para BERTimbau): lr da ordem de 1e-5..5e-5, poucas
# épocas, aquecimento curto. Sobrescritos por argparse — uma comparação entre
# tamanhos exige os MESMOS hiperparâmetros, senão a comparação mede hiperparâmetro e
# não modelo.
LR_PADRAO = 3e-5
EPOCAS_PADRAO = 4
LOTE_PADRAO = 8
AQUECIMENTO_PADRAO = 0.1
MAX_TOKENS_PADRAO = 512
PASSO_JANELA_PADRAO = 128

ARQUIVO_METADADOS = "metadata.json"


# O EXEMPLO


@dataclass(frozen=True)
class Exemplo:
    """
    Um caso de treino: o texto e o nome correto da contratada, vindo da fonte.

    Guarda o nome, e não o span já calculado: o span é derivado por
    `rotulo.localizar`, que pode falhar (devolver None), e esse caso precisa ser
    contado.
    """

    texto: str
    nome_contratada: str


# AMBIENTE — torch importado só dentro de função


def _exigir_torch():
    """
    Importa torch e falha com mensagem útil se ausente.

    O import mora aqui, e não no topo, para que importar este módulo não exija `torch`/`transformers` instalados.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "treinar exige torch: pip install torch transformers"
        ) from e
    return torch


def dispositivo() -> str:
    """MPS em Apple Silicon, CUDA em GPU NVIDIA (SageMaker), CPU quando não há."""
    torch = _exigir_torch()
    if torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():  # pragma: no cover
        return "mps"
    return "cpu"


def _versoes(modelo: str) -> dict:
    """Versões que é preciso declarar para o treino ser reprodutível."""
    torch = _exigir_torch()
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "transformers": transformers.__version__,
        "modelo_base": modelo,
    }


def _sha256_arquivo(caminho: Path) -> str:
    """SHA-256 de um arquivo, lido em blocos (os pesos são grandes)."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _sha256_exemplos(exemplos: Sequence[Exemplo]) -> str:
    """
    Hash do conjunto de treino, estável e independente da ordem.

    Identifica o insumo do treino: duas execuções com o mesmo hash de gabarito
    treinaram sobre exatamente os mesmos pares, o que
    `infraestrutura/scripts/consolidar_execucoes.py` exige para calcular a média. Os pares
    são ordenados antes de hashear, para que a ordem de leitura não mude o hash.
    """
    h = hashlib.sha256()
    for texto, nome in sorted((e.texto, e.nome_contratada) for e in exemplos):
        h.update(texto.encode("utf-8"))
        h.update(b"\x00")
        h.update((nome or "").encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


# JANELAS E LOTES


def preparar(exemplos: Sequence[Exemplo], tokenizador,
             max_tokens: int = MAX_TOKENS_PADRAO,
             passo: int = PASSO_JANELA_PADRAO) -> tuple[list[R.Janela], dict]:
    """
    Exemplos → janelas tokenizadas com rótulo BIO, mais o diagnóstico do alinhamento.

    Usa `rotulo.localizar` + `rotulo.rotular_palavras` + `rotulo.janelas`. Um
    exemplo cujo span não é localizável é contado em `sem_span` e não gera janela:
    treinar o modelo com um texto rotulado todo `O` ensinaria a não achar nada.
    """
    janelas: list[R.Janela] = []
    sem_span = 0
    for ex in exemplos:
        span = R.localizar(ex.nome_contratada, ex.texto)
        if span is None:
            sem_span += 1
            continue
        rotulos_palavra = R.rotular_palavras(ex.texto, [span])
        janelas.extend(R.janelas(ex.texto, rotulos_palavra, tokenizador,
                                 max_tokens=max_tokens, passo=passo))
    diag = {
        "exemplos": len(exemplos),
        "sem_span_localizavel": sem_span,
        "janelas": len(janelas),
    }
    return janelas, diag


def _ordenar_por_bloco(indices: list[int], comprimentos: Sequence[int],
                       lote: int, rng: random.Random) -> list[list[int]]:
    """
    A receita dos lotes homogêneos: bloco → ordena dentro → forma lotes → embaralha.

    Recebe os índices já embaralhados. Devolve a lista de lotes (listas de índices) com
    a ordem dos lotes embaralhada — a aleatoriedade que sobrevive à ordenação por
    comprimento. Isolada de torch de propósito: é aritmética sobre inteiros.
    """
    bloco = lote * BLOCO_ORDENACAO
    lotes: list[list[int]] = []
    for i in range(0, len(indices), bloco):
        pedaco = sorted(indices[i:i + bloco], key=lambda k: comprimentos[k])
        lotes.extend(pedaco[j:j + lote] for j in range(0, len(pedaco), lote))
    rng.shuffle(lotes)
    return lotes


def montar_lotes(janelas: Sequence[R.Janela], lote: int, pad_id: int, semente: int):
    """
    Tensores de um passo, com padding dinâmico ao comprimento máximo DO LOTE.

    Devolve a lista completa (não um gerador) porque o número de passos por época
    precisa ser conhecido antes do primeiro passo: o agendador linear da taxa de
    aprendizado é definido em função dele.
    """
    torch = _exigir_torch()
    rng = random.Random(semente)
    ordem = list(range(len(janelas)))
    rng.shuffle(ordem)
    comprimentos = [len(j.ids) for j in janelas]
    grupos = _ordenar_por_bloco(ordem, comprimentos, lote, rng)

    saida = []
    for indices in grupos:
        grupo = [janelas[k] for k in indices]
        largura = max(len(j.ids) for j in grupo)
        saida.append({
            "input_ids": torch.tensor(
                [j.ids + [pad_id] * (largura - len(j.ids)) for j in grupo]),
            "attention_mask": torch.tensor(
                [j.mascara + [0] * (largura - len(j.mascara)) for j in grupo]),
            "labels": torch.tensor(
                [j.rotulos + [IGNORAR] * (largura - len(j.rotulos)) for j in grupo]),
        })
    return saida


def diagnostico_padding(janelas: Sequence[R.Janela], lote: int, semente: int) -> dict:
    """
    Quanto do cálculo é padding, com lotes aleatórios contra lotes homogêneos.

    Verificável sem GPU — é aritmética sobre os comprimentos.
    """
    reais = sum(len(j.ids) for j in janelas)
    comprimentos = [len(j.ids) for j in janelas]

    def custo(grupos: list[list[int]]) -> int:
        return sum(len(g) * max(comprimentos[k] for k in g) for g in grupos)

    rng = random.Random(semente)
    ordem = list(range(len(janelas)))
    rng.shuffle(ordem)
    aleatorios = [ordem[i:i + lote] for i in range(0, len(ordem), lote)]
    homogeneos = _ordenar_por_bloco(list(ordem), comprimentos, lote, random.Random(semente))

    c_al, c_ho = custo(aleatorios), custo(homogeneos)
    return {
        "tokens_reais": reais,
        "tokens_lote_aleatorio": c_al,
        "tokens_lote_homogeneo": c_ho,
        "padding_aleatorio": round(1 - reais / c_al, 4) if c_al else 0.0,
        "padding_homogeneo": round(1 - reais / c_ho, 4) if c_ho else 0.0,
        "economia": round(1 - c_ho / c_al, 4) if c_al else 0.0,
    }


# O TETO — importado de rotulo, nunca recalculado nem inventado


def teto_da_rotulagem(exemplos: Sequence[Exemplo]) -> dict:
    """
    O teto da supervisão sobre ESTE conjunto, medido por `rotulo.medir_ida_e_volta`.

    Vai para o `metadata.json` para que nenhuma métrica do modelo possa ser lida como
    maior que ele. É medido com a porta (`rotulo.porta_padrao`) porque a saída do
    modelo em produção também atravessa a porta. O teto isolado vai junto para a
    diferença (o custo da porta) ficar visível.
    """
    pares = [(ex.texto, ex.nome_contratada) for ex in exemplos]
    isolado = R.medir_ida_e_volta(pares, porta=None)
    realizavel = R.medir_ida_e_volta(pares, porta=R.porta_padrao)
    return {
        "casos": isolado.casos,
        "teto_isolado": isolado.teto,
        "teto_realizavel": realizavel.teto,
        "efeito_da_porta": round(realizavel.teto - isolado.teto, 4),
        "sem_span": isolado.sem_span,
    }


# O CORTE POR PREDICADO


def separar_por_predicado(
    exemplos: Sequence[Exemplo],
    resolvido_pela_camada2: Callable[[str], bool],
) -> tuple[list[Exemplo], list[Exemplo]]:
    """
    Separa (treino, residuo) pelo predicado da camada 2 — não por sorteio.

    `resolvido_pela_camada2(texto)` é True quando a camada determinística responde:
    esse caso vai para o treino. O resíduo é o complemento — o que sobra para o
    modelo em produção, e o único conjunto sobre o qual o modelo é de fato avaliado.

    Separar por predicado, e não por `shuffle` com semente, torna a ausência de
    vazamento uma propriedade verificável do código (o mesmo predicado não devolve
    True e False para o mesmo texto). Devolve `(treino, residuo)`, nesta ordem.
    """
    treino, residuo = [], []
    for ex in exemplos:
        (treino if resolvido_pela_camada2(ex.texto) else residuo).append(ex)
    return treino, residuo


# O TREINO


def treinar(
    exemplos: Sequence[Exemplo],
    modelo_base: str = MODELO_LARGE,
    *,
    semente: int = 42,
    epocas: int = EPOCAS_PADRAO,
    lote: int = LOTE_PADRAO,
    lr: float = LR_PADRAO,
    aquecimento: float = AQUECIMENTO_PADRAO,
    max_tokens: int = MAX_TOKENS_PADRAO,
    passo_janela: int = PASSO_JANELA_PADRAO,
    saida: Path | None = None,
    progresso: Callable[[int, int, float], None] | None = None,
) -> dict:
    """
    Ajusta o classificador de tokens e grava o checkpoint mais o `metadata.json`.

    Args:
        exemplos: o complemento do resíduo (ver `separar_por_predicado`).
        modelo_base: `MODELO_LARGE` ou `MODELO_BASE`. Uma comparação entre tamanhos
            deve chamar os dois com os demais argumentos iguais, senão a comparação
            mediria hiperparâmetro.
        semente: fixa `torch.manual_seed` e a ordem dos lotes. Registrada no metadata.
        saida: onde gravar. Default resolve o contrato do SageMaker (`SM_MODEL_DIR`).
        progresso: `(passo, total, perda_média)`, a cada passo.

    Returns:
        O dicionário gravado em `metadata.json`. A métrica de qualidade é de
        `medicao`, sobre o resíduo, pela porta comum.
    """
    torch = _exigir_torch()
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(semente)
    random.seed(semente)

    tok = AutoTokenizer.from_pretrained(modelo_base)
    janelas, diag = preparar(exemplos, tok, max_tokens=max_tokens, passo=passo_janela)
    if not janelas:
        raise ValueError(
            "nenhuma janela de treino — todos os spans falharam em localizar? "
            "verifique que os exemplos são o complemento do resíduo e têm nome"
        )

    lotes = montar_lotes(janelas, lote, tok.pad_token_id, semente)
    total_passos = len(lotes) * epocas

    dev = dispositivo()
    rede = AutoModelForTokenClassification.from_pretrained(
        modelo_base,
        num_labels=len(ROTULOS),
        id2label=dict(enumerate(ROTULOS)),
        label2id=dict(INDICE),
    ).to(dev)

    otimizador = torch.optim.AdamW(rede.parameters(), lr=lr)
    agenda = get_linear_schedule_with_warmup(
        otimizador,
        num_warmup_steps=int(total_passos * aquecimento),
        num_training_steps=total_passos,
    )

    rede.train()
    passo = 0
    perda_por_epoca: list[float] = []
    for epoca in range(epocas):
        # Reembaralha a cada época: a semente muda com a época para as passadas não
        # verem a mesma ordem, e continua função da semente base para a execução
        # permanecer reproduzível.
        do_epoca = lotes if epoca == 0 else montar_lotes(
            janelas, lote, tok.pad_token_id, semente + 1000 * (epoca + 1))
        soma = 0.0
        for i, batch in enumerate(do_epoca, start=1):
            batch = {k: v.to(dev) for k, v in batch.items()}
            resultado = rede(**batch)
            resultado.loss.backward()
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            otimizador.step()
            agenda.step()
            otimizador.zero_grad(set_to_none=True)
            passo += 1
            soma += float(resultado.loss.detach())
            if progresso is not None:
                progresso(passo, total_passos, soma / i)
        perda_por_epoca.append(round(soma / len(do_epoca), 5))
        # `perda_treino=<valor>` no stdout é o que as `MetricDefinitions` do Training Job
        # (`infraestrutura/scripts/lancar_treino.py`) extraem por expressão regular
        # (`perda_treino=([0-9\.]+)`), formando uma série temporal no CloudWatch.
        #
        # Emitido por época, e não por passo, para a série ter um ponto por época. A
        # granularidade por passo está no callback `progresso`.
        #
        # `flush=True` porque o SageMaker lê o stdout do contêiner em fluxo; sem ele o
        # buffer só é descarregado no fim e a série chega toda de uma vez.
        print(f"perda_treino={perda_por_epoca[-1]:.5f}", flush=True)
        print(f"epoca={epoca + 1}", flush=True)

    destino = Path(saida) if saida is not None else diretorio_de_saida()
    destino.mkdir(parents=True, exist_ok=True)
    rede.save_pretrained(destino)
    tok.save_pretrained(destino)

    # Proveniência que amarra ESTE checkpoint à execução que o produziu. Três âncoras:
    #   - `job_name`: o job de treino do SageMaker (ou um id local), que identifica a execução;
    #   - `sha256_pesos`: o hash de `model.safetensors`, que identifica os pesos exatos;
    #   - `gabarito.hash`: o hash do conjunto de treino, que identifica o insumo.
    # `consolidar_execucoes.py` usa `job_name` (execuções distintas) e `gabarito.hash`
    # (mesmo insumo) para recusar a média de execuções que não são comparáveis.
    job_name = (
        os.environ.get("SM_TRAINING_JOB_NAME")
        or os.environ.get("TRAINING_JOB_NAME")
        or os.environ.get("DOMSP_JOB_NAME")
        or f"local-{destino.name}"
    )
    pesos = destino / "model.safetensors"
    sha_pesos = _sha256_arquivo(pesos) if pesos.exists() else None
    sha_gabarito = _sha256_exemplos(exemplos)

    metadados = {
        "modelo_base": modelo_base,
        "job_name": job_name,
        "sha256_pesos": sha_pesos,
        "gabarito": {"hash": sha_gabarito, "exemplos_treino": len(exemplos)},
        "rotulos": list(ROTULOS),
        "ignorar": IGNORAR,
        "semente": semente,
        "hiperparametros": {
            "epocas": epocas, "lote": lote, "lr": lr, "aquecimento": aquecimento,
            "max_tokens": max_tokens, "passo_janela": passo_janela,
            "bloco_ordenacao": BLOCO_ORDENACAO, "semente": semente,
        },
        "corpus": {"exemplos_treino": len(exemplos)},
        "corte_por_predicado": (
            "conjunto de treino = complemento exato do resíduo (casos em que a camada 2 "
            "responde); avaliação enviesada por construção"
        ),
        "dispositivo": dev,
        "passos_por_epoca": len(lotes),
        "passos_totais": total_passos,
        "perda_por_epoca": perda_por_epoca,
        "perda_final": perda_por_epoca[-1],
        "diagnostico_alinhamento": diag,
        "teto_da_rotulagem": teto_da_rotulagem(exemplos),
        "padding": diagnostico_padding(janelas, lote, semente),
        "versoes": _versoes(modelo_base),
    }
    (destino / ARQUIVO_METADADOS).write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    metadados["diretorio"] = str(destino)

    # O teto realizável, impresso no log. É o limite que a rotulagem por alinhamento
    # impõe: nenhuma métrica deste modelo pode ser lida como maior que ele.
    print(f"teto={metadados['teto_da_rotulagem']['teto_realizavel']:.4f}", flush=True)
    return metadados


# CONTRATO DO SAGEMAKER — e também local


def diretorio_de_saida(padrao: Path | None = None) -> Path:
    """`SM_MODEL_DIR` quando existe, senão o argumento, senão `./modelo_treinado`."""
    env = os.environ.get("SM_MODEL_DIR")
    if env:
        return Path(env)
    return Path(padrao) if padrao is not None else Path("modelo_treinado")


# CARGA DE EXEMPLOS — modo autônomo, sem o pacote `pipeline` nem pyspark


def exemplos_de_jsonl(caminho: Path) -> list[Exemplo]:
    """
    Lê exemplos de um JSONL com `{text|texto|documento_texto, contratada|nome_contratada}`.

    Aceita mais de um nome para o campo de texto. Linhas sem texto ou sem nome da
    contratada são ignoradas.
    """
    exemplos: list[Exemplo] = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            d = json.loads(linha)
            # `text`, `texto` e `documento_texto` são nomes aceitos para o campo de texto.
            texto = d.get("text") or d.get("texto") or d.get("documento_texto") or ""
            nome = d.get("contratada") or d.get("nome_contratada") or ""
            if texto and nome:
                exemplos.append(Exemplo(texto, nome))
    return exemplos


def exemplos_do_corpus(n_edicoes: int = 6) -> list[Exemplo]:
    """
    Os pares (prosa, contratada) que a redundância da fonte produz, do corpus local.

    Existe para o modo autônomo (script local e treino-fumaça) poder rodar sem o
    pipeline. Os imports ficam dentro da função porque só o modo autônomo os usa.
    """
    from .. import texto as T
    from ..dominio import formulario as F
    from ..dominio import triagem
    from ..fonte import edicao as fonte

    raiz = Path(__file__).resolve().parents[3]  # a raiz do projeto
    corpus = raiz / "dados" / "raw"
    if not corpus.exists():
        raise SystemExit(
            f"não encontrei o corpus em {corpus}. Passe --entrada com um JSONL de exemplos, "
            f"ou monte o corpus em dados/raw."
        )
    exemplos: list[Exemplo] = []
    for arquivo in fonte.arquivos(corpus)[:n_edicoes]:
        ed = fonte.carregar(arquivo)
        for pub in ed.publicacoes:
            html = pub.get("conteudo") or ""
            form = F.ler(html)
            if not form.existe or not form.tem_prosa:
                continue
            completo = T.html_para_texto(html)
            alvo = F.alvo_da_analise(form, completo)
            # `alvo` é passado para que a correção do campo do formulário truncado seja
            # aplicada aqui também.
            resultado = F.contratada_do_formulario(form, alvo)
            if not resultado.nome:
                continue
            if triagem.motivo_recusa(pub.get("serie"), pub.get("veiculo"), alvo):
                continue
            if R.localizar(resultado.nome, alvo) is None:
                continue
            exemplos.append(Exemplo(alvo, resultado.nome))
    return exemplos


# CLI


def _argumentos(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune do BERTimbau para rotulagem de token da parte contratada")
    p.add_argument("--modelo", choices=("large", "base"), default="large",
                   help="qual pré-treinado ajustar (padrão: large)")
    p.add_argument("--entrada", default=None,
                   help="JSONL de exemplos; default lê SM_CHANNEL_TREINO ou o corpus local")
    p.add_argument("--saida", default=None,
                   help="diretório do checkpoint; default SM_MODEL_DIR ou ./modelo_treinado")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--epocas", type=int, default=EPOCAS_PADRAO)
    p.add_argument("--lote", type=int, default=LOTE_PADRAO)
    p.add_argument("--lr", type=float, default=LR_PADRAO)
    p.add_argument("--aquecimento", type=float, default=AQUECIMENTO_PADRAO)
    p.add_argument("--edicoes-corpus", type=int, default=6,
                   help="quantas edições ler quando a entrada é o corpus local")
    p.add_argument("--passos-fumaca", type=int, default=0,
                   help="se > 0, treina só este número de passos (verificação)")
    return p.parse_args(argv)


def _carregar_exemplos(args: argparse.Namespace) -> list[Exemplo]:
    """
    A entrada, por ordem: `--entrada` explícito, canal do SageMaker, corpus local.

    O canal é um diretório com um ou mais arquivos JSONL, lido por `exemplos_do_canal`.
    """
    if args.entrada:
        return exemplos_de_jsonl(Path(args.entrada))
    env = os.environ.get("SM_CHANNEL_TREINO")
    if env:
        return exemplos_do_canal(Path(env))
    return exemplos_do_corpus(args.edicoes_corpus)


def exemplos_do_canal(diretorio: Path) -> list[Exemplo]:
    """
    Os exemplos de um canal do SageMaker, que é um diretório com arquivos de partição.

    O padrão não é `*.jsonl` porque escritores de dados particionados, como o Spark,
    gravam arquivos `part-*.json` e um marcador `_SUCCESS` vazio, sem extensão `.jsonl`.

    Aceita qualquer arquivo que não seja marcador do Spark (`_SUCCESS`, `.crc`) nem
    oculto; o conteúdo é JSONL em todos os casos, e é o conteúdo que decide, não o
    nome. Recursivo porque um canal pode vir particionado em subdiretórios.
    """
    arquivos = sorted(
        a for a in diretorio.rglob("*")
        if a.is_file()
        and not a.name.startswith((".", "_"))
        and not a.name.endswith(".crc")
    )
    exemplos: list[Exemplo] = []
    for arquivo in arquivos:
        exemplos.extend(exemplos_de_jsonl(arquivo))
    return exemplos


def main(argv: Sequence[str] | None = None) -> None:
    args = _argumentos(argv)
    modelo_base = MODELO_LARGE if args.modelo == "large" else MODELO_BASE
    exemplos = _carregar_exemplos(args)
    if not exemplos:
        raise SystemExit("nenhum exemplo carregado — verifique --entrada ou o corpus")

    epocas = args.epocas
    if args.passos_fumaca > 0:
        # Treino-fumaça: uma época, poucos exemplos. Não é modelo publicável; serve para
        # provar que o laço roda e a perda desce.
        exemplos = exemplos[: max(args.lote, args.passos_fumaca * args.lote)]
        epocas = 1

    def mostrar(passo: int, total: int, perda: float) -> None:
        if passo == 1 or passo % 5 == 0 or passo == total:
            print(f"  passo {passo}/{total}  perda {perda:.4f}", flush=True)

    meta = treinar(
        exemplos, modelo_base,
        semente=args.semente, epocas=epocas, lote=args.lote, lr=args.lr,
        aquecimento=args.aquecimento,
        saida=Path(args.saida) if args.saida else None,
        progresso=mostrar,
    )
    print(f"\n  → checkpoint em {meta['diretorio']}")
    print(f"  perda por época: {meta['perda_por_epoca']}")
    print(f"  teto realizável da rotulagem: {meta['teto_da_rotulagem']['teto_realizavel']}")


if __name__ == "__main__":
    main()
