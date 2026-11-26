"""
ajustar — o laço de treino com validação por época, early stopping e checkpointing.

Reaproveita de `treinar.py` o tipo `Exemplo`, `montar_lotes` (lotes ordenados por
bloco), `dispositivo` e `_versoes`. A seleção de janelas (`preparar_janelas`), o laço de
treino e o `metadata.json` são próprios deste módulo. O modelo é o de token
classification, com `num_labels`, `id2label` e `label2id` salvos no checkpoint.

O que este módulo oferece:

    gradient checkpointing         reduz a memória de ativações, trocando por tempo
    acumulação de gradiente        lote efetivo múltiplo do lote físico
    validação por época            F1 de entidade, não acurácia de token
    early stopping                 para quando o F1 de validação deixa de subir
    save_best_only                 grava o melhor, não o último
    retomada                       o treino pode continuar após uma interrupção

## Por que F1 de entidade e não acurácia de token

A acurácia de token é enganosa nesta tarefa: predominam palavras `O`, então um modelo
que responda `O` para tudo acerta a maior parte dos tokens e não acha nenhum nome.

O F1 aqui é de **span exato**: o span predito conta como acerto quando coincide com o
do gabarito em começo e fim.

## Por que MPS pede cuidado

O backend `mps` do PyTorch não implementa toda operação. Duas consequências tratadas
neste módulo:

- **`PYTORCH_ENABLE_MPS_FALLBACK`** deixa a operação faltante cair para a CPU em vez de
  interromper o treino.
- **A memória não é liberada como numa GPU discreta.** `torch.mps.empty_cache()`,
  chamado periodicamente, evita o crescimento contínuo do uso que termina em `MPS
  backend out of memory`.
"""
from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import CONTRATADA, IGNORAR, INDICE, MODELO_LARGE, ROTULOS
from . import rotulo as R
from . import treinar as T

# O fallback tem de ser ligado ANTES de o torch inicializar o backend, e importar este módulo deve ser
# a primeira coisa que o script de treino faz. Colocar isto dentro da função seria tarde.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

LOTE_EFETIVO_PADRAO = 8
PACIENCIA_PADRAO = 2
ARQUIVO_RETOMADA = "retomada.pt"
ARQUIVO_HISTORICO = "historico.json"

# QUANTAS JANELAS POR EXEMPLO, E POR QUE HÁ UM TETO
#
# Um texto longo produz várias janelas de `max_tokens` tokens com passo `passo`, e o número
# de janelas cresce com o comprimento do exemplo.
#
# O teto não é só economia de memória; também evita um desequilíbrio de classes:
#
# **A entidade está em poucas janelas, não em todas.** Em um exemplo longo, a maioria das
# janelas não contém nenhum rótulo diferente de `O`. Treinar sobre todas elas ensina o modelo
# a não achar nada e o desloca para a resposta vazia, caso em que a acurácia de token sobe
# enquanto o F1 de entidade cai.
#
# Então a seleção é declarada: **todas** as janelas que contêm a entidade, mais um número
# limitado de janelas negativas por exemplo. As negativas não são descartadas de todo porque o
# modelo precisa aprender a recusar — um treino só com positivas produziria um modelo que acha
# entidade em qualquer texto.
MAX_NEGATIVAS_POR_EXEMPLO = 2

# A cada quantos passos do otimizador a memória do backend é liberada (ver
# `_liberar_memoria`). No MPS a memória não volta ao pool sozinha, e sem a liberação
# periódica o treino pode terminar em `MPS backend out of memory`.
INTERVALO_LIBERAR_MEMORIA = 10


def dispositivo_escolhido() -> str:
    """
    O dispositivo do treino, com `DOMSP_DISPOSITIVO` podendo forçar.

    ## Por que a variável existe

    O MPS do Apple Silicon conta a memória **unificada inteira** no seu limite: a mensagem de
    erro traz `MPS allocated` e `other allocations` somados contra um `max allowed`. Outros
    processos do sistema entram em `other allocations`, então a memória disponível ao treino
    depende do que mais está em uso na máquina.

    A CPU não tem esse limite: usa memória virtual com swap. É mais lenta por janela, mas a
    configuração do experimento é a mesma (modelo, hiperparâmetros, semente, dados). O
    dispositivo vai para o `metadata.json`, que é onde a diferença fica registrada.
    """
    forcado = os.environ.get("DOMSP_DISPOSITIVO", "").strip().lower()
    if forcado in ("cpu", "mps", "cuda"):
        return forcado
    return T.dispositivo()


@dataclass
class Epoca:
    """O que uma época produziu. Vai para o histórico e para o `metadata.json`."""

    numero: int
    perda_treino: float
    precisao: float
    revocacao: float
    f1: float
    exatos: int
    preditos: int
    esperados: int
    segundos: float
    memoria_gib: float

    def como_dicionario(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Resultado:
    """O resultado do ajuste: o diretório do melhor checkpoint e o histórico completo."""

    diretorio: Path
    melhor_epoca: int
    melhor_f1: float
    historico: list[Epoca] = field(default_factory=list)
    parou_por_paciencia: bool = False
    metadados: dict = field(default_factory=dict)


# A MÉTRICA DE VALIDAÇÃO


def _spans_do_bio(rotulos: Sequence[str]) -> set[tuple[int, int]]:
    """
    Os spans de `CONTRATADA` como pares (início, fim) em índice de palavra.

    Conjunto e não lista: a comparação entre predito e esperado é de conjuntos, e um span
    duplicado numa das listas inflaria a contagem sem corresponder a acerto.
    """
    spans: set[tuple[int, int]] = set()
    inicio: int | None = None
    for i, r in enumerate(rotulos):
        if r == f"B-{CONTRATADA}":
            if inicio is not None:
                spans.add((inicio, i))
            inicio = i
        elif r == f"I-{CONTRATADA}":
            if inicio is None:
                # `I-` órfão: trata como começo. A alternativa — descartar — perderia predição
                # legítima do modelo, que às vezes começa a entidade com `I-`.
                inicio = i
        else:
            if inicio is not None:
                spans.add((inicio, i))
                inicio = None
    if inicio is not None:
        spans.add((inicio, len(rotulos)))
    return spans


def avaliar_f1(
    rede,
    tok,
    exemplos: Sequence[T.Exemplo],
    *,
    dispositivo: str,
    max_tokens: int,
    passo_janela: int,
) -> tuple[float, float, float, int, int, int]:
    """
    F1 de span exato sobre os exemplos. Devolve (P, R, F1, exatos, preditos, esperados).

    Roda em `no_grad` e volta o modelo para `train` ao sair: esquecer o `rede.train()` no fim é
    um defeito silencioso — o dropout fica desligado no resto do treino e a perda cai de forma
    enganosa.
    """
    torch = T._exigir_torch()

    rede.eval()
    exatos = preditos = esperados = 0
    try:
        with torch.no_grad():
            for exemplo in exemplos:
                span = R.localizar(exemplo.nome_contratada, exemplo.texto, classe=CONTRATADA)
                if span is None:
                    continue
                palavras = R.palavras(exemplo.texto)
                verdade = _spans_do_bio(R.rotular_palavras(exemplo.texto, [span]))
                esperados += len(verdade)

                predito = _predizer_bio(
                    rede, tok, exemplo.texto, palavras,
                    dispositivo=dispositivo, max_tokens=max_tokens, passo=passo_janela,
                )
                achados = _spans_do_bio(predito)
                preditos += len(achados)
                exatos += len(achados & verdade)
    finally:
        rede.train()

    precisao = exatos / preditos if preditos else 0.0
    revocacao = exatos / esperados if esperados else 0.0
    f1 = (
        2 * precisao * revocacao / (precisao + revocacao)
        if (precisao + revocacao)
        else 0.0
    )
    return precisao, revocacao, f1, exatos, preditos, esperados


def _predizer_bio(
    rede, tok, texto: str, palavras: list, *, dispositivo: str, max_tokens: int, passo: int
) -> list[str]:
    """
    Os rótulos BIO preditos, uma entrada por palavra, reconstruídos das janelas.

    Quando duas janelas cobrem a mesma palavra, vence a **primeira** que a rotulou como não-`O`.
    A regra é necessária porque as janelas se sobrepõem (o passo é `passo` tokens, menor que
    `max_tokens`), e uma palavra da sobreposição recebe predição de mais de uma janela.
    """
    torch = T._exigir_torch()

    neutro = ["O"] * len(palavras)
    janelas = R.janelas(texto, neutro, tok, max_tokens=max_tokens, passo=passo)
    predito = ["O"] * len(palavras)

    for janela in janelas:
        ids = torch.tensor([janela.ids], device=dispositivo)
        mascara = torch.tensor([janela.mascara], device=dispositivo)
        saida = rede(input_ids=ids, attention_mask=mascara)
        indices = saida.logits[0].argmax(dim=-1).tolist()
        for posicao, indice_palavra in enumerate(janela.ids_palavra):
            if indice_palavra is None or indice_palavra >= len(predito):
                continue
            rotulo = ROTULOS[indices[posicao]] if indices[posicao] < len(ROTULOS) else "O"
            if predito[indice_palavra] == "O" and rotulo != "O":
                predito[indice_palavra] = rotulo
    return predito


# O AJUSTE


def preparar_janelas(
    exemplos: Sequence[T.Exemplo],
    tokenizador,
    *,
    max_tokens: int,
    passo: int,
    max_negativas: int = MAX_NEGATIVAS_POR_EXEMPLO,
) -> tuple[list[R.Janela], dict]:
    """
    As janelas de treino, com a seleção declarada em `MAX_NEGATIVAS_POR_EXEMPLO`.

    Alternativa a `treinar.preparar`, que devolve uma lista **plana** de janelas, sem
    identidade de exemplo: sem saber a que exemplo cada janela pertence, não há como limitar as
    negativas por exemplo. Aqui as janelas são construídas exemplo a exemplo.

    Devolve `(janelas, diagnóstico)`. O diagnóstico traz, entre outros, quantas janelas
    positivas e negativas foram mantidas (`negativas_mantidas`), quantas negativas ficaram
    fora da seleção (`negativas_descartadas`) e o maior número de janelas num exemplo.
    """
    janelas: list[R.Janela] = []
    positivas = negativas = descartadas = sem_span = 0
    maior = 0

    for exemplo in exemplos:
        span = R.localizar(exemplo.nome_contratada, exemplo.texto, classe=CONTRATADA)
        if span is None:
            sem_span += 1
            continue

        rotulos_palavra = R.rotular_palavras(exemplo.texto, [span])
        do_exemplo = R.janelas(
            exemplo.texto, rotulos_palavra, tokenizador,
            max_tokens=max_tokens, passo=passo,
        )
        maior = max(maior, len(do_exemplo))

        # Uma janela é positiva quando algum rótulo dela não é `O` nem IGNORAR.
        com_entidade = [
            j for j in do_exemplo
            if any(r not in (INDICE["O"], IGNORAR) for r in j.rotulos)
        ]
        sem_entidade = [j for j in do_exemplo if j not in com_entidade]

        janelas.extend(com_entidade)
        positivas += len(com_entidade)
        # As primeiras negativas, e não uma amostra aleatória: a seleção é determinística, e as
        # primeiras janelas cobrem o começo do ato, onde se espera a identificação das partes.
        # Negativas do começo são mais informativas que as do fim do documento, que tendem a
        # ser cláusulas sem relação com as partes.
        janelas.extend(sem_entidade[:max_negativas])
        negativas += min(len(sem_entidade), max_negativas)
        descartadas += max(0, len(sem_entidade) - max_negativas)

    diagnostico = {
        "exemplos": len(exemplos),
        "sem_span_localizavel": sem_span,
        "janelas": len(janelas),
        "positivas": positivas,
        "negativas_mantidas": negativas,
        "negativas_descartadas": descartadas,
        "max_janelas_num_exemplo": maior,
        "max_negativas_por_exemplo": max_negativas,
    }
    return janelas, diagnostico


def ajustar(
    treino: Sequence[T.Exemplo],
    validacao: Sequence[T.Exemplo],
    *,
    modelo_base: str = MODELO_LARGE,
    saida: Path,
    semente: int = 42,
    epocas: int = 3,
    lote_fisico: int = 1,
    lote_efetivo: int = LOTE_EFETIVO_PADRAO,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    aquecimento: float = 0.1,
    max_tokens: int = 512,
    passo_janela: int = 128,
    paciencia: int = PACIENCIA_PADRAO,
    gradient_checkpointing: bool = True,
    max_negativas: int = MAX_NEGATIVAS_POR_EXEMPLO,
    retomar: bool = True,
    progresso: Callable[[int, int, float], None] | None = None,
) -> Resultado:
    """
    Treina com validação por época, early stopping e retomada. Grava só o melhor.

    `lote_fisico` é quantas janelas vão ao dispositivo por passo, e `lote_efetivo` é o tamanho
    de lote que o otimizador vê. A razão entre os dois é o número de passos de acumulação.
    Separá-los permite um lote efetivo maior do que a memória do dispositivo comporta de uma
    vez; mudar o lote físico, mantendo o efetivo, muda o uso de memória e não o tamanho do
    lote que o otimizador vê.
    """
    torch = T._exigir_torch()
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if lote_efetivo % lote_fisico:
        raise ValueError(
            f"lote_efetivo {lote_efetivo} não é múltiplo de lote_fisico {lote_fisico}: "
            f"a acumulação ficaria com um passo curto no fim de cada ciclo, e o lote efetivo "
            f"real seria outro que o registrado no metadata"
        )
    acumulacao = lote_efetivo // lote_fisico

    torch.manual_seed(semente)
    random.seed(semente)
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(modelo_base)
    janelas, diagnostico = preparar_janelas(
        treino, tok, max_tokens=max_tokens, passo=passo_janela, max_negativas=max_negativas
    )
    print(f"janelas de treino: {diagnostico}", flush=True)
    if not janelas:
        raise ValueError("nenhuma janela de treino — os spans não localizaram")

    dev = dispositivo_escolhido()
    rede = AutoModelForTokenClassification.from_pretrained(
        modelo_base,
        num_labels=len(ROTULOS),
        id2label=dict(enumerate(ROTULOS)),
        label2id=dict(INDICE),
    ).to(dev)

    if gradient_checkpointing:
        # Recomputa as ativações no `backward` em vez de guardá-las. Troca tempo de execução por
        # uma queda grande de memória de ativações.
        rede.gradient_checkpointing_enable()
        # `use_cache` e checkpointing são incompatíveis: o transformers avisa e desliga o cache
        # sozinho, mas desligar aqui evita o aviso em toda época e deixa a escolha explícita.
        rede.config.use_cache = False

    # `foreach=False` desliga o caminho multi-tensor do AdamW. O padrão agrupa os parâmetros em
    # buffers grandes para atualizar tudo de uma vez, o que é mais rápido e custa um pico de
    # memória do tamanho do maior grupo. Com memória limitada, esse pico pode ser a diferença
    # entre caber e não caber. A troca é de velocidade por memória, e não altera o resultado
    # do treino.
    otimizador = torch.optim.AdamW(
        rede.parameters(), lr=lr, weight_decay=weight_decay, foreach=False
    )
    passos_por_epoca = max(1, len(janelas) // lote_efetivo)
    total_passos = passos_por_epoca * epocas
    agenda = get_linear_schedule_with_warmup(
        otimizador,
        num_warmup_steps=int(total_passos * aquecimento),
        num_training_steps=total_passos,
    )

    historico: list[Epoca] = []
    melhor_f1 = -1.0
    melhor_epoca = 0
    sem_melhora = 0
    primeira_epoca = 0

    caminho_retomada = saida / ARQUIVO_RETOMADA
    if retomar and caminho_retomada.exists():
        estado = torch.load(caminho_retomada, map_location=dev, weights_only=False)
        rede.load_state_dict(estado["rede"])
        otimizador.load_state_dict(estado["otimizador"])
        agenda.load_state_dict(estado["agenda"])
        primeira_epoca = estado["epoca"]
        melhor_f1 = estado["melhor_f1"]
        melhor_epoca = estado["melhor_epoca"]
        historico = [Epoca(**e) for e in estado["historico"]]
        print(f"retomado da época {primeira_epoca} (melhor F1 {melhor_f1:.4f})", flush=True)
        # O `torch.load` acima materializa rede + estado do AdamW no dispositivo de uma vez, por
        # cima do que `from_pretrained` já alocou. Esse pico pode exceder o limite de memória do
        # MPS no início da época retomada, antes de a liberação periódica do laço acontecer.
        # Liberar aqui, logo após
        # carregar o estado, reduz o pico onde ele ocorre.
        _liberar_memoria()

    rede.train()
    passo_global = 0

    for epoca in range(primeira_epoca, epocas):
        inicio = time.time()
        lotes = T.montar_lotes(
            janelas, lote_fisico, tok.pad_token_id, semente + 1000 * (epoca + 1)
        )
        soma = 0.0
        otimizador.zero_grad(set_to_none=True)

        for i, batch in enumerate(lotes, start=1):
            batch = {k: v.to(dev) for k, v in batch.items()}
            resultado = rede(**batch)
            # Divide a perda pela acumulação para que a magnitude do gradiente seja a mesma que
            # a de um lote real desse tamanho. Sem dividir, o passo efetivo seria `acumulacao`
            # vezes maior que o `lr` declarado, e o `lr` do metadata mentiria.
            (resultado.loss / acumulacao).backward()
            soma += float(resultado.loss.detach())

            if i % acumulacao == 0 or i == len(lotes):
                torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
                otimizador.step()
                agenda.step()
                otimizador.zero_grad(set_to_none=True)
                passo_global += 1
                if progresso is not None:
                    progresso(passo_global, total_passos, soma / i)
                # No MPS a memória não volta ao pool sozinha, e o alocador fragmenta ao longo
                # da época; liberar só entre épocas pode não bastar. Por isso a liberação
                # também ocorre a cada `INTERVALO_LIBERAR_MEMORIA` passos de otimizador, que
                # tem custo baixo.
                if passo_global % INTERVALO_LIBERAR_MEMORIA == 0:
                    _liberar_memoria()

        perda = soma / max(len(lotes), 1)
        # Antes da validação, e não só depois da época: a validação faz inferência em janelas e
        # aloca por cima do que o treino ainda tem reservado, e o pico dos dois somados pode
        # chegar perto do limite de memória.
        _liberar_memoria()
        precisao, revocacao, f1, exatos, preditos, esperados = avaliar_f1(
            rede, tok, validacao,
            dispositivo=dev, max_tokens=max_tokens, passo_janela=passo_janela,
        )

        registro = Epoca(
            numero=epoca + 1,
            perda_treino=round(perda, 5),
            precisao=round(precisao, 4),
            revocacao=round(revocacao, 4),
            f1=round(f1, 4),
            exatos=exatos, preditos=preditos, esperados=esperados,
            segundos=round(time.time() - inicio, 1),
            memoria_gib=round(_memoria_gib(), 2),
        )
        historico.append(registro)
        # As duas linhas que o `MetricDefinitions` do SageMaker Training Job captura por
        # expressão regular, transformando o treino numa série no CloudWatch.
        print(f"perda_treino={registro.perda_treino:.5f}", flush=True)
        print(f"entity_f1={registro.f1:.4f}", flush=True)
        print(
            f"época {registro.numero}/{epocas}  perda {registro.perda_treino:.4f}  "
            f"P {precisao:.4f}  R {revocacao:.4f}  F1 {f1:.4f}  "
            f"{registro.segundos:.0f}s  {registro.memoria_gib:.1f} GiB",
            flush=True,
        )

        if f1 > melhor_f1:
            melhor_f1, melhor_epoca, sem_melhora = f1, epoca + 1, 0
            # `save_best_only`: grava agora, sobrescrevendo. O último checkpoint no disco é
            # sempre o melhor, e não o mais recente — o que evita publicar um modelo que
            # passou do ponto.
            rede.save_pretrained(saida)
            tok.save_pretrained(saida)
            print(f"  melhor até agora — checkpoint gravado em {saida}", flush=True)
        else:
            sem_melhora += 1

        torch.save({
            "rede": rede.state_dict(),
            "otimizador": otimizador.state_dict(),
            "agenda": agenda.state_dict(),
            "epoca": epoca + 1,
            "melhor_f1": melhor_f1,
            "melhor_epoca": melhor_epoca,
            "historico": [e.como_dicionario() for e in historico],
        }, caminho_retomada)
        _liberar_memoria()

        if sem_melhora >= paciencia:
            print(
                f"early stopping: {sem_melhora} épocas sem melhora "
                f"(paciência {paciencia})", flush=True
            )
            break

    parou_cedo = sem_melhora >= paciencia
    metadados = {
        "modelo_base": modelo_base,
        "rotulos": list(ROTULOS),
        "id2label": dict(enumerate(ROTULOS)),
        "label2id": dict(INDICE),
        "ignorar": IGNORAR,
        "semente": semente,
        "dispositivo": dev,
        "hiperparametros": {
            "epocas_pedidas": epocas,
            "epocas_executadas": len(historico),
            "lote_fisico": lote_fisico,
            "lote_efetivo": lote_efetivo,
            "acumulacao": acumulacao,
            "lr": lr,
            "weight_decay": weight_decay,
            "aquecimento": aquecimento,
            "max_tokens": max_tokens,
            "passo_janela": passo_janela,
            "paciencia": paciencia,
            "gradient_checkpointing": gradient_checkpointing,
            "max_negativas_por_exemplo": max_negativas,
        },
        "corpus": {
            "exemplos_treino": len(treino),
            "exemplos_validacao": len(validacao),
            "janelas_treino": len(janelas),
            "diagnostico_janelas": diagnostico,
        },
        "melhor_epoca": melhor_epoca,
        "melhor_entity_f1": round(melhor_f1, 4),
        "parou_por_paciencia": parou_cedo,
        "historico": [e.como_dicionario() for e in historico],
        "versoes": T._versoes(modelo_base),
    }
    (saida / T.ARQUIVO_METADADOS).write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (saida / ARQUIVO_HISTORICO).write_text(
        json.dumps([e.como_dicionario() for e in historico], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # O arquivo de retomada existe para sobreviver a uma **interrupção**. Se a função chegou
    # aqui, o treino terminou — por épocas esgotadas ou por early stopping — e o estado do
    # otimizador não serve mais. Ele guarda pesos, momento e variância do AdamW, bem maior
    # que os pesos sozinhos.
    #
    # A remoção cobre os dois modos de término: épocas esgotadas e early stopping.
    if caminho_retomada.exists():
        caminho_retomada.unlink()

    return Resultado(
        diretorio=saida,
        melhor_epoca=melhor_epoca,
        melhor_f1=melhor_f1,
        historico=historico,
        parou_por_paciencia=parou_cedo,
        metadados=metadados,
    )


def _memoria_gib() -> float:
    """O pico de memória residente do processo (`ru_maxrss`), em GiB. Zero se não medível."""
    try:
        import resource
        # `ru_maxrss` é em bytes no macOS e em KiB no Linux. O fator é escolhido pelo sistema.
        bruto = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys
        return bruto / 2**30 if sys.platform == "darwin" else bruto / 2**20
    except Exception:
        return 0.0


def _liberar_memoria() -> None:
    """
    Devolve a memória do backend ao sistema (coleta de lixo e `empty_cache`).

    No MPS a memória não é liberada como numa GPU discreta, e sem isto o uso pode crescer de
    forma contínua até `MPS backend out of memory`.
    """
    torch = T._exigir_torch()
    import gc

    gc.collect()
    # `torch.mps` existe como módulo mesmo numa máquina sem MPS (por exemplo a GPU CUDA do
    # SageMaker), e chamar `empty_cache()` ali lança `RuntimeError: Cannot execute
    # emptyCache() without MPS backend`. A guarda tem de ser por DISPONIBILIDADE do backend,
    # não por presença do atributo.
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        mps.empty_cache()
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.empty_cache()
