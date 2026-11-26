"""
entrada_treino.py — entry point do SageMaker Training Job para o ajuste fino.

Chama `ajustar.ajustar`, que faz validação por época, early stopping por `entity_f1`
e `save_best_only`. Difere de `treinar.treinar`, que treina um número fixo de épocas,
sem validação, e grava o modelo da última.

## Como o SageMaker invoca

O PyTorch estimator roda:

    python entrada_treino.py --epocas 3 --lr 3e-5 ...   (hiperparâmetros como flags)

com o `source_dir` (a raiz `src/`, que contém o pacote `domsp/`) no `PYTHONPATH`, então
`import domsp.*` resolve. Os canais viram variáveis de ambiente:

    SM_CHANNEL_TRAIN        diretório com train.jsonl
    SM_CHANNEL_VALIDATION   diretório com validation.jsonl
    SM_MODEL_DIR            onde gravar o checkpoint (vira o model.tar.gz)
    SM_OUTPUT_DATA_DIR      artefatos extras (metadata, histórico)

As duas linhas `perda_treino=` e `entity_f1=` que `ajustar` imprime por época são
capturadas pelo `MetricDefinitions` do estimator e viram séries no CloudWatch.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import sys

# O container roda este arquivo como script solto; `source_dir=src` põe a raiz do pacote
# no PYTHONPATH, então o import absoluto de `domsp` resolve.
import domsp
from domsp import modelo as M
from domsp.modelo import ajustar
from domsp.modelo import treinar as T


def _canal(nome_env: str, arquivo_padrao: str) -> pathlib.Path:
    """
    Resolve um canal do SageMaker para o arquivo JSONL dentro dele.

    O canal é um diretório. Usa `arquivo_padrao` se ele existir; senão, o primeiro
    arquivo de dados do diretório, porque escritores de dados particionados (como o
    Spark) gravam nomes do tipo `part-*.json`.
    """
    base = os.environ.get(nome_env)
    if not base:
        raise SystemExit(f"canal {nome_env} não definido — o estimator precisa passá-lo")
    diretorio = pathlib.Path(base)
    candidato = diretorio / arquivo_padrao
    if candidato.exists():
        return candidato
    # senão, o primeiro arquivo não-marcador do diretório
    for a in sorted(diretorio.rglob("*")):
        if a.is_file() and not a.name.startswith((".", "_")) and not a.name.endswith(".crc"):
            return a
    raise SystemExit(f"nenhum arquivo de dados em {diretorio} (canal {nome_env})")


def _argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino do ajuste fino no SageMaker")
    # Os defaults são os hiperparâmetros padrão; o estimator os sobrescreve por flag.
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lote-fisico", type=int, default=1)
    p.add_argument("--lote-efetivo", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--aquecimento", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--passo-janela", type=int, default=128)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--paciencia", type=int, default=2)
    p.add_argument(
        "--max-negativas",
        type=int,
        default=ajustar.MAX_NEGATIVAS_POR_EXEMPLO,
        help=(
            "janelas SEM entidade mantidas por exemplo. Mais negativas ensinam o modelo a "
            "RECUSAR, ao custo de mais janelas por época e mais memória e tempo de treino"
        ),
    )
    p.add_argument("--gradient-checkpointing", type=int, default=1,
                   help="1 liga (padrão), 0 desliga; ligado reduz a memória de ativações "
                        "ao custo de mais tempo por passo")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _argumentos(argv)

    print(f"domsp {domsp.__version__}", flush=True)
    print(f"python {platform.python_version()}  {platform.machine()}", flush=True)

    import torch
    dispositivo = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"torch {torch.__version__}  dispositivo={dispositivo}  "
          f"cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB", flush=True)

    # Os dois canais, com os nomes de arquivo esperados em cada um.
    treino = T.exemplos_de_jsonl(_canal("SM_CHANNEL_TRAIN", "train.jsonl"))
    validacao = T.exemplos_de_jsonl(_canal("SM_CHANNEL_VALIDATION", "validation.jsonl"))
    print(f"treino .... {len(treino):>6,} exemplos", flush=True)
    print(f"validação . {len(validacao):>6,} exemplos", flush=True)
    if not treino or not validacao:
        raise SystemExit("canal vazio — treino ou validação sem exemplos")

    # Guarda contra trocar o Large silenciosamente.
    assert M.eh_bertimbau(M.MODELO_LARGE), "fora da família BERTimbau"
    assert M.tamanho(M.MODELO_LARGE) == "large", "o treino exige o BERTimbau Large"

    saida = pathlib.Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    saida.mkdir(parents=True, exist_ok=True)

    def progresso(passo: int, total: int, perda: float) -> None:
        print(f"    passo {passo}/{total}  perda média {perda:.4f}", flush=True)

    resultado = ajustar.ajustar(
        treino,
        validacao,
        saida=saida,
        modelo_base=M.MODELO_LARGE,
        epocas=args.epocas,
        lote_fisico=args.lote_fisico,
        lote_efetivo=args.lote_efetivo,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aquecimento=args.aquecimento,
        max_tokens=args.max_tokens,
        passo_janela=args.passo_janela,
        semente=args.semente,
        paciencia=args.paciencia,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        max_negativas=args.max_negativas,
        # No SageMaker cada job começa limpo; não há checkpoint anterior para retomar.
        retomar=False,
        progresso=progresso,
    )

    print(f"\nmelhor época ......... {resultado.melhor_epoca}", flush=True)
    print(f"melhor entity_f1 ..... {resultado.melhor_f1:.4f}", flush=True)
    print(f"parou por paciência .. {resultado.parou_por_paciencia}", flush=True)
    print(f"checkpoint ........... {resultado.diretorio}", flush=True)

    # ── Proveniência: amarra ESTE checkpoint à execução e ao insumo ─────────
    # O `ajustar.ajustar` já gravou um `metadata.json` básico junto ao checkpoint; aqui ele é
    # relido e completado com as três âncoras de proveniência (job do SageMaker, sha256
    # dos pesos, hash do gabarito), além de `job_name` no topo e `corpus.exemplos_treino` —
    # o formato que `consolidar_execucoes` consome.
    job_name = (
        os.environ.get("SM_TRAINING_JOB_NAME")
        or os.environ.get("TRAINING_JOB_NAME")
        or os.environ.get("DOMSP_JOB_NAME")
        or f"local-{saida.name}"
    )
    pesos = saida / "model.safetensors"
    sha_pesos = T._sha256_arquivo(pesos) if pesos.exists() else None
    sha_gabarito = T._sha256_exemplos(treino)

    meta_checkpoint = saida / "metadata.json"
    meta = {}
    if meta_checkpoint.exists():
        meta = json.loads(meta_checkpoint.read_text(encoding="utf-8"))
    meta.update({
        "job_name": job_name,
        "sha256_pesos": sha_pesos,
        "gabarito": {"hash": sha_gabarito, "exemplos_treino": len(treino)},
        "corpus": {"exemplos_treino": len(treino)},
        "semente": args.semente,
        "melhor_epoca": resultado.melhor_epoca,
        "melhor_entity_f1": resultado.melhor_f1,
    })
    # garante que os hiperparâmetros carregam a semente (o consolidador os compara)
    meta.setdefault("hiperparametros", {})
    if isinstance(meta["hiperparametros"], dict):
        meta["hiperparametros"].setdefault("semente", args.semente)
    meta_checkpoint.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"proveniência gravada: job={job_name} sha_pesos={(sha_pesos or '')[:12]} "
          f"sha_gabarito={sha_gabarito[:12]}", flush=True)

    # Grava a parte de treino do metadata junto ao checkpoint. A avaliação, feita depois e
    # fora deste arquivo, acrescenta a parte da métrica de teste.
    registro = {
        "gerado_em": datetime.datetime.now(datetime.UTC).isoformat(),
        "modo": "experimento",
        "plataforma": "sagemaker-training-job",
        "versao_regras": domsp.VERSAO_REGRAS,
        "dispositivo": dispositivo,
        "hiperparametros": {
            "modelo_base": M.MODELO_LARGE,
            "epocas": args.epocas,
            "lote_fisico": args.lote_fisico,
            "lote_efetivo": args.lote_efetivo,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "aquecimento": args.aquecimento,
            "max_tokens": args.max_tokens,
            "passo_janela": args.passo_janela,
            "semente": args.semente,
            "paciencia": args.paciencia,
            "max_negativas": args.max_negativas,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        },
        "melhor_epoca": resultado.melhor_epoca,
        "melhor_entity_f1_validacao": resultado.melhor_f1,
        "parou_por_paciencia": resultado.parou_por_paciencia,
        "historico": [e.como_dicionario() for e in resultado.historico],
    }
    (saida / "metadata_treino.json").write_text(
        json.dumps(registro, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # Também no diretório de saída de dados, para inspeção sem baixar o model.tar.gz.
    extra = os.environ.get("SM_OUTPUT_DATA_DIR")
    if extra:
        pathlib.Path(extra, "metadata_treino.json").write_text(
            json.dumps(registro, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    print("\ntreino concluído; checkpoint gravado em SM_MODEL_DIR", flush=True)


if __name__ == "__main__":
    sys.exit(main())
