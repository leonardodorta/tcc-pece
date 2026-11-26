"""
lancar_treino.py — configura e dispara o SageMaker Training Job de ajuste fino.

Uso:
    # só sobe os dados e imprime os URIs (não dispara treino):
    python infraestrutura/scripts/lancar_treino.py --subir

    # dispara o treino (assíncrono) e imprime o nome do job:
    python infraestrutura/scripts/lancar_treino.py --treinar

    # baixa o model.tar.gz de um job concluído:
    python infraestrutura/scripts/lancar_treino.py --baixar <nome-do-job>

Lê o profile e a região do ambiente (`AWS_PROFILE`, `AWS_REGION`); as credenciais não ficam no
código. A imagem é o Deep Learning Container de PyTorch do SageMaker; o container instala
`src/requirements.txt` (transformers) por cima. O entry point é `src/entrada_treino.py`.

Pastas locais (fora do repositório): `--subir` e `--treinar` leem `dados/gabarito/train.jsonl` e
`dados/gabarito/validation.jsonl`; `--baixar` grava o `model.tar.gz` em `modelos/sagemaker/`.
Requer as variáveis `DOMSP_BUCKET` e `DOMSP_SAGEMAKER_ROLE` (ver abaixo).
"""
from __future__ import annotations

import argparse
import os
import pathlib

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

RAIZ = pathlib.Path(__file__).resolve().parents[2]
GABARITO = RAIZ / "dados" / "gabarito"

# Bucket e papel de execução vêm de variáveis de ambiente, não do código. Exemplo:
#   export DOMSP_BUCKET=domsp-prod-lake-<conta>-us-east-1
#   export DOMSP_SAGEMAKER_ROLE=arn:aws:iam::<conta>:role/<papel-de-execucao>
BUCKET = os.environ.get("DOMSP_BUCKET")
ROLE = os.environ.get("DOMSP_SAGEMAKER_ROLE")
REGIAO = os.environ.get("AWS_REGION", "us-east-1")
PREFIXO = "sagemaker/treino"
INSTANCIA = os.environ.get("DOMSP_INSTANCIA_TREINO", "ml.g5.xlarge")
VERSAO_PYTORCH = "2.5.1"
VERSAO_PY = "py311"


def _exigir_ambiente() -> None:
    """Falha cedo e claro se a conta/bucket/papel não vierem do ambiente."""
    faltando = [n for n, v in (("DOMSP_BUCKET", BUCKET),
                               ("DOMSP_SAGEMAKER_ROLE", ROLE)) if not v]
    if faltando:
        raise SystemExit(
            "defina as variáveis de ambiente " + ", ".join(faltando) + " antes de lançar o "
            "treino. Elas informam o bucket e o papel de execução (ARN) e não são lidas "
            "de nenhum arquivo do repositório."
        )


def _sessao() -> sagemaker.Session:
    _exigir_ambiente()
    bs = boto3.Session(region_name=REGIAO)
    return sagemaker.Session(boto_session=bs, default_bucket=BUCKET)


def subir(sess: sagemaker.Session) -> dict[str, str]:
    """Sobe train.jsonl e validation.jsonl para o S3. Retorna os URIs dos canais."""
    uris = {}
    for canal, arquivo in [("train", "train.jsonl"), ("validation", "validation.jsonl")]:
        local = GABARITO / arquivo
        if not local.exists():
            raise SystemExit(f"não encontrei {local}")
        destino = f"s3://{BUCKET}/{PREFIXO}/entrada/{canal}"
        uri = sess.upload_data(str(local), bucket=BUCKET, key_prefix=f"{PREFIXO}/entrada/{canal}")
        assert uri.startswith(destino), f"upload foi para {uri}, esperado sob {destino}"
        uris[canal] = uri
        print(f"  {canal:12} {local.stat().st_size/1024:.1f} KB -> {uri}", flush=True)
    return uris


def estimador(sess: sagemaker.Session) -> PyTorch:
    return PyTorch(
        entry_point="entrada_treino.py",
        source_dir=str(RAIZ / "src"),
        role=ROLE,
        instance_type=INSTANCIA,
        instance_count=1,
        framework_version=VERSAO_PYTORCH,
        py_version=VERSAO_PY,
        sagemaker_session=sess,
        base_job_name="domsp-contratada",
        output_path=f"s3://{BUCKET}/{PREFIXO}/saida",
        # Hiperparâmetros do treino; viram flags para o entry point.
        hyperparameters={
            "epocas": 3,
            "lote-fisico": 1,
            "lote-efetivo": 8,
            "lr": 3e-5,
            "weight-decay": 0.01,
            "aquecimento": 0.1,
            "max-tokens": 512,
            "passo-janela": 128,
            "semente": 42,
            "paciencia": 2,
            "gradient-checkpointing": 1,
            # Número máximo de janelas negativas por exemplo (padrão de `ajustar`: 2).
            "max-negativas": 4,
        },
        # As duas séries que o entry point imprime, capturadas para o CloudWatch.
        metric_definitions=[
            {"Name": "perda_treino", "Regex": r"perda_treino=([0-9\.]+)"},
            {"Name": "entity_f1", "Regex": r"entity_f1=([0-9\.]+)"},
        ],
        volume_size=50,
        max_run=4 * 60 * 60,
        environment={"HF_HOME": "/opt/ml/input/hf", "TOKENIZERS_PARALLELISM": "false"},
    )


def treinar(sess: sagemaker.Session, uris: dict[str, str]) -> str:
    est = estimador(sess)
    entradas = {
        "train": TrainingInput(uris["train"], content_type="application/jsonl"),
        "validation": TrainingInput(uris["validation"], content_type="application/jsonl"),
    }
    # wait=False: dispara e volta, para acompanhar por fora.
    est.fit(entradas, wait=False)
    nome = est.latest_training_job.name
    print(f"\nTraining Job disparado: {nome}", flush=True)
    return nome


def baixar(sess: sagemaker.Session, nome_job: str, destino: pathlib.Path) -> pathlib.Path:
    cli = sess.boto_session.client("sagemaker")
    desc = cli.describe_training_job(TrainingJobName=nome_job)
    status = desc["TrainingJobStatus"]
    print(f"status do job: {status}", flush=True)
    if status != "Completed":
        raise SystemExit(f"job não concluído (status {status}); não há artefato para baixar")
    artefato = desc["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"artefato: {artefato}", flush=True)
    destino.mkdir(parents=True, exist_ok=True)
    local = destino / "model.tar.gz"
    bucket = artefato.split("/", 3)[2]
    chave = artefato.split("/", 3)[3]
    sess.boto_session.client("s3").download_file(bucket, chave, str(local))
    print(f"baixado para {local} ({local.stat().st_size/2**20:.1f} MiB)", flush=True)
    return local


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subir", action="store_true", help="só sobe os dados")
    p.add_argument("--treinar", action="store_true", help="sobe os dados e dispara o treino")
    p.add_argument("--baixar", metavar="NOME_JOB", help="baixa o model.tar.gz do job")
    args = p.parse_args(argv)

    sess = _sessao()
    print(f"região {sess.boto_region_name}  bucket {BUCKET}", flush=True)

    if args.baixar:
        baixar(sess, args.baixar, RAIZ / "modelos" / "sagemaker")
        return

    if args.subir or args.treinar:
        print("subindo dados de treino/validação...", flush=True)
        uris = subir(sess)
        if args.treinar:
            nome = treinar(sess, uris)
            print(f"\npara acompanhar:\n  aws sagemaker describe-training-job "
                  f"--training-job-name {nome} --region {REGIAO} "
                  f"--query TrainingJobStatus --profile $AWS_PROFILE", flush=True)
        return

    p.print_help()


if __name__ == "__main__":
    main()
