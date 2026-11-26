"""
empacotar — monta o `model.tar.gz` que o SageMaker Batch Transform consome.

Arquivos no pacote:

    config.json              a arquitetura E o id2label — sem ele o serving devolve LABEL_0
    model.safetensors        os pesos
    tokenizer.json           o tokenizador rápido
    tokenizer_config.json    a configuração dele
    vocab.txt                o vocabulário
    special_tokens_map.json  [CLS], [SEP], [PAD]
    metadata.json            a procedência do modelo (checkpoint base e versões de
                             bibliotecas, intervalo e hash do corpus, gabarito,
                             hiperparâmetros, semente, dados do treino, avaliação,
                             classes e versão das regras)
    code/inference.py        os quatro ganchos do SageMaker
    code/requirements.txt    o que o contêiner instala

`inference.py` recusa carregar um checkpoint cujo `id2label` não corresponda às classes
esperadas: um checkpoint sem cabeça de NER carrega sem erro e produz `LABEL_0`/`LABEL_1`,
e o pipeline seguiria devolvendo zero entidades como se o modelo não achasse nada.

O `.tar.gz` é montado à mão (não com `shutil.make_archive`) porque o SageMaker exige os
arquivos na raiz do tar; `make_archive` de um diretório produziria `./meu_modelo/config.json`
e o contêiner falharia com "model artifact not found".
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

from .. import VERSAO_REGRAS
from . import ROTULOS

NOME_PACOTE = "model.tar.gz"
ARQUIVO_METADADOS = "metadata.json"

# Arquivos do checkpoint que o contêiner precisa achar na raiz do tar. `empacotar()` grava
# `metadata.json` à parte, e `conferir()` o exige além destes.
OBRIGATORIOS: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)

# O que nunca entra no pacote. `retomada.pt` guarda o estado do otimizador (AdamW) para
# retomar o treino e é muito maior que os pesos; não serve para inferência e inflaria o
# `model.tar.gz`.
EXCLUIDOS: tuple[str, ...] = ("retomada.pt", "historico.json", "training_args.bin")

# O que o contêiner instala. O teto (`<5`) impede que uma versão major futura do
# `transformers`, com API diferente, seja instalada; o piso é a versão mínima aceita.
#
# `torch` não está aqui: a imagem de inferência é uma DLC `huggingface-pytorch-inference`,
# que já traz torch e o toolkit de serving compilados juntos; pedir outra versão faria o
# pip rebaixar o torch da imagem no start do contêiner. A versão do torch usada no treino
# fica registrada em `metadata.json` (`origem.versoes`).
#
# O `domsp` entra como wheel embutido no próprio pacote, para que a inferência da nuvem e
# a da avaliação sejam o MESMO código (ver `inferencia_sagemaker.py`). O
# `requirements.txt` referencia o wheel por caminho ABSOLUTO (`/opt/ml/model/code/…`)
# porque o SageMaker roda `pip install` com o diretório de trabalho na raiz, não em
# `code/` — ver o comentário em `empacotar()`, onde `requisitos` é montado.
REQUISITOS_BASE = """\
transformers>=4.46,<5
safetensors>=0.4.5
"""

#: Alias de `REQUISITOS_BASE`: as dependências sem a linha do wheel, que `empacotar()`
#: acrescenta.
REQUISITOS = REQUISITOS_BASE


@dataclass(frozen=True)
class Pacote:
    """O resultado do empacotamento."""

    caminho: Path
    bytes: int
    sha256: str
    arquivos: tuple[str, ...]
    metadados: dict


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def montar_metadados(
    checkpoint: Path,
    *,
    manifesto_corpus: dict,
    manifesto_gabarito: dict,
    metricas_modelo: dict,
    avaliacao: dict,
) -> dict:
    """
    Monta o `metadata.json` do pacote a partir de artefatos já existentes (checkpoint,
    manifestos de corpus e gabarito, métricas e avaliação), em vez de receber valores
    soltos — evita que um campo digitado à mão divirja do artefato de origem.
    """
    do_treino = json.loads((checkpoint / ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    return {
        "gerado_em": dt.datetime.now(dt.UTC).isoformat(),
        "versao_regras": VERSAO_REGRAS,
        "classes": list(ROTULOS),
        "id2label": do_treino.get("id2label"),
        "label2id": do_treino.get("label2id"),
        "origem": {
            "checkpoint_base": do_treino.get("modelo_base"),
            "versoes": do_treino.get("versoes"),
        },
        "corpus": {
            "intervalo": manifesto_corpus.get("intervalo"),
            "meses": manifesto_corpus.get("meses_do_experimento"),
            "edicoes": manifesto_corpus.get("edicoes"),
            # Hash do corpus: SHA-256 da lista ordenada de hashes de arquivo. Muda se
            # qualquer edição mudar, o que torna "treinado sobre exatamente este corpus"
            # verificável.
            "hash": hashlib.sha256(
                "".join(
                    sorted(a["sha256"] for a in manifesto_corpus.get("arquivos", []))
                ).encode("utf-8")
            ).hexdigest(),
        },
        "gabarito": {
            "divisao": manifesto_gabarito.get("divisao"),
            "conjuntos": {
                nome: dados.get("escritos")
                for nome, dados in (manifesto_gabarito.get("conjuntos") or {}).items()
            },
            "teto": manifesto_gabarito.get("teto", {}).get("limpo_realizavel"),
        },
        "hiperparametros": do_treino.get("hiperparametros"),
        "semente": do_treino.get("semente"),
        "treino": {
            "melhor_epoca": do_treino.get("melhor_epoca"),
            "melhor_entity_f1_validacao": do_treino.get("melhor_entity_f1"),
            "duracao_minutos": metricas_modelo.get("duracao_minutos"),
            "dispositivo": do_treino.get("dispositivo"),
            "historico": do_treino.get("historico"),
        },
        "avaliacao": avaliacao,
    }


def _wheel_do_pacote(explicito: Path | None = None) -> Path | None:
    """
    O wheel do `domsp` a embutir em `code/`.

    Usa o caminho passado explicitamente, se houver (e devolve None se o arquivo não
    existir). Senão, procura em `dist/` na raiz do repositório, onde `python -m build
    --wheel` grava, e devolve o wheel mais recente de lá.

    Não procura em diretórios temporários: o wheel mais recente de lá poderia ser de outra
    construção. O wheel tem de vir de `dist/` ou ser passado explicitamente.
    """
    if explicito is not None:
        return explicito if explicito.is_file() else None
    raiz = Path(__file__).resolve().parents[3]
    candidatos = [c for c in raiz.glob("dist/domsp-*.whl") if c.is_file()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda c: c.stat().st_mtime)


def empacotar(
    checkpoint: Path,
    destino: Path,
    *,
    metadados: dict,
    codigo_inferencia: Path | None = None,
    wheel: Path | None = None,
) -> Pacote:
    """
    Monta o `model.tar.gz` a partir do checkpoint.

    Levanta `FileNotFoundError` se faltar algum arquivo de `OBRIGATORIOS` ou o wheel do
    `domsp`, e `ValueError` se o `id2label` do checkpoint não for `ROTULOS`. As checagens
    do checkpoint vêm antes da montagem: um tar parcial no S3 só seria percebido depois
    que o Batch Transform o baixasse e provisionasse a instância.
    """
    checkpoint = Path(checkpoint)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    faltando = [nome for nome in OBRIGATORIOS if not (checkpoint / nome).exists()]
    if faltando:
        raise FileNotFoundError(
            f"o checkpoint {checkpoint} não tem {faltando}. "
            f"Sem esses arquivos o contêiner falha com 'model artifact not found' "
            f"depois de provisionar a instância"
        )

    id2label = {
        int(k): v
        for k, v in (
            json.loads((checkpoint / "config.json").read_text(encoding="utf-8")).get(
                "id2label"
            )
            or {}
        ).items()
    }
    esperado = tuple(id2label[i] for i in sorted(id2label))
    if esperado != ROTULOS:
        raise ValueError(
            f"o id2label do checkpoint é {esperado} e o esperado é {ROTULOS}. "
            f"Empacotar assim produziria um modelo que devolve rótulo errado sem erro"
        )

    # Área de montagem, para que o tar tenha exatamente o que se quer e nada mais.
    montagem = destino / "_montagem"
    if montagem.exists():
        shutil.rmtree(montagem)
    (montagem / "code").mkdir(parents=True)

    incluidos: list[str] = []
    for arquivo in sorted(checkpoint.iterdir()):
        if not arquivo.is_file() or arquivo.name in EXCLUIDOS:
            continue
        shutil.copy2(arquivo, montagem / arquivo.name)
        incluidos.append(arquivo.name)

    (montagem / ARQUIVO_METADADOS).write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if ARQUIVO_METADADOS not in incluidos:
        incluidos.append(ARQUIVO_METADADOS)

    origem_codigo = codigo_inferencia or (Path(__file__).parent / "inferencia_sagemaker.py")
    shutil.copy2(origem_codigo, montagem / "code" / "inference.py")

    # O wheel do próprio pacote vai dentro de `code/`, referenciado em `requirements.txt`
    # por caminho ABSOLUTO no contêiner. É o que faz o contêiner rodar o mesmo algoritmo
    # da avaliação. Sem o wheel, o empacotamento é recusado.
    requisitos = REQUISITOS_BASE
    roda = _wheel_do_pacote(wheel)
    if roda is not None:
        shutil.copy2(roda, montagem / "code" / roda.name)
        # Caminho ABSOLUTO no contêiner: o SageMaker roda
        # `pip install -r /opt/ml/model/code/requirements.txt` com o diretório de
        # trabalho na raiz (`/`), não em `code/`: um caminho relativo resolveria para
        # `/domsp-...whl`, que não existe.
        requisitos = f"{REQUISITOS_BASE}/opt/ml/model/code/{roda.name}\n"
        incluidos.append(f"code/{roda.name}")
    else:
        raise FileNotFoundError(
            "não encontrei o wheel do domsp para embutir no pacote. "
            "Rode `python -m build --wheel` antes de empacotar, ou passe `wheel=`. "
            "Sem o wheel, code/inference.py não importa domsp.modelo.servir e a "
            "inferência da nuvem deixaria de ser o mesmo código da avaliação"
        )

    (montagem / "code" / "requirements.txt").write_text(requisitos, encoding="utf-8")
    incluidos += ["code/inference.py", "code/requirements.txt"]

    caminho = destino / NOME_PACOTE
    with tarfile.open(caminho, "w:gz") as tar:
        for item in sorted(montagem.rglob("*")):
            if item.is_file():
                # `arcname` relativo à montagem: põe os arquivos na raiz do tar, como o
                # SageMaker exige.
                tar.add(item, arcname=str(item.relative_to(montagem)))

    shutil.rmtree(montagem)
    return Pacote(
        caminho=caminho,
        bytes=caminho.stat().st_size,
        sha256=_sha256(caminho),
        arquivos=tuple(sorted(incluidos)),
        metadados=metadados,
    )


def conferir(pacote: Path) -> dict:
    """Abre o `model.tar.gz` e confere o que ele contém, sem extrair."""
    pacote = Path(pacote)
    with tarfile.open(pacote, "r:gz") as tar:
        nomes = sorted(m.name for m in tar.getmembers() if m.isfile())
        metadados_brutos = tar.extractfile(ARQUIVO_METADADOS)
        metadados = json.loads(metadados_brutos.read()) if metadados_brutos else {}

    faltando = [n for n in (*OBRIGATORIOS, ARQUIVO_METADADOS) if n not in nomes]
    aninhados = [n for n in nomes if "/" in n and not n.startswith("code/")]
    return {
        "arquivos": nomes,
        "faltando": faltando,
        "tem_code_inference": "code/inference.py" in nomes,
        "tem_code_requirements": "code/requirements.txt" in nomes,
        "aninhados_indevidos": aninhados,
        "excluidos_presentes": [n for n in EXCLUIDOS if n in nomes],
        "classes": metadados.get("classes"),
        "versao_regras": metadados.get("versao_regras"),
        "bytes": pacote.stat().st_size,
        "valido": not faltando
        and not aninhados
        and "code/inference.py" in nomes
        and metadados.get("classes") == list(ROTULOS),
    }
