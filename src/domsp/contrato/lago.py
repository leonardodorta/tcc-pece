"""
lago — onde cada coisa mora no S3, e qual tabela a lê.

A inferência não escreve no caminho que lê.

## Por que a separação

Se um estágio lesse `silver/contract/` e regravasse `silver/contract/` com
`mode("overwrite")`, uma falha entre o `read` e o `write` deixaria a prata parcialmente
reescrita, sem forma de saber quais partições estavam completas. E como o Spark é
preguiçoso, o `read` só acontece durante o `write`: o overwrite pode apagar a entrada antes
de tê-la lido inteira.

São quatro caminhos e um sentido só:

    silver/contract/              a prata determinística, escrita pelo silver e nunca reescrita
    silver/inference-raw/         o que o modelo devolveu, com escore, sem validar
    silver/inference-accepted/    o que passou pela porta de validação
    silver/contract-final/        a consolidação: determinístico + inferência aceita

Cada estágio lê de um lugar e escreve em outro. Nenhum reescreve a própria entrada, e
qualquer estágio pode ser reexecutado sozinho sem risco para o que veio antes.

`silver/inference-raw/` guardar o que o modelo disse antes da validação é a única forma de
medir quantos spans a porta recusou. Sem esse caminho, um modelo que degradasse e passasse a
produzir lixo apareceria só como queda de cobertura, e não como aumento de recusa — e os dois
pedem correções diferentes.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

# As camadas

BRONZE = "bronze"
SILVER = "silver"
GOLD = "gold"

CAMADAS: tuple[str, ...] = (BRONZE, SILVER, GOLD)

# A coluna de partição. Uma só, e é a data da EDIÇÃO — não a da assinatura.
#
# A data de assinatura pode faltar e pode ser anterior em anos à
# publicação (aditamento de contrato velho). Particionar por ela colocaria o ato de 2019 numa
# partição de 2019 que o pipeline nunca mais visita, e o reprocessamento de uma edição
# deixaria de ser "reescreva a partição desta data".
PARTICAO = "dt"


@dataclass(frozen=True)
class Conjunto:
    """
    Um conjunto de dados: camada, nome, e o que ele responde.

    `descricao` vai para o `Comment` da tabela no Glue Data Catalog, e daí para a interface
    do Athena, que é onde alguém que nunca leu este repositório vai encontrá-la.

    `nome` usa hífen quando o prefixo S3 usa hífen (`inference-raw`), e `tabela` troca por
    sublinhado, porque o Athena não aceita hífen em identificador sem aspas.
    """

    camada: str
    nome: str
    descricao: str
    particionado: bool = True

    @property
    def prefixo(self) -> str:
        """O prefixo relativo à raiz do lago, sem barra em nenhuma ponta."""
        return f"{self.camada}/{self.nome}"

    @property
    def tabela(self) -> str:
        """O nome da tabela no Glue Data Catalog: `camada_nome`, com sublinhado."""
        return f"{self.camada}_{self.nome}".replace("-", "_")

    def caminho(self, raiz: str) -> str:
        """
        A URI S3 completa, **com** barra final.

        A barra final é obrigatória: `LOCATION` de tabela externa sem barra faz
        `s3://balde/silver/contract` casar também com `s3://balde/silver/contract-final/`,
        e o Athena leria os dois como uma tabela.
        """
        return f"{raiz.rstrip('/')}/{self.prefixo}/"


# ── bronze ─────────────────────────────────────────────────────────────────

PUBLICACAO = Conjunto(
    BRONZE, "publication",
    "Publicação do DOM-SP como a fonte devolveu, uma linha por publicação, HTML intocado. "
    "Imutável por força do art. 12 e § 2º do Decreto 62.177/2023.",
)

# ── prata ──────────────────────────────────────────────────────────────────

CONTRATO = Conjunto(
    SILVER, "contract",
    "Ato de contratação extraído pelas camadas determinísticas, uma linha por ato, "
    "tipado, com proveniência por campo. Nunca reescrito pelo modelo.",
)
DESCARTE = Conjunto(
    SILVER, "discard",
    "Publicação recusada na triagem, com o motivo e o estágio da decisão. Existe para que "
    "o descarte seja auditável: toda publicação recusada fica registrada e não desaparece em silêncio.",
)
RESIDUO = Conjunto(
    SILVER, "residual",
    "Ato cuja parte contratada as camadas determinísticas não resolveram. "
    "É a entrada do modelo, e a única.",
)
INFERENCIA_BRUTA = Conjunto(
    SILVER, "inference-raw",
    "Saída do BERTimbau sobre o resíduo, com escore por span, ANTES da validação. "
    "Guardar o bruto é o que permite medir quantos spans a porta recusou.",
)
INFERENCIA_ACEITA = Conjunto(
    SILVER, "inference-accepted",
    "Os spans do modelo que passaram pela porta de validação de nome — a mesma porta das "
    "camadas determinísticas.",
)
CONTRATO_FINAL = Conjunto(
    SILVER, "contract-final",
    "A prata consolidada: determinístico mais inferência aceita, reconciliados por "
    "precedência. É a única entrada do ouro.",
)

# ── ouro ───────────────────────────────────────────────────────────────────

FORNECEDOR_MES = Conjunto(
    GOLD, "supplier_month",
    "Quanto cada fornecedor recebeu por mês, em quantos órgãos. Soma apenas valores que "
    "satisfazem a regra financeira — ver contrato.financeiro.",
    particionado=False,
)
ORGAO_MES = Conjunto(
    GOLD, "agency_month",
    "Quanto cada órgão contratou por mês, de quantos fornecedores. Mesma regra financeira.",
    particionado=False,
)
QUALIDADE = Conjunto(
    GOLD, "quality",
    "As métricas do próprio pipeline, por execução: cobertura por camada e por campo, com "
    "intervalo de Wilson. É o que torna a degradação visível sem ninguém ir olhar.",
    particionado=False,
)

CONJUNTOS: tuple[Conjunto, ...] = (
    PUBLICACAO,
    CONTRATO, DESCARTE, RESIDUO, INFERENCIA_BRUTA, INFERENCIA_ACEITA, CONTRATO_FINAL,
    FORNECEDOR_MES, ORGAO_MES, QUALIDADE,
)

POR_TABELA: dict[str, Conjunto] = {c.tabela: c for c in CONJUNTOS}


def da_camada(camada: str) -> tuple[Conjunto, ...]:
    """Os conjuntos de uma camada, na ordem de declaração."""
    return tuple(c for c in CONJUNTOS if c.camada == camada)


# Os prefixos que não são dado
#
# Separados dos conjuntos porque não têm tabela nem esquema.

ENTRADA_MODELO = "model-input"
"""JSONL que o Batch Transform lê. Gerado do resíduo, descartável."""

SAIDA_MODELO = "model-output"
"""JSONL que o Batch Transform escreve. É a origem de `silver/inference-raw/`."""

PESOS = "model"
"""O `model.tar.gz` do BERTimbau ajustado, por versão. Nunca sobrescrito: `model/v1/`."""

RESULTADO_ATHENA = "athena-results"
"""Saída de consulta do workgroup. Tem regra de ciclo de vida própria."""

MEDIDA = "metrics"
"""Métricas por execução."""


# A janela de processamento
#
# Qual intervalo de edições um estágio processa. Vive aqui, e não em cada job, porque bronze e
# prata têm de recortar o mesmo intervalo: se cada um resolvesse a sua, uma divergência entre
# os dois não teria sintoma — a prata leria partições que o bronze não escreveu naquela
# execução.
#
# A janela tem duas formas. Um valor absoluto (`2026-09-14` a `2026-09-18`) é a ferramenta
# certa para reprocessar um intervalo à mão. Com a regra do EventBridge em `rate(24 hours)`,
# porém, uma janela absoluta fixa reprocessaria todos os dias as mesmas edições e nunca
# ingeriria edição nova. O fluxo diário precisa de janela relativa à data de execução, e é o
# que o modo `ultimos:N` fornece.
#
# Valor malformado falha alto, e não silenciosamente para "sem limite". Um `ultimos:7x`
# tratado como `tudo` processaria o corpus inteiro todos os dias sem ninguém notar.

SEM_LIMITE = "tudo"
"""Sentinela de janela sem limite. Palavra e não string vazia: valor vazio em
`DefaultArguments` do Glue não sobrevive de forma confiável, e um sentinela explícito aparece no
log do job — que é onde alguém vai procurar por que só metade do corpus foi processada."""

PREFIXO_RELATIVO = "ultimos:"
"""Prefixo do modo relativo. `ultimos:7` são os sete dias que terminam hoje, inclusive."""

_RE_DATA_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def resolver_janela(
    desde: str, ate: str, *, hoje: dt.date | None = None
) -> tuple[str, str]:
    """
    Resolve a especificação da janela em valores absolutos, ou no sentinela.

    `desde` aceita três formas: `tudo`, uma data ISO, ou `ultimos:N`. `ate` aceita duas: `tudo`
    ou uma data ISO. O relativo só existe no início porque é assim que ele é usado — "os últimos
    N dias" é uma janela inteira, e o fim dela é hoje, que `tudo` já cobre.

    `hoje` é injetável para fixar a data. Sem isso, o modo relativo daria resultados
    diferentes a cada dia.

    Raises:
        ValueError: se qualquer uma das pontas não tiver forma reconhecida. Falhar aqui custa o
            start do job; tratar como `tudo` custaria o corpus inteiro, todos os dias.
    """
    base = hoje or dt.date.today()

    if desde == SEM_LIMITE:
        inicio = SEM_LIMITE
    elif _RE_DATA_ISO.match(desde):
        inicio = desde
    elif desde.startswith(PREFIXO_RELATIVO):
        dias = desde[len(PREFIXO_RELATIVO):]
        if not dias.isdigit() or int(dias) < 1:
            raise ValueError(
                f"janela relativa inválida: {desde!r}. Esperado "
                f"{PREFIXO_RELATIVO}N com N inteiro maior que zero"
            )
        inicio = (base - dt.timedelta(days=int(dias) - 1)).isoformat()
    else:
        raise ValueError(
            f"início de janela inválido: {desde!r}. Esperado {SEM_LIMITE!r}, "
            f"uma data AAAA-MM-DD, ou {PREFIXO_RELATIVO}N"
        )

    if ate == SEM_LIMITE:
        fim = SEM_LIMITE
    elif _RE_DATA_ISO.match(ate):
        fim = ate
    else:
        raise ValueError(
            f"fim de janela inválido: {ate!r}. Esperado {SEM_LIMITE!r} ou uma data AAAA-MM-DD"
        )

    if inicio != SEM_LIMITE and fim != SEM_LIMITE and inicio > fim:
        raise ValueError(f"janela vazia: início {inicio} é posterior ao fim {fim}")

    return inicio, fim


def na_janela(data: str, desde: str, ate: str) -> bool:
    """
    Diz se a edição de `data` entra na janela já resolvida, ambas as pontas inclusivas.

    Compara **string** e não data: `AAAA-MM-DD` ordena lexicograficamente igual a
    cronologicamente, e comparar texto dispensa converter.
    """
    if desde != SEM_LIMITE and data < desde:
        return False
    return ate == SEM_LIMITE or data <= ate


def uri(raiz: str, *partes: str) -> str:
    """
    Monta uma URI S3 juntando as partes com uma barra só.

    Existe para que nenhum chamador escreva `f"{raiz}/{a}/{b}"` — a barra dupla que sai de
    `raiz` já terminada em barra não quebra o S3, mas faz `s3://b//prefixo` e
    `s3://b/prefixo` serem chaves diferentes, e a metade do pipeline que usa uma não vê o
    que a outra escreveu.
    """
    limpas = [p.strip("/") for p in partes if p]
    return "/".join([raiz.rstrip("/"), *limpas])
