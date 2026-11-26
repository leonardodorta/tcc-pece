"""modelo — define o modelo neural (BERTimbau) usado na camada de resíduo: qual variante, classes de rótulo e guardas de consistência."""
from __future__ import annotations

# Identificador do modelo publicado e da alternativa conhecida.
MODELO_LARGE = "neuralmind/bert-large-portuguese-cased"
MODELO_BASE = "neuralmind/bert-base-portuguese-cased"

# O modelo deste projeto é sempre BERTimbau; a única variação admitida é de tamanho
# (large x base). A guarda abaixo roda na importação e rejeita qualquer outra família.
PREFIXO_BERTIMBAU = "neuralmind/bert-"
SUFIXO_BERTIMBAU = "-portuguese-cased"

CANDIDATOS: tuple[str, ...] = (MODELO_LARGE, MODELO_BASE)

# Nome curto de cada candidato (`large` | `base`), devolvido por `tamanho()`.
TAMANHOS: dict[str, str] = {MODELO_LARGE: "large", MODELO_BASE: "base"}


def tamanho(identificador: str) -> str:
    """
    O nome curto (`large` | `base`) de um identificador de modelo.

    Levanta `KeyError` em vez de devolver um valor plausível: um identificador
    desconhecido produziria um nome vazio ou incorreto, e um caminho montado com ele
    seria válido, mas apontaria para um lugar errado.
    """
    if identificador not in TAMANHOS:
        raise KeyError(
            f"'{identificador}' não é um candidato declarado. Candidatos: "
            f"{sorted(TAMANHOS)}"
        )
    return TAMANHOS[identificador]


def eh_bertimbau(identificador: str) -> bool:
    """
    Verdadeiro só para os modelos BERTimbau publicados por `neuralmind`.

    Checa prefixo **e** sufixo: `neuralmind` publica outros artefatos, e
    `bert-...-portuguese-cased` é a família do BERTimbau. Um `bert-base-multilingual`
    ou um `bert-...-uncased` não passa na checagem.
    """
    return (identificador.startswith(PREFIXO_BERTIMBAU)
            and identificador.endswith(SUFIXO_BERTIMBAU))


# O modelo publicado.
MODELO_PUBLICADO = MODELO_LARGE

# As classes do rótulo, no esquema IOB2 do CoNLL-2003.
#
# CONTRATANTE é uma classe própria para que o modelo distinga o polo público do polo
# contratado: sem ela, o nome de um órgão no texto teria o mesmo rótulo `O` que o
# restante do texto e não haveria como ensinar a diferença em relação à contratada.
CONTRATADA = "CONTRATADA"
CONTRATANTE = "CONTRATANTE"

ROTULOS: tuple[str, ...] = (
    "O",
    f"B-{CONTRATADA}", f"I-{CONTRATADA}",
    f"B-{CONTRATANTE}", f"I-{CONTRATANTE}",
)
INDICE: dict[str, int] = {r: i for i, r in enumerate(ROTULOS)}

# O índice que a função de perda do PyTorch ignora. Usado nos subtokens que não são o
# primeiro de cada palavra e nos tokens especiais.
IGNORAR = -100


# Guarda executável: roda na importação e falha se algum candidato ou o modelo
# publicado não for BERTimbau.
for _identificador in (*CANDIDATOS, MODELO_PUBLICADO):
    if not eh_bertimbau(_identificador):
        raise AssertionError(
            f"'{_identificador}' não é BERTimbau. O modelo deste projeto é BERTimbau "
            f"sempre; a única variação admitida é de TAMANHO (large × base), "
            f"nunca de família. Ver a restrição declarada neste módulo."
        )
if MODELO_PUBLICADO not in CANDIDATOS:
    raise AssertionError(
        f"MODELO_PUBLICADO ({MODELO_PUBLICADO}) não está entre os CANDIDATOS: o modelo "
        f"servido tem de ser um dos candidatos declarados"
    )
if len(set(CANDIDATOS)) != len(CANDIDATOS):
    raise AssertionError("CANDIDATOS tem repetição: os candidatos devem ser distintos")
del _identificador
