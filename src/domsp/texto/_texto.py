"""
_texto.py — converte o HTML publicado em texto Unicode comparável, sem tag, entidade ou homóglifo.

## A ordem das sete operações é contrato

    1. remover comentário HTML
    2. tag de bloco   -> "\\n"     (antes das demais tags)
    3. demais tags    -> " "
    4. html.unescape                (uma vez, DEPOIS de remover tags)
    5. tabela de tradução
    6. NFC                          (nunca NFKC/NFKD)
    7. colapso de espaço horizontal, preservando "\\n"

**O passo 2 vem antes do 3.** Se todas as tags virassem espaço de uma vez, o fim de
um bloco colaria no começo do seguinte: `...LIMPO</h5><p>EXTRATO...` sem espaço entre
eles viraria `LIMPOEXTRATO`. A quebra de bloco em `\\n` cria a fronteira de propósito;
as demais tags viram espaço porque dentro de um parágrafo o que separa dois trechos é
espaço, não linha.

**O passo 4 vem DEPOIS do 3.** O corpo carrega `&lt;` e `&gt;` escapados de propósito,
e desescapá-los antes de remover as tags os transformaria em `<` e `>` reais, que o
regex de tag comeria junto com o texto entre eles. `unescape` vem **antes** de toda
operação sensível a acento (passos 5 e 6), porque é ele quem cria o acento a partir
da entidade.

## Duas coisas que este módulo deliberadamente NÃO faz

Não corrige dupla codificação — isso é feito na fronteira do bronze, em `fonte/edicao.py`,
e só nos campos de metadado, não no corpo.

Não usa NFKC nem NFKD. O NFKC alteraria o ordinal `º` (destruindo "art. 1º"), `ª` e
`²`, e ainda assim não resolve o hífen inseparável U+2011. A normalização de
compatibilidade é feita por uma tabela, que colhe o benefício do NFKC só onde ele
acerta. Consequência: `remover_acentos` usa **NFD**.

## A tabela é derivada por classe de ação

A tabela classifica cada caractere pela **ação** que a normalização precisa tomar:
homóglifo (substituir por latino), invisível (remover ou virar espaço), pontuação
equivalente (ASCII exato), PUA (remover), compatibilidade (preservar `º ª ²`,
normalizar o resto), combinante solto (NFC compõe), acento pré-composto e notação
grega (preservar).
"""
from __future__ import annotations

import html
import re
import unicodedata

# HTML — OS TRÊS RECORTES

# Comentário primeiro. O corpo traz `<!--style=...-->` no meio da prosa. `re.DOTALL`
# porque o comentário pode atravessar linha. Vira espaço, não vazio, para não fundir
# as palavras dos dois lados.
_RE_COMENTARIO = re.compile(r"<!--.*?-->", re.DOTALL)

# Tags de bloco -> "\n". Precisam virar quebra ANTES de as demais serem removidas,
# senão as palavras dos dois lados da fronteira de bloco se fundem. `br` e `hr`
# entram porque, embora vazios, também são quebra visual.
_RE_BLOCO = re.compile(
    r"</?(?:p|div|h[1-6]|br|hr|tr|td|th|li|ul|ol|dl|dt|dd|table|thead|tbody|tfoot"
    r"|section|article|aside|header|footer|blockquote|pre|form|fieldset)\b[^>]*>",
    re.IGNORECASE,
)

# Toda tag restante -> " ". Dentro de um bloco o que separa dois trechos é espaço.
_RE_TAG = re.compile(r"<[^>]+>")

# Espaço HORIZONTAL: espaço, tabulação, form feed, vertical tab. Não inclui "\n"
# nem "\r" de propósito — a quebra de linha é informação (em rótulo agregado como
# "PARTÍCIPES:" ela é o único separador entre as partes).
_RE_ESPACO_H = re.compile(r"[ \t\f\v]+")

# Duas ou mais quebras viram uma. Blocos vazios geram sequências de "\n".
_RE_LINHAS = re.compile(r"\n{2,}")


# A TABELA DE TRADUÇÃO, POR CLASSE DE AÇÃO
#
# Toda chave é code point numérico, nunca caractere entre aspas: várias entradas
# abaixo são invisíveis, e escrevê-las como literal encheria o arquivo de caracteres
# que não se revisam em leitura de código. Assim cada entrada fica explícita.

# ── invisíveis que SÃO espaço -> espaço comum ──────────────────────────────
#
# A classe "invisível" tem DUAS ações — virar espaço ou sumir. O U+202F aparece na
# fronteira rótulo-valor do formulário (`<strong>Assunto: </strong>&#8239;...`);
# removê-lo colaria os dois, por isso vira espaço, não some.
_ESPACOS = {
    0x00A0: " ",   # NO-BREAK SPACE
    0x202F: " ",   # NARROW NO-BREAK SPACE — fronteira rótulo-valor
    0x2002: " ",   # EN SPACE
    0x2003: " ",   # EM SPACE
    0x2004: " ",   # THREE-PER-EM SPACE
    0x2005: " ",   # FOUR-PER-EM SPACE
    0x2006: " ",   # SIX-PER-EM SPACE
    0x2007: " ",   # FIGURE SPACE
    0x2008: " ",   # PUNCTUATION SPACE
    0x2009: " ",   # THIN SPACE
    0x200A: " ",   # HAIR SPACE
    0x205F: " ",   # MEDIUM MATHEMATICAL SPACE
    0x3000: " ",   # IDEOGRAPHIC SPACE
    0x1680: " ",   # OGHAM SPACE MARK
}

# ── invisíveis de largura zero e controles -> removidos ────────────────────
#
# O `&shy;` (hífen suave) é removido e não vira espaço: ele aparece em corridas
# seguidas (`e&shy;&shy;&shy;...`), enchimento de diagramação, e virá-las espaço
# criaria uma corrida de espaços onde havia uma palavra.
#
# O que NÃO está aqui, de propósito: os controles C1 0x81 e 0x8D. Os contextos são
# `FUNDIÃ<0x81>RIA` e `CONCLUÃ<0x8D>DA` — metade de uma letra duplo-codificada (`Á` =
# C3 81, `Í` = C3 8D). Removê-los produziria `FUNDIRIA` e `CONCLUDA`: palavras
# plausíveis e erradas. Ficam intocados.
_REMOVIDOS = {
    0x00AD: "",    # SOFT HYPHEN — enchimento, não hifenização
    0xFE0F: "",    # VARIATION SELECTOR-16 — categoria Mn, mas não compõe nada
    0x2060: "",    # WORD JOINER
    0x200B: "",    # ZERO WIDTH SPACE
    0x200C: "",    # ZERO WIDTH NON-JOINER
    0x200D: "",    # ZERO WIDTH JOINER
    0x200E: "",    # LEFT-TO-RIGHT MARK
    0x200F: "",    # RIGHT-TO-LEFT MARK
    0xFEFF: "",    # ZERO WIDTH NO-BREAK SPACE (BOM no meio do texto)
    0xFFFA: "",    # INTERLINEAR ANNOTATION SEPARATOR
    0xFFFB: "",    # INTERLINEAR ANNOTATION TERMINATOR
}

# ── homóglifo cirílico -> latino ───────────────────────────────────────────
#
# O cirílico infiltrado pode cair nos campos que a extração lê. Exemplos:
# `ОВЈЕТО: CONTRATAÇÃO` (o rótulo inteiro em cirílico),
# `CОMÉRCIO E SERVIÇOS LTDA (MЕ)`, `CAIХА ECONÔMICA FEDERAL`, `FULANO DE TAL NETО`,
# `САPS Adulto III`.
#
# O mapeamento segue a identidade visual e preserva a caixa. Duas entradas são
# decididas pelo contexto, não só pela forma visual:
#   З (U+0417) -> "3", não "Z": aparece em `SALA ЗА` (designação de sala).
#   Ӧ (U+04E6) -> "Ö", não "O": aparece em `SCHӦLLY`, nome de empresa que se escreve
#   com "ö".
_CIRILICO = {
    0x0410: "A", 0x0412: "B", 0x0415: "E", 0x041C: "M", 0x041D: "H",
    0x041E: "O", 0x0420: "P", 0x0421: "C", 0x0422: "T", 0x0425: "X",
    0x0408: "J", 0x0417: "3",   # З -> 3, pelo contexto `SALA ЗА`
    0x04E6: "Ö",                # Ӧ -> Ö, pelo contexto `SCHӦLLY`
    0x0430: "a", 0x0435: "e", 0x043A: "k", 0x043C: "m", 0x043D: "n",
    0x043E: "o", 0x043F: "p", 0x0440: "p", 0x0441: "c", 0x0442: "t",
    0x0445: "x",
    # Outros confundíveis cirílicos com forma latina exata; entram por prevenção.
    0x0406: "I", 0x0456: "i", 0x0405: "S", 0x0455: "s",
    0x04AE: "Y", 0x04BA: "h", 0x04C0: "I", 0x04CF: "l",
    0x0433: "r", 0x0475: "v",
}

# ── notação grega: maiúscula com confundível é homóglifo; o resto preserva ──
#
# Tratar grego e cirílico juntos (por faixa) quebraria as MINÚSCULAS gregas, que
# são notação legítima: `Σ DR / Σ OR` (somatório),
# `desvio-padrão (σ)`, `550 μm` (micrômetro), `correlação ρ`, `τ = +10`. Mapeá-las
# para latino destruiria essas fórmulas. Preservar TODO o grego, por outro lado,
# deixaria passar três MAIÚSCULAS que são homóglifo: `FILHO ΜΕ` (microempresa),
# `Ο Diretor` (o artigo). A fronteira é **maiúscula com confundível latino exato** ×
# tudo o mais — Σ não entra porque não tem confundível latino, e nenhuma minúscula
# grega entra.
_GREGO = {
    0x039C: "M",   # Μ — `FILHO ΜΕ`
    0x0395: "E",   # Ε — `FILHO ΜΕ`
    0x039F: "O",   # Ο — `Ο Diretor de Escola`
    0x037E: ";",   # GREEK QUESTION MARK — visualmente idêntico ao ponto-e-vírgula
    # Demais maiúsculas gregas com confundível exato; entram por prevenção.
    0x0391: "A", 0x0392: "B", 0x0396: "Z", 0x0397: "H", 0x0399: "I",
    0x039A: "K", 0x039D: "N", 0x03A1: "P", 0x03A4: "T", 0x03A5: "Y",
    0x03A7: "X",
}

# ── pontuação com equivalente ASCII exato ──────────────────────────────────
#
# As aspas curvas são o grupo principal deste bloco. Mantê-las obrigaria toda regra que casa
# aspas a escrever quatro variantes.
_PONTUACAO = {
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',   # aspas duplas curvas
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",   # aspas simples curvas
    0x00AB: '"', 0x00BB: '"',                              # « »
    0x2039: "<", 0x203A: ">",                             # ‹ ›
    0x2010: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-",   # traços com equivalente
    0x2015: "-", 0x2212: "-",                             # horizontal bar, minus
    0x2024: ".", 0x2026: "...",                           # ponto de líder, reticências
}

# ── compatibilidade: a maioria PRESERVA; só entra o que é ruído ou grafia alt. ──
#
# `º ª ² ³ ¹` são significantes (por exemplo o ordinal `º`), e é por
# eles que o NFKC está recusado — nenhum deles entra na tabela. Só entram as
# compatibilidades que são ruído tipográfico ou grafia alternativa de algo que já
# existe.
_COMPATIBILIDADE = {
    0x2011: "-",       # NON-BREAKING HYPHEN. Normaliza porque custa zero e unifica
                       # com o hífen ASCII.
    0x2116: "nº",      # № NUMERO SIGN. Terceira grafia de "nº" (`CNPJ № ...`).
    0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl",
                       # ligaduras: a tabela as desfaz para que "ofício" com ligadura
                       # case com "ofício" sem. NFC as preserva.
}

# ── o grau usado como ordinal -> unificar em º ──────────────────────────────
#
# `°` e `º` são ambos usados como abreviação de "número" (`LEI N°`, `processo
# SEI n°`), além de `°` como grau (`° C`). Sem unificar, um padrão de "nº" escrito
# com `º` perde as ocorrências escritas com `°`. O sentido é `°` -> `º` porque as
# regras de extração dependem do ordinal.
_GRAU = {0x00B0: "º"}

# ── PUA: presente no HTML, ausente do texto ────────────────────────────────
#
# A PUA vive dentro de atributo de tag (marcador de lista do Word:
# `data-leveltext="&#61485;"`), e como o passo 3 remove tag antes de o passo 4
# desescapar, ela normalmente não chega ao texto. Ficam por prevenção.
_PUA = {
    0xF0B7: "-",   # marcador de lista da fonte Symbol
    0xF02D: "-",   # idem
    0xE113: "",    # desconhecido: remover é mais seguro que adivinhar
}


def _matematicos() -> dict[int, str]:
    """
    Letra matemática (U+1D400–U+1D7FF) -> latino/dígito, gerada e não transcrita.

    Uma fórmula de indicador pode vir com variáveis em itálico matemático, e sem esta
    conversão o nome da variável não casaria com ele mesmo escrito em ASCII.

    Gerada por NFKC sobre este bloco só, para colher o benefício do NFKC exatamente
    onde ele acerta sem pagar os ordinais que ele destruiria fora daqui.
    """
    tabela: dict[int, str] = {}
    for cp in range(0x1D400, 0x1D800):
        ch = chr(cp)
        alvo = unicodedata.normalize("NFKC", ch)
        if alvo != ch and len(alvo) == 1 and alvo.isascii() and alvo.isalnum():
            tabela[cp] = alvo
    return tabela


# A tabela-base, antes da faixa C1.
_BASE: dict[int, str] = {
    **_ESPACOS, **_REMOVIDOS, **_CIRILICO, **_GREGO, **_PONTUACAO,
    **_COMPATIBILIDADE, **_GRAU, **_PUA, **_matematicos(),
}


def _faixa_c1(base: dict[int, str]) -> dict[int, str]:
    """
    A faixa 0x80–0x9F lida como Windows-1252 — gerada.

    A faixa tem categoria Unicode `Cc` ("controle"), mas a fonte emite valores de
    byte do Windows-1252, e o que chega são sinais de pontuação (apóstrofo de plural,
    travessão, marcador de lista, aspa de abertura). O mapa é derivado em vez de
    escrito porque `cp1252` **é** a tabela.

    - **encadeia na tabela já montada** (`base.get`): `0x91` não para em U+2018, vai
      até `'`, o destino que `_PONTUACAO` já decidiu.
    - **`UnicodeDecodeError` significa "deixa como está"**, o que protege 0x81 e
      0x8D: o cp1252 não define essas posições, e em texto duplo-codificado elas são
      fragmentos de `Á` e `Í`.

    Referências numéricas da faixa (`&#146;`) são remapeadas por `html.unescape`
    pela tabela do HTML5 antes deste passo. O que chega aqui é o C1 cru no corpo.
    """
    tabela: dict[int, str] = {}
    for cp in range(0x80, 0xA0):
        try:
            alvo = bytes([cp]).decode("cp1252")
        except UnicodeDecodeError:
            continue          # posição indefinida no cp1252: metade de letra, não controle
        tabela[cp] = base.get(ord(alvo), alvo)
    return tabela


TRADUCAO: dict[int, str] = {**_BASE, **_faixa_c1(_BASE)}


# AS OPERAÇÕES


def traduzir(s: str) -> str:
    """Aplica `TRADUCAO` a um texto. É o passo 5 da ordem contratual."""
    return s.translate(TRADUCAO)


def normalizar(s: str | None) -> str:
    """
    Os passos 4, 5 e 6 sobre texto que já não tem tag: desescapar, traduzir, compor.

    `html.unescape` uma vez só: duas passadas transformariam `&amp;lt;` em `<`, isto
    é, um `&lt;` que a fonte escapou de propósito viraria sintaxe.

    `NFC` por último porque é ele que compõe os diacríticos que a fonte entrega
    soltos, como entidade numérica de combinante (`&#771;`, `&#769;`, `&#807;`).
    Compor depois de traduzir, porque a tradução pode remover um invisível que estava
    entre a letra e o seu acento.
    """
    if not s:
        return ""
    return unicodedata.normalize("NFC", traduzir(html.unescape(s)))


def html_para_texto(bruto: str | None) -> str:
    """
    HTML do Diário em texto, na ordem contratual das sete operações.

    A saída não tem tag, não tem entidade HTML, não tem homóglifo, e usa "\\n" como
    única quebra de linha significativa (as quebras de bloco), com o espaço
    horizontal já colapsado. Linhas em branco consecutivas colapsam em uma.
    """
    if not bruto:
        return ""
    t = _RE_COMENTARIO.sub(" ", bruto)
    t = _RE_BLOCO.sub("\n", t)
    t = _RE_TAG.sub(" ", t)
    t = normalizar(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_ESPACO_H.sub(" ", t)
    t = "\n".join(linha.strip() for linha in t.split("\n"))
    return _RE_LINHAS.sub("\n", t).strip()


def colapsar_espacos(s: str | None) -> str:
    """
    Colapsa espaço horizontal **preservando** a quebra de linha.

    Distinto de `re.sub(r"\\s+", " ", s)`: em rótulo que agrega mais de uma parte
    ("PARTÍCIPES:"), a quebra de linha é o único separador entre contratante e
    contratado.
    """
    return _RE_ESPACO_H.sub(" ", s or "")


def remover_acentos(s: str | None) -> str:
    """
    Remove diacrítico preservando caixa, para comparação de nome.

    NFD, não NFKD: o NFKD achataria `º ª ²` nas janelas de rótulo e dobraria os
    ordinais em letra. O NFD decompõe só a letra acentuada, cujo combinante é então
    descartado.
    """
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if not unicodedata.combining(c))


def normalizar_chave(s: str | None) -> str:
    """
    Rótulo de campo -> chave de dicionário: minúscula, sem acento, sem pontuação de borda.

    Torna o mapeamento robusto às variações reais da fonte — "Órgão"/"Orgão",
    "Número"/"NUMERO"/"Nº"/"N°"/"№". Assume que a entrada já passou por `normalizar`
    (ou ao menos por `traduzir`): não desescapa nem traduz de novo.
    """
    if not s:
        return ""
    chave = remover_acentos(s).lower()
    chave = re.sub(r"\s+", " ", chave).strip()
    return chave.strip(" :.-–—")
