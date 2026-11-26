"""rotulo.py — converte o gabarito da fonte em rótulo BIO e mede a ida e volta."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from domsp import texto as T
from domsp.dominio import parte
from domsp.modelo import CONTRATADA, IGNORAR, INDICE

# O SPAN


@dataclass(frozen=True)
class Span:
    """
    Um trecho rotulado: posição em caracteres do texto, classe, e como foi achado.

    `origem` indica se o span foi achado por ocorrência exata do nome no texto ou por
    prefixo estendido — as duas situações têm confiabilidade diferente.
    """

    inicio: int
    fim: int
    classe: str
    origem: str          # 'exato' | 'prefixo'

    def sobrepoe(self, outro: Span) -> bool:
        return self.inicio < outro.fim and outro.inicio < self.fim


ORIGEM_EXATO = "exato"
ORIGEM_PREFIXO = "prefixo"

# Quantos caracteres do começo do nome são exigidos para tentar o caminho por prefixo.
# Abaixo disso um prefixo curto casaria com outra razão social do mesmo texto, e o
# span sairia no lugar errado.
PREFIXO_MINIMO = 25

# Quanto o fim do span pode ser estendido além do prefixo casado, quando o nome
# completo não ocorre literalmente. A folga cobre a diferença típica entre a grafia do
# formulário e a da prosa (ponto, hífen, sufixo abreviado), e não mais que isso.
FOLGA_FIM = 16


def _comparavel(s: str) -> str:
    """
    A forma em que dois nomes são comparados: sem acento, minúscula, espaço colapsado.

    O mesmo nome pode ser grafado de formas diferentes no formulário e na prosa, por
    acento, caixa e espaço; comparar a forma crua faria o alinhamento falhar nesses
    casos.
    """
    return T.remover_acentos(T.colapsar_espacos(s or "")).lower()


def localizar(nome: str | None, texto: str, classe: str = CONTRATADA) -> Span | None:
    """
    A posição do `nome` em `texto`, por ocorrência exata ou por prefixo estendido.

    Caminho 1 — ocorrência exata: o nome comparável aparece no texto comparável.

    Caminho 2 — prefixo estendido: o nome não aparece inteiro, mas seus primeiros
    `PREFIXO_MINIMO` caracteres aparecem (por exemplo, prosa que abrevia o nome). O fim é estendido até o comprimento do nome mais `FOLGA_FIM`,
    limitado pela fronteira de palavra.

    Devolve `None` quando nenhum dos dois funciona.

    O alinhamento é feito sobre a forma **comparável**, mas o span devolvido é em
    posição do texto **original** — válido quando `_comparavel` preserva comprimento.
    `remover_acentos` usa NFD e descarta as combinantes sem mudar o número de caracteres
    base; já `colapsar_espacos` encurta o texto se houver espaço horizontal repetido.
    Por isso, se `len(fonte) != len(texto)`, a função devolve `None`. A guarda no fim do
    módulo verifica a premissa para as amostras sem espaço repetido.
    """
    if not nome or not texto:
        return None

    alvo = _comparavel(nome)
    fonte = _comparavel(texto)
    if len(fonte) != len(texto):
        # Sem correspondência de posição (por exemplo, espaço repetido no texto), devolver
        # `None` evita um span deslocado.
        return None

    inicio = fonte.find(alvo)
    if inicio >= 0:
        return Span(inicio, inicio + len(alvo), classe, ORIGEM_EXATO)

    if len(alvo) < PREFIXO_MINIMO:
        return None
    inicio = fonte.find(alvo[:PREFIXO_MINIMO])
    if inicio < 0:
        return None

    limite = min(len(texto), inicio + len(alvo) + FOLGA_FIM)
    fim = limite
    while fim < len(texto) and not texto[fim].isspace():
        fim += 1        # não corta palavra ao meio
    return Span(inicio, fim, classe, ORIGEM_PREFIXO)


# A GRADE DE PALAVRA


@dataclass(frozen=True)
class Palavra:
    """Uma palavra e sua posição em caracteres. A unidade da grade de rotulagem."""

    texto: str
    inicio: int
    fim: int


def palavras(texto: str) -> list[Palavra]:
    """
    A segmentação em palavras, por espaço, com as posições preservadas.

    Por espaço e não por pontuação: "LTDA." é uma palavra, e o ponto entra no span. O
    tokenizador do BERT parte "LTDA." em subtokens de qualquer forma, e o rótulo é
    atribuído ao **primeiro subtoken de cada palavra** — a granularidade que importa é
    a palavra.
    """
    saida: list[Palavra] = []
    i = 0
    n = len(texto)
    while i < n:
        while i < n and texto[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not texto[j].isspace():
            j += 1
        saida.append(Palavra(texto[i:j], i, j))
        i = j
    return saida


def rotular_palavras(texto: str, spans: Sequence[Span]) -> list[str]:
    """
    O rótulo IOB2 de cada palavra: `B-` na primeira que o span cobre, `I-` nas demais.

    A cobertura é por **interseção**, não por contenção: uma palavra conta como dentro
    do span se qualquer parte dela está dentro. Isso absorve o desalinhamento de
    fronteira — um span que termina no meio de "LTDA." rotula "LTDA." inteira. O trecho
    decodificado é depois comparado ao nome por `medir_ida_e_volta`.
    """
    pals = palavras(texto)
    rotulos = ["O"] * len(pals)
    for s in spans:
        primeira = True
        for i, p in enumerate(pals):
            if p.inicio < s.fim and s.inicio < p.fim:
                rotulos[i] = f"{'B' if primeira else 'I'}-{s.classe}"
                primeira = False
    return rotulos


def decodificar(texto: str, rotulos: Sequence[str],
                classe: str = CONTRATADA) -> list[str]:
    """
    O caminho de volta: de rótulos por palavra para os trechos que eles nomeiam.

    Um `B-` abre trecho, `I-` da mesma classe continua, qualquer outra coisa fecha. A
    junção é por espaço simples, e não pelo texto original entre as palavras, porque a
    saída desta função é comparada com o nome do gabarito, que também é espaço simples.
    """
    pals = palavras(texto)
    achados: list[str] = []
    atual: list[str] = []
    for p, r in zip(pals, rotulos, strict=False):
        if r == f"B-{classe}":
            if atual:
                achados.append(" ".join(atual))
            atual = [p.texto]
        elif r == f"I-{classe}" and atual:
            atual.append(p.texto)
        else:
            if atual:
                achados.append(" ".join(atual))
            atual = []
    if atual:
        achados.append(" ".join(atual))
    return achados


# A CODIFICAÇÃO PARA O MODELO


@dataclass(frozen=True)
class Janela:
    """
    Uma janela de tokens pronta para o modelo, com o mapa de volta para palavra.

    `ids_palavra` é o que permite decodificar: sem ele, recolar a saída do modelo em
    palavras exigiria refazer a tokenização.
    """

    ids: list[int]
    mascara: list[int]
    rotulos: list[int]
    ids_palavra: list[int | None]
    deslocamento: int          # índice da primeira palavra desta janela


def janelas(texto: str, rotulos_palavra: Sequence[str], tokenizador,
            max_tokens: int = 512, passo: int = 128) -> list[Janela]:
    """
    O texto em janelas deslizantes de tokens, com rótulo por primeiro subtoken.

    `tokenizador` é recebido por parâmetro, e não importado, para que este módulo
    continue utilizável sem `transformers` instalado.

    **Janela deslizante com sobreposição**, e não truncamento: o texto de uma
    publicação pode passar de `max_tokens`, e a parte contratada pode estar em qualquer
    posição. `passo` menor que `max_tokens` faz janelas consecutivas se sobreporem, para
    que um span perto da fronteira de uma janela apareça na seguinte.

    **Rótulo só no primeiro subtoken da palavra**; os demais e os especiais recebem
    `IGNORAR`, já que a decodificação agrega por palavra.
    """
    pals = palavras(texto)
    if not pals:
        return []

    # `is_split_into_words` porque a grade é a palavra: deixar o tokenizador segmentar
    # o texto cru produziria um alinhamento diferente deste.
    codificado = tokenizador(
        [p.texto for p in pals], is_split_into_words=True,
        add_special_tokens=True, truncation=False,
    )
    ids_todos = codificado["input_ids"]
    palavra_de = codificado.word_ids()

    saida: list[Janela] = []
    inicio = 0
    total = len(ids_todos)
    while inicio < total:
        fim = min(inicio + max_tokens, total)
        ids = ids_todos[inicio:fim]
        mapa = palavra_de[inicio:fim]
        rot: list[int] = []
        vista: set[int] = set()
        for idp in mapa:
            if idp is None or idp in vista:
                rot.append(IGNORAR)
            else:
                vista.add(idp)
                rot.append(INDICE.get(rotulos_palavra[idp], INDICE["O"]))
        saida.append(Janela(ids, [1] * len(ids), rot, mapa,
                            deslocamento=next((i for i in mapa if i is not None), 0)))
        if fim >= total:
            break
        inicio += passo
    return saida


# O TETO DA SUPERVISÃO


@dataclass(frozen=True)
class Teto:
    """
    O resultado da ida e volta: o melhor que o modelo poderia fazer com este rótulo.

    `similaridade_media` é o teto propriamente dito. `identicos` conta as voltas que
    `acerto` aceita como o mesmo nome, `ajustados` as que ela não aceita e `perdidos`
    as sem span, sem trecho decodificado ou recusadas pela porta. A soma
    `identicos + ajustados + perdidos` é igual a `casos`.
    """

    casos: int
    exatos: int
    por_prefixo: int
    sem_span: int
    identicos: int
    ajustados: int
    perdidos: int
    recusados_pela_porta: int
    similaridade_media: float

    @property
    def teto(self) -> float:
        return self.similaridade_media


def _similaridade(a: str, b: str) -> float:
    """
    Semelhança entre dois nomes, na forma comparável, por `SequenceMatcher`.
    """
    from difflib import SequenceMatcher

    return SequenceMatcher(None, _comparavel(a), _comparavel(b)).ratio()


def acerto_na_metrica(obtido: str, esperado: str) -> bool:
    """
    Se dois nomes contam como **o mesmo** pela régua que a avaliação cobra.

    Compara `parte.normalizar_nome` dos dois lados, a mesma chave usada para agrupar
    variações do nome de um fornecedor (remove acento, pontuação de borda e sufixos
    societários).
    """
    return parte.normalizar_nome(obtido) == parte.normalizar_nome(esperado)


def medir_ida_e_volta(casos: Iterable[tuple[str, str]],
                      porta: Callable[[str], object] | None = None,
                      acerto: Callable[[str, str], bool] | None = None) -> Teto:
    """
    Refaz o caminho completo sobre `(texto, nome_correto)` e devolve o teto.

    O caminho é: `localizar` → `rotular_palavras` → `decodificar` → porta de limpeza.
    Passar pela porta é essencial: o modelo em produção também passa por ela
    (`dominio.parte.porta_de_nome`), então um teto medido sem a porta seria otimista em
    relação ao que o pipeline realiza. Casos que a porta recusa são contados em
    `recusados_pela_porta`.

    `porta` é injetável para medir o teto da rotulagem isolada, passando `None`. A
    diferença entre os dois é o custo da porta.

    `acerto` é a régua que decide se a volta devolveu o mesmo nome; o default é
    `acerto_na_metrica`.

    `similaridade_media` é calculada por `SequenceMatcher` e reportada como
    diagnóstico, não como critério: diz o quanto as voltas que falharam ficaram perto.
    """
    decide = acerto or acerto_na_metrica
    n = exatos = prefixo = sem_span = 0
    identicos = ajustados = perdidos = recusados = 0
    soma = 0.0
    aplicar = porta if porta is not None else (lambda s: s)

    for texto, nome in casos:
        n += 1
        span = localizar(nome, texto, CONTRATADA)
        if span is None:
            sem_span += 1
            perdidos += 1
            continue
        exatos += span.origem == ORIGEM_EXATO
        prefixo += span.origem == ORIGEM_PREFIXO

        rotulos = rotular_palavras(texto, [span])
        voltas = decodificar(texto, rotulos, CONTRATADA)
        if not voltas:
            perdidos += 1
            continue

        bruto = voltas[0]
        resultado = aplicar(bruto)
        obtido = getattr(resultado, "nome", resultado)
        if not obtido:
            recusados += 1
            perdidos += 1
            continue

        soma += _similaridade(obtido, nome)
        if decide(obtido, nome):
            identicos += 1
        else:
            ajustados += 1

    return Teto(casos=n, exatos=exatos, por_prefixo=prefixo, sem_span=sem_span,
                identicos=identicos, ajustados=ajustados, perdidos=perdidos,
                recusados_pela_porta=recusados,
                similaridade_media=round(soma / n, 4) if n else 0.0)


def porta_padrao(bruto: str):
    """A porta que o pipeline usa, para `medir_ida_e_volta` medir o teto realizável."""
    return parte.porta_de_nome(bruto)


# A GUARDA
#
# `localizar` calcula posição na forma comparável e devolve posição no texto
# original, e isso só é válido se `_comparavel` preservar comprimento. A premissa é
# verificada na importação sobre os casos que a quebrariam se estivesse errada:
# acento precomposto e as compatibilidades que `texto.remover_acentos` preserva de
# propósito (`º`, `ª`).
#
# Sem esta guarda, uma mudança em `remover_acentos` faria `localizar` devolver span
# deslocado, sem lançar exceção.

for _amostra in ("CONTRATAÇÃO", "art. 1º", "ASSOCIAÇÃO CRISTÃ", "PRODAM-SP S/A",
                 "3ª VARA", "MARIA ÂNGELA SOUZA"):
    if len(_comparavel(_amostra)) != len(_amostra):
        raise AssertionError(
            f"_comparavel não preserva comprimento em {_amostra!r}: "
            f"{len(_comparavel(_amostra))} != {len(_amostra)}. "
            "localizar() devolveria span deslocado, que é falha silenciosa."
        )
del _amostra
