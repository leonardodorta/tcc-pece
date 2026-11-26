"""dinheiro.py — elege qual ocorrência monetária de um texto é o valor do ajuste e classifica a sua grandeza (total, mensal, multa etc.)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..contrato import vocabulario as V

try:  # pragma: no cover
    from domsp import texto as _texto

    _colapsar_espacos = _texto.colapsar_espacos
    _remover_acentos = _texto.remover_acentos
except Exception:  # pragma: no cover
    def _colapsar_espacos(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _remover_acentos(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "")
        return "".join(c for c in s if not unicodedata.combining(c))


# AS GRANDEZAS — exatamente o vocabulário da coluna `amount_magnitude`

TOTAL = "total"
ESTIMADO = "estimado"
ANUAL = "anual"
MENSAL = "mensal"
UNITARIO = "unitario"
PARCELA = "parcela"
PENALIDADE = "penalidade"
SALDO = "saldo"
INDEFINIDO = "indefinido"

# Somar grandezas diferentes não tem sentido; o que é somável na camada gold é
# declarado em `contrato/financeiro.py` (coluna `is_summable`: total/estimado de
# natureza gasto). Aqui só se CLASSIFICA.
GRANDEZAS_SOMAVEIS = frozenset({TOTAL, ESTIMADO})


# OS RÓTULOS — duas famílias, para manter grandeza e eleição distintas
#
# `qualificador=True` significa que o rótulo NOMEIA a grandeza; `False` que só a
# INTRODUZ. `peso` ordena valores concorrentes na eleição (qual R$ é o do ajuste) e
# NÃO decide grandeza. Os pesos exprimem uma ORDEM relativa (nota de empenho > no
# valor de > multa), não magnitudes exatas: é só a ordem que importa.

@dataclass(frozen=True)
class Rotulo:
    padrao: re.Pattern[str]
    grandeza: str
    qualificador: bool
    peso: int


def _p(expr: str) -> re.Pattern[str]:
    # os textos são comparados já sem acento e em minúsculas; os padrões seguem isso
    return re.compile(expr, re.IGNORECASE)


# Ordem de declaração importa só como desempate estável quando dois padrões casam no
# mesmo ponto; a decisão real vem de (qualificador, proximidade). Os padrões operam
# sobre texto SEM ACENTO (`_remover_acentos` é aplicado à janela antes de casar);
# as classes com acento (ex.: `[eê]`) são tolerâncias redundantes.
_ROTULOS: tuple[Rotulo, ...] = (
    # Valor vigente após alteração: o maior peso. Numa repactuação/apostilamento a
    # fonte escreve "passa de R$ X para R$ Y", "altera o valor ... para R$ Y": o valor
    # vigente é Y, não X nem o maior dos dois. Como `classificar_grandeza` corta a
    # janela no último 'R$ <dígito>' anterior, ao classificar Y a janela começa logo
    # após 'R$ X' e contém o conector de alteração — então este rótulo casa SÓ para Y.
    # Peso acima de 90 para vencer o total declarado de X. Grandeza TOTAL: o valor
    # vigente do ajuste é o total corrente.
    #
    # O conector é restrito: 'para' sozinho é comum demais ('destinado para', 'para a
    # empresa'), então exige-se um verbo de alteração antes ('passa a ser R$', 'altera
    # para R$', 'reajusta para R$', 'majorado para R$'), ou a forma 'para o valor de
    # R$'/'para R$ Y, perfazendo' típica do apostilamento.
    Rotulo(_p(r"(?:passa(?:ndo|r[áa]?)?\s+a\s+ser|passa\s+a\s+ser|"
              r"alterad[oa]\s+para|altera(?:ndo|r)?\s+(?:o\s+valor\s+)?para|"
              r"reajustad[oa]\s+para|reajusta(?:ndo|r)?\s+para|"
              r"majorad[oa]\s+para|elevad[oa]\s+para|"
              r"para\s+o\s+(?:novo\s+)?valor\s+de)\s*(?:r\$)?\s*$"),
           TOTAL, True, 95),
    # A forma 'passa de R$ X para R$ Y': ao classificar Y, a janela é cortada logo após o
    # último 'R$ <dígito>' (o corte consome só o primeiro dígito), restando algo como
    # '0.000,00 para ' para X = 50.000,00. O conector
    # 'para' aparece no fim da janela, precedido apenas pelo resto do numeral de X (dígitos,
    # pontos, vírgulas, espaço) — nunca por outra palavra. `[\d.,\s]*para\s*$` casa esse
    # caso e nunca 'para a empresa ... R$' (que tem palavras depois de 'para').
    Rotulo(_p(r"(?:^|[\d.,\s])\s*(?:de\s+)?para\s*$"), TOTAL, True, 95),
    # Qualificadores: nomeiam a grandeza. Vencem genéricos independentemente de peso.
    Rotulo(_p(r"valor\s+mensal|mensalidade|por\s+m[eê]s|/m[eê]s|ao\s+m[eê]s"), MENSAL, True, 30),
    Rotulo(_p(r"valor\s+anual|por\s+ano|/ano|ao\s+ano|anualmente"), ANUAL, True, 30),
    Rotulo(_p(r"valor\s+unit[aá]rio|pre[çc]o\s+unit[aá]rio|valor\s+un\.?|unit[aá]rio"),
           UNITARIO, True, 25),
    # 'prestação' só é PARCELA quando qualifica um pagamento ('prestação mensal/parcelada',
    # 'em N prestações'): 'prestação de serviços' é OBJETO, não grandeza, e casá-la como
    # parcela tiraria do agregado o valor de um contrato de serviço comum. 'parcela'
    # continua suficiente sozinha.
    Rotulo(_p(r"parcela|presta[çc][aã]o\s+(?:mensal|[úu]nica|parcelad[ao]|do\s+valor)|"
              r"em\s+\d+\s+presta[çc][õo]es"), PARCELA, True, 20),
    Rotulo(_p(r"multa|penalidade|infra[çc][aã]o|san[çc][aã]o|corre[çc][aã]o\s+monet"),
           PENALIDADE, True, 15),
    Rotulo(_p(r"saldo\s+(?:remanescente|residual|contratual)?|saldo"), SALDO, True, 15),
    # 'valor total estimado' é o total do ajuste anunciado como estimativa — o caso do
    # registro de preços e da ata. Precisa vir antes de `valor estimado`, por ser mais
    # específico, e com o PESO do total: senão um `no valor de R$ X` genérico (peso 50)
    # venceria um `Valor Total Estimado R$ Y` muito maior (peso 40 do estimado solto),
    # publicando o valor menor. Mantém a grandeza ESTIMADO, que é a correta — é estimativa,
    # não total fechado —, mas com peso de total para a eleição.
    Rotulo(_p(r"valor\s+total\s+estimad[oa]|valor\s+global\s+estimad[oa]|"
              r"estimativa\s+(?:total|global)"), ESTIMADO, True, 88),
    Rotulo(_p(r"valor\s+estimado|estimativa|estimad[oa]"), ESTIMADO, True, 40),
    Rotulo(_p(r"valor\s+total|total\s+(?:global|geral|do\s+ajuste|do\s+contrato)"
              r"|valor\s+global|totalizando|total\s+totalizado"), TOTAL, True, 90),
    # `valor do contrato` e as variantes de instrumento: `VALOR DO CONTRATO:` é um rótulo
    # comum do extrato em prosa. Entra como qualificador, e com o peso do total:
    # `valor do contrato` nomeia a grandeza com a mesma clareza de `valor total`, e um
    # extrato que o traz está dizendo o valor do ajuste.
    Rotulo(_p(r"valor\s+d[oae]\s+(?:contrato|ajuste|termo|aditamento|instrumento|"
              r"conv[eê]nio|parceria|apostilamento|registro\s+de\s+pre[çc]os)"),
           TOTAL, True, 90),
    # Genéricos: só introduzem um número. Não nomeiam grandeza; grandeza fica total
    # quando um genérico é o único rótulo (é o significado esperado de "no valor de
    # R$ X" num extrato de contrato), mas um qualificador presente na janela o vence
    # sempre.
    #
    # Os pesos ficam abaixo dos qualificadores de total (90): num extrato que traz o
    # total do ajuste e o valor de uma nota de empenho, o total declarado tem de
    # ordenar acima de um pagamento parcial.
    Rotulo(_p(r"nota\s+de\s+empenho|empenho"), TOTAL, False, 60),
    Rotulo(_p(r"no\s+valor\s+de|no\s+importe\s+de|importe\s+de|montante\s+de|montante|"
              r"valor\s+de|import[aâ]ncia\s+de|quantia\s+de|pelo\s+valor\s+de|"
              r"custo\s+de|no\s+total\s+de"), TOTAL, False, 50),
)


# REGEX DE MOEDA E A JANELA DE CONTEXTO

# 'R$' seguido, mais adiante, do numeral brasileiro. Aceita a duplicação 'R$ R$'
# porque ela ocorre na fonte. O grupo `numero` é o numeral.
#
# A ordem das alternativas e a fronteira `(?!\d)` no fim evitam que um numeral sem
# separador de milhar seja lido truncado (ex.: 'R$ 2179,00' lido como R$ 217,00). A
# alternativa pontuada exige **ao menos um** grupo de milhar (`+` em vez de `*`), de
# modo que um numeral sem separador cai na segunda alternativa e é lido inteiro. A
# fronteira `(?!\d)` recusa casar um prefixo de número maior.
_RE_MOEDA = re.compile(
    r"R\$\s*(?:R\$\s*)?"                       # 'R$' ou 'R$ R$'
    r"(?P<numero>\d{1,3}(?:\.\d{3})+(?:,\d{2})?|\d+(?:,\d{2})?)"
    r"(?![.,]?\d)"
)

# Um 'R$' seguido de um numeral MALFORMADO: ponto decimal estrangeiro ('887,833.6'),
# dígitos a mais depois dos centavos ('12.345,6789') ou mais de um grupo de milhar
# fora do padrão ('1.234.5678'). `_RE_MOEDA` não casa esses (a fronteira
# `(?![.,]?\d)` recusa), de modo que não são lidos truncados. Este padrão só DETECTA
# o caso; não produz valor.
_RE_MOEDA_MALFORMADA = re.compile(
    r"R\$\s*(?:R\$\s*)?"
    r"\d{1,3}(?:[.,]\d+){2,}"                  # dois ou mais grupos [.,]dígitos: malformado
)

# O corte da janela: o ÚLTIMO 'R$ <dígito>' ANTES da ocorrência atual. Exige o
# dígito para que 'R$ R$' não trunque no primeiro 'R$' — que não é seguido de dígito.
#
# IGNORECASE é obrigatório: `classificar_grandeza` aplica este padrão sobre uma
# janela já passada por `.lower()`, onde 'R$' virou 'r$'. Sem o flag, o padrão nunca
# casa, a regra de corte não executa, e o rótulo de grandeza de um valor anterior vaza
# para o seguinte — um valor mensal passa a ser lido como total, ou o contrário.
_RE_MOEDA_ANTERIOR = re.compile(r"R\$\s*\d", re.IGNORECASE)

JANELA_ANTES = 80   # caracteres de contexto antes do 'R$'
JANELA_CONTEXTO = 60  # ± para o campo de auditoria `contexto`
JANELA_APOS = 160   # depois do numeral, onde vive o extenso entre parênteses

# Faixa de plausibilidade: recusa número que é claramente processo/protocolo lido
# como moeda. Conservadora — o piso é baixo (R$ 0,01) e o teto alto (R$ 100 bilhões)
# porque contratos municipais de grande vulto existem, e recusar por teto baixo
# perderia dado real.
VALOR_MINIMO = 0.01
VALOR_MAXIMO = 100_000_000_000.0


# CONVERSÃO DO NUMERAL BRASILEIRO


def para_float(numeral: str | None) -> float | None:
    """
    Converte numeral em convenção brasileira para float, ou None se malformado.

    Trata os formatos presentes na fonte: '60.090,94' (milhar+decimal),
    '60090,94' (só decimal), '60.090' (só milhar, sem centavos), '5500' (nu). Grupos
    de milhar mal formados — '12.34.567,00', resíduo de OCR ou digitação — são
    REJEITADOS (None) em vez de silenciosamente reinterpretados: tolerar isso deixaria
    um erro de digitação virar R$ 1,2 milhão.
    """
    if not numeral:
        return None
    s = numeral.strip()
    if not s:
        return None
    tem_virgula = "," in s
    tem_ponto = "." in s

    if tem_virgula:
        # vírgula é o separador decimal; ponto (se houver) é milhar.
        inteiro, _, decimal = s.rpartition(",")
        if not re.fullmatch(r"\d{2}", decimal):
            return None
        if tem_ponto and not _milhar_bem_formado(inteiro):
            return None
        inteiro = inteiro.replace(".", "")
        if not inteiro.isdigit():
            return None
        return float(f"{inteiro}.{decimal}")

    if tem_ponto:
        # sem vírgula: ponto é milhar (ex.: '60.090'); só aceita grupos bem formados.
        if not _milhar_bem_formado(s):
            return None
        return float(s.replace(".", ""))

    if s.isdigit():
        return float(s)
    return None


def _milhar_bem_formado(inteiro: str) -> bool:
    """Grupos de milhar: 1-3 dígitos, depois grupos de exatamente 3 ('1.234.567')."""
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{3})*", inteiro))


# NUMERAL POR EXTENSO — o segundo parser independente

_UNIDADES = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "onze": 11, "doze": 12, "treze": 13, "quatorze": 14, "catorze": 14,
    "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18, "dezenove": 19,
}
_DEZENAS = {
    "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
    "setenta": 70, "oitenta": 80, "noventa": 90,
}
_CENTENAS = {
    "cem": 100, "cento": 100, "duzentos": 200, "trezentos": 300, "quatrocentos": 400,
    "quinhentos": 500, "seiscentos": 600, "setecentos": 700, "oitocentos": 800,
    "novecentos": 900,
}
_ESCALAS = {
    "mil": 1_000,
    "milhao": 1_000_000, "milhoes": 1_000_000,
    "bilhao": 1_000_000_000, "bilhoes": 1_000_000_000,
}
_PALAVRAS_VALIDAS = set(_UNIDADES) | set(_DEZENAS) | set(_CENTENAS) | set(_ESCALAS) | {"e"}


def numero_por_extenso(frase: str | None, *, exigir_moeda: bool = False) -> float | None:
    """
    Interpreta um valor escrito em palavras (pt-BR); None se não for numeral.

    Suporta até bilhões e a parte de centavos ('... reais e noventa e quatro
    centavos'). Retorna None quando a frase não é um numeral.

    `exigir_moeda`: quando True, só aceita frases que mencionem 'reais'/'real'. É
    obrigatório na CONFERÊNCIA de valores — '(doze) meses' também é numeral em
    palavras e confirmaria por acidente um valor de 12. Fora da conferência (ex.: ler
    um extenso solto) pode ser False.
    """
    if not frase:
        return None
    limpa = _remover_acentos(frase).lower()
    tem_reais = bool(re.search(r"\breais?\b", limpa))
    if exigir_moeda and not tem_reais:
        return None

    # separa a parte de centavos, se houver 'e ... centavos'
    reais_txt = limpa
    centavos_txt = ""
    m = re.search(r"\bcentavos?\b", limpa)
    if m:
        antes = limpa[: m.start()]
        # 'X reais e Y centavos' → reais=antes-de-'reais', centavos=entre 'reais' e 'centavos'
        mr = re.search(r"\breais?\b", antes)
        if mr:
            reais_txt = antes[: mr.start()]
            centavos_txt = antes[mr.end():]
        else:
            reais_txt = ""
            centavos_txt = antes
    else:
        mr = re.search(r"\breais?\b", limpa)
        if mr:
            reais_txt = limpa[: mr.start()]

    reais = _soma_extenso(reais_txt)
    centavos = _soma_extenso(centavos_txt) if centavos_txt else 0

    if reais is None and centavos is None:
        return None
    if reais is None:
        reais = 0
    if centavos is None:
        centavos = 0
    if reais == 0 and centavos == 0 and not re.search(r"\bzero\b", limpa):
        # nenhuma palavra numérica reconhecida
        return None
    return float(reais) + float(centavos) / 100.0


def _soma_extenso(txt: str) -> int | None:
    """Soma as palavras numéricas de um trecho (sem 'reais'/'centavos')."""
    palavras = re.findall(r"[a-z]+", txt)
    palavras = [p for p in palavras if p in _PALAVRAS_VALIDAS]
    if not palavras:
        return None
    total = 0        # acumulado fechado (escalas já aplicadas)
    grupo = 0        # grupo corrente (< 1000) sendo montado
    viu_numero = False
    for p in palavras:
        if p == "e":
            continue
        if p in _UNIDADES:
            grupo += _UNIDADES[p]
            viu_numero = True
        elif p in _DEZENAS:
            grupo += _DEZENAS[p]
            viu_numero = True
        elif p in _CENTENAS:
            grupo += _CENTENAS[p]
            viu_numero = True
        elif p == "mil":
            grupo = (grupo or 1) * 1_000
            total += grupo
            grupo = 0
            viu_numero = True
        elif p in ("milhao", "milhoes", "bilhao", "bilhoes"):
            escala = _ESCALAS[p]
            total += (grupo or 1) * escala
            grupo = 0
            viu_numero = True
    total += grupo
    return total if viu_numero else None


def confere_extenso(
    valor: float | None, extenso: float | None, *, tolerancia: float = 0.02
) -> bool:
    """
    Compara numeral e extenso com a tolerância que a redação real exige.

    Devolve False — nunca None — quando não há extenso a comparar: o resultado é o
    campo booleano `confere` e entra na chave de ordenação da eleição, onde None se
    compararia diferente de False. A tolerância absoluta pequena absorve o arredondamento de
    centavos que a redação por extenso às vezes tem.
    """
    if valor is None or extenso is None:
        return False
    return abs(valor - extenso) <= tolerancia


def extenso_apos(apos: str | None) -> float | None:
    """
    Lê o valor por extenso escrito logo APÓS um numeral, entre parênteses.

    Recebe só a janela posterior. Quem compara numeral e extenso é `confere_extenso`;
    esta função apenas LÊ o extenso.

    Devolve o valor lido (não um booleano) para que a base distinga dois casos
    diferentes: a fonte não trazer extenso (ausência de evidência, None) e o extenso
    discordar do numeral (erro, contável).

    A janela posterior é TRUNCADA no próximo 'R$ <dígito>' antes de procurar o
    parêntese: sem isso, um valor sem extenso próprio ('R$ 16.200,00 Total Global
    R$ 169.200,00 (cento...)') captura o extenso do valor SEGUINTE e produz falsa
    discordância. O extenso só conta quando está antes do próximo valor.
    """
    if not apos:
        return None
    janela = apos
    corte = _RE_MOEDA_ANTERIOR.search(janela)
    if corte:
        janela = janela[:corte.start()]
    # o extenso costuma vir entre parênteses logo após o numeral
    m = re.search(r"\(([^)]*)\)", janela)
    if m:
        trecho = m.group(1)
    else:
        # sem parênteses dentro da janela truncada: não há extenso próprio
        return None
    return numero_por_extenso(trecho, exigir_moeda=True)


# CLASSIFICAÇÃO — decide a grandeza, isolada numa função


def classificar_grandeza(antes: str) -> tuple[str, str | None]:
    """
    Decide a GRANDEZA de um valor a partir do texto que o antecede.

    Duas regras:

      1. **A janela não atravessa outro 'R$ <dígito>' anterior.** Um rótulo antes de
         outro valor qualifica AQUELE valor, não este. Corta-se no último
         'R$ <dígito>' da janela (o que trata 'R$ R$' corretamente).
      2. **Qualificador vence genérico, independentemente de peso; entre iguais, o
         mais próximo do valor.** Um rótulo que NOMEIA a grandeza ('multa', 'valor
         mensal') prevalece sobre um que só a INTRODUZ ('no valor de'), mesmo tendo
         peso menor. Sem isso, 'no valor de' (genérico) venceria 'multa'
         (qualificador) e a multa entraria como total.

    A grandeza default é INDEFINIDO quando nenhum rótulo casa (não TOTAL: afirmar
    total sem rótulo infla o agregado). Quando só há genérico, a grandeza é a que o
    genérico carrega — TOTAL para 'no valor de', porque num extrato de contrato "no
    valor de R$ X" isolado É o total do ajuste.

    Returns:
        (grandeza, rotulo_casado|None).
    """
    janela = _remover_acentos(antes or "").lower()
    # regra 1: cortar no último 'R$ <dígito>' anterior dentro da janela
    cortes = list(_RE_MOEDA_ANTERIOR.finditer(janela))
    if cortes:
        janela = janela[cortes[-1].end():]

    qualificadores: list[tuple[int, int, Rotulo]] = []  # (posicao_do_fim, peso, rotulo)
    genericos: list[tuple[int, int, Rotulo]] = []
    for rot in _ROTULOS:
        # a última ocorrência do rótulo na janela é a mais próxima do valor
        ult = None
        for m in rot.padrao.finditer(janela):
            ult = m
        if ult is None:
            continue
        registro = (ult.end(), rot.peso, rot)
        (qualificadores if rot.qualificador else genericos).append(registro)

    pool = qualificadores if qualificadores else genericos
    if not pool:
        return INDEFINIDO, None

    # regra 2: entre os do pool, o mais próximo do valor (maior posição de fim).
    # Empate de posição é desempatado por peso (o mais confiável), e depois pela
    # ordem de declaração — determinístico.
    pool.sort(key=lambda r: (r[0], r[1]), reverse=True)
    _, _, vencedor = pool[0]
    return vencedor.grandeza, vencedor.padrao.pattern


# OCORRÊNCIAS E ELEIÇÃO


@dataclass(frozen=True)
class Ocorrencia:
    """Uma ocorrência monetária com sua evidência e classificação."""

    valor: float
    grandeza: str
    rotulo: str | None
    extenso: float | None
    confere: bool
    contexto: str
    peso: int
    posicao: int


def ocorrencias_monetarias(texto: str | None) -> list[Ocorrencia]:
    """
    Localiza TODAS as ocorrências de valor e classifica cada uma.

    Existe porque um extrato típico cita valor total, unitário, mensal e eventual
    multa. Tomar o primeiro 'R$' mistura grandezas incomparáveis. Cada ocorrência
    carrega:
      - `valor` (numeral convertido; ocorrências implausíveis são descartadas);
      - `grandeza`/`rotulo` de `classificar_grandeza` sobre a janela de 80 antes;
      - `extenso`/`confere` do extenso na janela posterior;
      - `contexto` (±60) e `posicao` para auditoria e eleição.

    As janelas saem do deslocamento REAL em caracteres de cada 'R$', não de um
    split: usar bordas de segmento faz a janela posterior atravessar a ocorrência
    seguinte e a janela anterior herdar dígitos do valor anterior quando a fronteira
    dos 80 caracteres corta um 'R$ <dígito>' pelo meio.
    """
    if not texto:
        return []
    ocs: list[Ocorrencia] = []
    for m in _RE_MOEDA.finditer(texto):
        valor = para_float(m.group("numero"))
        if valor is None or not (VALOR_MINIMO <= valor <= VALOR_MAXIMO):
            continue
        ini = m.start()
        antes = texto[max(0, ini - JANELA_ANTES):ini]
        apos = texto[m.end():m.end() + JANELA_APOS]
        contexto = _colapsar_espacos(
            texto[max(0, ini - JANELA_CONTEXTO):m.end() + JANELA_CONTEXTO]
        )
        grandeza, rotulo = classificar_grandeza(antes)
        extenso = extenso_apos(apos)
        confere = confere_extenso(valor, extenso)
        # o peso vem do rótulo casado, para a eleição; se não casou rótulo, peso 0.
        peso = _peso_do_rotulo(rotulo)
        ocs.append(Ocorrencia(valor, grandeza, rotulo, extenso, confere,
                              contexto, peso, ini))
    return ocs


def _peso_do_rotulo(padrao: str | None) -> int:
    if padrao is None:
        return 0
    for rot in _ROTULOS:
        if rot.padrao.pattern == padrao:
            return rot.peso
    return 0


@dataclass(frozen=True)
class ValorEleito:
    """O valor que representa a contratação, com toda a sua evidência."""

    valor: float
    grandeza: str
    rotulo: str | None
    extenso: float | None
    confere: bool
    contexto: str


def escolher_valor(ocorrencias: list[Ocorrencia]) -> ValorEleito | None:
    """
    Elege o valor que representa a contratação, entre as ocorrências CONCORRENTES.

    É aqui — e só aqui — que o PESO decide, fazendo o que lhe cabe: ordenar valores
    concorrentes do mesmo texto (qual 'R$' é o do ajuste). A chave de ordenação, na
    ordem: (1) maior peso do rótulo; (2) confirmação por extenso; (3) maior magnitude
    — em extratos com vários itens, o total supera as parcelas; (4) posição no texto
    (o primeiro, como desempate estável).

    Devolve None quando não há ocorrência plausível — o que acontece nos atos sem
    valor. None é a resposta correta, não uma lacuna a preencher.
    """
    if not ocorrencias:
        return None
    melhor = max(
        ocorrencias,
        key=lambda o: (o.peso, o.confere, o.valor, -o.posicao),
    )
    return ValorEleito(
        valor=melhor.valor,
        grandeza=melhor.grandeza,
        rotulo=melhor.rotulo,
        extenso=melhor.extenso,
        confere=melhor.confere,
        contexto=melhor.contexto,
    )


def valor_do_ajuste(texto: str | None) -> ValorEleito | None:
    """
    A porta de valor: o valor eleito de um texto, ou None se não há valor a declarar.

    Composição de `ocorrencias_monetarias` + `escolher_valor`. Existe como função
    nomeada para que o pipeline tenha um ponto único de entrada análogo à
    `porta_de_nome`.
    """
    return escolher_valor(ocorrencias_monetarias(texto))


# A GUARDA CONTRA O VOCABULÁRIO PUBLICADO
# Grandeza nova aqui, sem entrada no vocabulário, cairia numa soma que o consumidor
# acredita ser de valor contratual — então a divergência falha na importação.

GRANDEZAS: tuple[str, ...] = V.GRANDEZAS

_produzidas = {TOTAL, ESTIMADO, ANUAL, MENSAL, UNITARIO, PARCELA, PENALIDADE,
               SALDO, INDEFINIDO}
if _produzidas != set(V.GRANDEZAS):
    raise AssertionError(
        f"grandezas divergem do vocabulário: "
        f"produzidas-declaradas={sorted(_produzidas - set(V.GRANDEZAS))} "
        f"declaradas-produzidas={sorted(set(V.GRANDEZAS) - _produzidas)}"
    )
if GRANDEZAS_SOMAVEIS != V.GRANDEZAS_SOMAVEIS:
    raise AssertionError(
        f"GRANDEZAS_SOMAVEIS divergem: {GRANDEZAS_SOMAVEIS} != {V.GRANDEZAS_SOMAVEIS}"
    )
del _produzidas
