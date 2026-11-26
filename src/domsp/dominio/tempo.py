"""
tempo.py — extrai datas e prazo do ajuste: data de assinatura, início e fim de
vigência, e prazo em meses.

Datas: formato brasileiro ('30/04/2026') e por extenso ('30 de abril de 2026').
Prazo: número com extenso confirmatório ('12 (doze) meses'), período em dias
('pelo período de 90 dias'), ou intervalo de datas ('a partir de 17/10/2026 até
16/10/2027').

Prazo é normalizado em MESES. Dias são convertidos por convenção de 30 dias/mês
(aproximação declarada por ressalva). Quando início e fim existem mas o prazo não
foi declarado, o prazo é DERIVADO da diferença entre as datas (aritmética
verificável, com ressalva `prazo_derivado_das_datas`). Já `vigencia_fim` NUNCA é
derivada de início+prazo, porque somar meses a uma data depende de uma convenção
de calendário que a fonte não publicou; nesse caso `vigencia_fim` fica nula com a
ressalva `fim_nao_derivado_de_prazo`.

Restrições: só biblioteca padrão e `domsp.texto`; não importa `domsp.contrato`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

try:  # pragma: no cover
    from domsp import texto as _texto

    _remover_acentos = _texto.remover_acentos
    _colapsar_espacos = _texto.colapsar_espacos
except Exception:  # pragma: no cover
    def _remover_acentos(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "")
        return "".join(c for c in s if not unicodedata.combining(c))

    def _colapsar_espacos(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()


# Meses por extenso, comparados sobre a forma sem acento e minúscula ('março' e
# 'marco' são o mesmo). Abreviações não são reconhecidas.
_MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11,
    "dezembro": 12,
}
_MES_ALT = "|".join(_MESES)  # já sem acento


# Numérica: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY. Ano de 4 dígitos.
_RE_DATA_NUM = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b")

# Por extenso: '30 de abril de 2026'. O 'de' entre dia e mês pode faltar ('30 abril
# de 2026'); o 'de' antes do ano é constante. O ordinal do primeiro dia do mês
# ('1º de março de 2026') é aceito nas três marcas (º, o, °) que aparecem em texto
# extraído de HTML.
_RE_DATA_EXT = re.compile(
    r"\b(\d{1,2})[º°o]?\s+(?:de\s+)?(" + _MES_ALT + r")\s+de\s+(\d{4})\b",
    re.IGNORECASE,
)


def _data_valida(dia: int, mes: int, ano: int) -> date | None:
    """A data, se calendaricamente válida; None se não (ex.: 31 de fevereiro)."""
    try:
        return date(ano, mes, dia)
    except ValueError:
        return None


def datas(texto: str | None) -> list[date]:
    """
    Todas as datas do texto — numéricas e por extenso — calendaricamente válidas,
    na ordem em que aparecem. Datas impossíveis (31/02) são descartadas em silêncio,
    porque são erro de redação ou de OCR, não datas.
    """
    if not texto:
        return []
    achados: list[tuple[int, date]] = []
    for m in _RE_DATA_NUM.finditer(texto):
        d = _data_valida(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            achados.append((m.start(), d))
    sem_acento = _remover_acentos(texto).lower()
    for m in _RE_DATA_EXT.finditer(sem_acento):
        mes = _MESES.get(m.group(2))
        if mes:
            d = _data_valida(int(m.group(1)), mes, int(m.group(3)))
            if d:
                achados.append((m.start(), d))
    achados.sort(key=lambda x: x[0])
    return [d for _, d in achados]


def primeira_data(texto: str | None) -> date | None:
    """A primeira data válida do texto, ou None."""
    ds = datas(texto)
    return ds[0] if ds else None


# ═══════════════════════════════════════════════════════════════════════════
# PRAZO EM MESES
# ═══════════════════════════════════════════════════════════════════════════

DIAS_POR_MES = 30  # convenção declarada para converter dias em meses

# `(?<!\d)` impede que um número maior seja lido pelos últimos dígitos: sem ele, "1200 meses"
# casaria "200" e o prazo sairia seis vezes menor, sem erro visível.
_RE_MESES = re.compile(
    r"(?<!\d)(\d{1,3})\s*(?:\([^)]*\)\s*)?(?:\bmes(?:es)?\b|\bm[eê]s\b)", re.IGNORECASE
)
_RE_ANOS = re.compile(r"(?<!\d)(\d{1,2})\s*(?:\([^)]*\)\s*)?\banos?\b", re.IGNORECASE)
_RE_DIAS = re.compile(r"(?<!\d)(\d{1,4})\s*(?:\([^)]*\)\s*)?\bdias?\b", re.IGNORECASE)

# Marcadores de que o número de meses/anos/dias que vem A SEGUIR é o PRAZO do ajuste, e não
# uma duração qualquer citada no texto: sem restringir ao contexto, "comemoração de 93 anos"
# viraria prazo de 1.116 meses. A busca contextual tenta a janela logo após o marcador antes
# de cair na busca global.
_RE_ABERTURA_DE_PRAZO = re.compile(
    r"(?:prazo|vig[êe]ncia|per[íi]odo|vigorar[áa]?|dura[çc][ãa]o|"
    r"prorrogad[oa]\s+por|pelo\s+prazo\s+de|pelo\s+per[íi]odo\s+de)",
    re.IGNORECASE,
)
# Janela de caracteres após o marcador em que o número do prazo costuma aparecer.
_JANELA_DE_PRAZO = 60
# Teto de plausibilidade para prazo em ANOS: acima disso, um "N anos" fora de contexto de
# prazo é tratado como idade/aniversário citado no corpo do ato.
_MAX_ANOS_PLAUSIVEL = 60


@dataclass(frozen=True)
class Prazo:
    """Prazo normalizado em meses e como foi obtido."""

    meses: int | None
    ressalvas: tuple[str, ...]

    def __iter__(self):
        yield self.meses
        yield tuple(self.ressalvas)


def prazo_em_meses(texto: str | None) -> Prazo:
    """
    Normaliza o prazo do ajuste em MESES, a partir das formas reais da fonte.

    Reconhece, em ordem de preferência (a mais explícita primeiro):
      1. '... meses' — '12 (doze) meses', '24 meses'. O extenso entre parênteses é
         ignorado (é confirmação, não valor); registramos só o número.
      2. '... anos' — convertido a meses (×12), exato.
      3. '... dias' — 'pelo período de 90 dias'. Convertido por 30 dias/mês,
         APROXIMADO: guarda o inteiro mais próximo e a ressalva de que houve conversão.

    Retorna `meses=None` quando o texto não traz prazo em nenhuma dessas formas — o
    que inclui o intervalo de datas puro ('13/07 à 17/07'), tratado por
    `vigencia` (derivação a partir das datas), não aqui. Não inventa prazo onde não há.

    Returns:
        Prazo(meses, ressalvas).
    """
    if not texto:
        return Prazo(None, ())
    t = _remover_acentos(texto).lower()

    # Primeiro, procurar o prazo NO CONTEXTO de um marcador de prazo/vigência. Só a
    # janela logo após o marcador é inspecionada, o que impede que uma duração citada
    # no objeto ('comemoração de 93 anos') seja lida como prazo.
    for marcador in _RE_ABERTURA_DE_PRAZO.finditer(t):
        janela = t[marcador.end(): marcador.end() + _JANELA_DE_PRAZO]
        mm = _RE_MESES.search(janela)
        if mm:
            return Prazo(int(mm.group(1)), ())
        aa = _RE_ANOS.search(janela)
        if aa:
            return Prazo(int(aa.group(1)) * 12, ())
        dd = _RE_DIAS.search(janela)
        if dd:
            dias = int(dd.group(1))
            meses = round(dias / DIAS_POR_MES)
            ressalvas = ("prazo_convertido_de_dias",)
            if dias % DIAS_POR_MES != 0:
                ressalvas = ressalvas + ("prazo_em_dias_nao_multiplo_de_mes",)
            return Prazo(meses if meses > 0 else None, ressalvas)

    m = _RE_MESES.search(t)
    if m:
        return Prazo(int(m.group(1)), ())

    a = _RE_ANOS.search(t)
    if a:
        anos = int(a.group(1))
        # fora de contexto de prazo, um "N anos" implausivelmente grande é idade/aniversário,
        # não prazo contratual. Recusa em vez de publicar um prazo de centenas de meses.
        if anos > _MAX_ANOS_PLAUSIVEL:
            return Prazo(None, ("prazo_em_anos_implausivel_ignorado",))
        return Prazo(anos * 12, ())

    d = _RE_DIAS.search(t)
    if d:
        dias = int(d.group(1))
        meses = round(dias / DIAS_POR_MES)
        ressalvas = ("prazo_convertido_de_dias",)
        if dias % DIAS_POR_MES != 0:
            ressalvas = ressalvas + ("prazo_em_dias_nao_multiplo_de_mes",)
        return Prazo(meses if meses > 0 else None, ressalvas)

    return Prazo(None, ())


# ═══════════════════════════════════════════════════════════════════════════
# VIGÊNCIA E A DERIVAÇÃO CONTROLADA DO PRAZO
# ═══════════════════════════════════════════════════════════════════════════

_RE_INICIO = re.compile(
    r"(?:a\s+partir\s+de|com\s+in[ií]cio\s+em|vig[êe]ncia\s+a\s+partir\s+de|"
    r"in[ií]cio\s+(?:de\s+vig[êe]ncia\s+)?em|a\s+contar\s+de|com\s+efeitos\s+a\s+partir\s+de)"
    r"\s+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}|\d{1,2}\s+(?:de\s+)?[a-zç]+\s+de\s+\d{4})",
    re.IGNORECASE,
)
_RE_FIM = re.compile(
    r"(?:at[ée]|ao\s+final\s+em|com\s+t[ée]rmino\s+em|t[ée]rmino\s+em|"
    r"encerrando(?:-se)?\s+em|vig[êe]ncia\s+at[ée])"
    r"\s+(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}|\d{1,2}\s+(?:de\s+)?[a-zç]+\s+de\s+\d{4})",
    re.IGNORECASE,
)
_RE_ASSINATURA = re.compile(
    r"(?:assinat[ura]*|assinad[oa]|celebrad[oa]|firmad[oa]|data\s+da\s+assinatura|"
    r"aos?\s+)\D{0,40}?(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}|\d{1,2}\s+(?:de\s+)?[a-zç]+\s+de\s+\d{4})",
    re.IGNORECASE,
)


def _primeira_data_do_trecho(trecho: str | None) -> date | None:
    return primeira_data(trecho)


@dataclass(frozen=True)
class Vigencia:
    """Datas de vigência, assinatura, prazo em meses, e as ressalvas do que foi derivado."""

    data_assinatura: date | None
    vigencia_inicio: date | None
    vigencia_fim: date | None
    prazo_meses: int | None
    ressalvas: tuple[str, ...]


def vigencia(texto: str | None) -> Vigencia:
    """
    Extrai assinatura, início e fim de vigência, e o prazo em meses.

    A derivação segue a decisão declarada no cabeçalho do módulo:
      • se a fonte não declara prazo MAS declara início e fim, o prazo é DERIVADO da
        diferença entre as datas (aritmética verificável) e a ressalva
        `prazo_derivado_das_datas` o registra;
      • a data de fim NUNCA é derivada de início+prazo. Se só há início e prazo,
        `vigencia_fim` fica nula e `fim_nao_derivado_de_prazo` o diz.

    `data_assinatura`, quando não vem de um rótulo explícito de assinatura, cai para
    a primeira data do texto. É uma
    HEURÍSTICA marcada como tal (`assinatura_por_primeira_data`).
    """
    ressalvas: list[str] = []
    if not texto:
        return Vigencia(None, None, None, None, ())

    m_ini = _RE_INICIO.search(texto)
    inicio = _primeira_data_do_trecho(m_ini.group(1)) if m_ini else None

    m_fim = _RE_FIM.search(texto)
    fim = _primeira_data_do_trecho(m_fim.group(1)) if m_fim else None

    m_ass = _RE_ASSINATURA.search(texto)
    assinatura = _primeira_data_do_trecho(m_ass.group(1)) if m_ass else None
    if assinatura is None:
        assinatura = primeira_data(texto)
        if assinatura is not None:
            ressalvas.append("assinatura_por_primeira_data")

    prazo = prazo_em_meses(texto)
    prazo_meses = prazo.meses
    ressalvas.extend(prazo.ressalvas)

    # derivação controlada do prazo a partir das datas
    if prazo_meses is None and inicio is not None and fim is not None:
        derivado = _meses_entre(inicio, fim)
        if derivado is not None:
            prazo_meses = derivado
            ressalvas.append("prazo_derivado_das_datas")

    # a data de fim NÃO é derivada de início+prazo — decisão declarada
    if fim is None and inicio is not None and prazo_meses is not None:
        ressalvas.append("fim_nao_derivado_de_prazo")

    # sanidade: fim antes do início é redação/OCR errado; não corrige, só assinala
    if inicio is not None and fim is not None and fim < inicio:
        ressalvas.append("vigencia_fim_anterior_ao_inicio")

    return Vigencia(assinatura, inicio, fim, prazo_meses, tuple(ressalvas))


def do_formulario(
    *,
    assinatura: str | None = None,
    inicio: str | None = None,
    fim: str | None = None,
    prazo: str | None = None,
    prazo_unidade: str | None = None,
) -> Vigencia:
    """
    Os quatro valores de tempo a partir dos CAMPOS do formulário, não da prosa.

    Recebe valores soltos, e não o `Formulario`, porque este módulo não pode
    importar `formulario` (restrição declarada no cabeçalho). Quem faz a ponte é o chamador,
    que conhece as chaves do formulário.

    O prazo vem em dois campos — número (`12`) e unidade (`Mês`) — porque
    concatená-los numa string gravaria `'12Mês'`, inutilizável para cálculo. Aqui
    os dois são reunidos num texto e submetidos ao mesmo `prazo_em_meses` que lê a
    prosa.

    Não há queda para heurística: se o campo não traz data interpretável, o
    resultado é nulo e a ressalva `campo_do_formulario_ilegivel` o registra.
    """
    ressalvas: list[str] = []

    def _data(bruto: str | None, nome: str) -> date | None:
        if not bruto or not bruto.strip():
            return None
        d = primeira_data(bruto)
        if d is None:
            ressalvas.append(f"campo_do_formulario_ilegivel:{nome}")
        return d

    data_assinatura = _data(assinatura, "data_assinatura")
    vigencia_inicio = _data(inicio, "vigencia_inicio")
    vigencia_fim = _data(fim, "vigencia_fim")

    prazo_meses: int | None = None
    if prazo and prazo.strip():
        p = prazo_em_meses(f"{prazo} {prazo_unidade or ''}")
        prazo_meses = p.meses
        ressalvas.extend(p.ressalvas)
        if prazo_meses is None:
            ressalvas.append("campo_do_formulario_ilegivel:prazo")

    # Mesma assimetria declarada no cabeçalho: prazo derivado de duas datas é contar;
    # data de fim derivada de prazo seria escolher uma convenção de calendário.
    if prazo_meses is None and vigencia_inicio and vigencia_fim:
        prazo_meses = _meses_entre(vigencia_inicio, vigencia_fim)
        if prazo_meses is not None:
            ressalvas.append("prazo_derivado_das_datas")
    if vigencia_fim is None and vigencia_inicio and prazo_meses is not None:
        ressalvas.append("fim_nao_derivado_de_prazo")

    return Vigencia(
        data_assinatura=data_assinatura,
        vigencia_inicio=vigencia_inicio,
        vigencia_fim=vigencia_fim,
        prazo_meses=prazo_meses,
        ressalvas=tuple(dict.fromkeys(ressalvas)),
    )


def _meses_entre(inicio: date, fim: date) -> int | None:
    """
    Meses inteiros entre duas datas, para derivar prazo de vigência declarada.

    Conta meses de calendário e arredonda pelo resto em dias. A derivação é
    APROXIMADA. Regra: meses cheios de calendário entre as datas, mais um mês se o
    resto em dias passar de meio mês. 'a partir de 17/10/2026 até 16/10/2027' → 12
    (véspera do aniversário conta como o ano cheio); '01/01 a 31/12' → 12 (faltando
    1 dia para o ano, arredonda para cima). Devolve None se o fim é anterior ao início.
    """
    if fim < inicio:
        return None
    meses = (fim.year - inicio.year) * 12 + (fim.month - inicio.month)
    resto_dias = fim.day - inicio.day + 1  # +1: a vigência inclui o dia final
    if resto_dias >= DIAS_POR_MES // 2:
        meses += 1
    elif resto_dias < 0:
        meses -= 1
    return meses if meses > 0 else (0 if fim >= inicio else None)
