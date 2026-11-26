"""
parte.py — identifica a parte contratada (razão social) e o documento fiscal
(CNPJ/CPF), e constrói a chave de agrupamento usada para medir concentração de
fornecedores.

Três peças, cada uma num ponto só do código:

  1. **porta de nome** (`porta_de_nome`): valida/limpa toda cadeia candidata a
     razão social — venha do campo do formulário, de rótulo em prosa
     ("CONTRATADA:") ou de span de modelo. Devolve `(nome_limpo|None, ressalvas)`.
  2. **documento fiscal** (`classificar_documento`): CNPJ/CPF com dígito
     verificador módulo 11, devolvendo `tipo` e `valido` separados — um documento bem formado cujo DV não fecha é dado publicado, mas
     não serve para agrupar.
  3. **chave de agrupamento** (`chave_de_agrupamento`): raiz do CNPJ quando o
     documento existe e o DV fechou, nome normalizado quando não, sempre
     prefixada por 'cnpj:' ou 'nome:'.

As mesmas classes de não-nome (a a h: papel contratual sozinho, signatário,
identificador administrativo, conectivo pendurado, cauda com número do ajuste,
qualificação pendurada, quebra de linha e forma que não é de nome) são verificadas
para o candidato de qualquer camada de origem; por isso a validação é
concentrada numa porta única em vez de replicada por camada.

LIMITAÇÃO DECLARADA: a chave por raiz de CNPJ reúne matriz e filiais (mesmos 8
primeiros dígitos), mas um grupo econômico com CNPJs de raízes distintas continua
fragmentado. Qualquer indicador de concentração (HHI, Gini) calculado a partir
desta chave é um PISO, não o valor real.

Restrições: só biblioteca padrão, `domsp.texto` e `contrato.vocabulario`. As
ressalvas são DADO (acompanham o valor produzido), não log.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..contrato import vocabulario as V

# Dependência interna de normalização de texto. Importada de forma tolerante para que
# este módulo funcione isoladamente caso `domsp.texto` não esteja disponível — o
# fallback replica a mesma semântica (minúscula, sem acento, espaço colapsado).
# Como `normalizar_chave` é minúscula, todas as listas e padrões deste módulo
# comparam em minúscula.
try:  # pragma: no cover
    from domsp import texto as _texto

    _normalizar_chave = _texto.normalizar_chave
    _colapsar_espacos = _texto.colapsar_espacos
except Exception:  # pragma: no cover
    def _colapsar_espacos(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _normalizar_chave(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "")
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"\s+", " ", s).strip()
        return s.lower().strip(" :.-–—")


# ═══════════════════════════════════════════════════════════════════════════
# VOCABULÁRIO DE RECUSA — os motivos que a porta grava na ressalva
# ═══════════════════════════════════════════════════════════════════════════
# Cada constante é um motivo nomeado, não texto livre: a ressalva é um valor de
# um conjunto fechado, para que recusas por classe possam ser contadas.

RECUSA_VAZIO = "nome_vazio"
RECUSA_PAPEL = "papel_contratual_sozinho"          # classe (a)
RECUSA_SIGNATARIO = "signatario_nao_e_parte"       # classe (b)
RECUSA_IDENTIFICADOR = "identificador_administrativo"  # classe (c)
RECUSA_CONECTIVO = "conectivo_pendurado_removido"  # classe (d) — corrige, não recusa
RECUSA_CAUDA = "cauda_de_ajuste_removida"          # classe (e) — corrige, não recusa
RECUSA_QUALIFICACAO = "qualificacao_removida"      # classe (f) — corrige, não recusa
RECUSA_LINHA = "corte_na_quebra_de_linha"          # classe (g) — corrige, não recusa
RECUSA_CURTO = "nome_curto_demais"
RECUSA_LADO_PUBLICO = "lado_publico_nao_e_contratada"
RECUSA_LIXO = "forma_nao_e_de_nome"                 # classe (h) — minúscula, rótulo, endereço


# ═══════════════════════════════════════════════════════════════════════════
# AS LISTAS — vocabulário fechado, cada entrada com sua justificativa
# ═══════════════════════════════════════════════════════════════════════════

# Classe (a): papéis contratuais. Quando o nome candidato é SÓ um destes, não há
# razão social — é o rótulo do papel que sobrou. Comparado sobre a forma
# normalizada (sem acento, maiúsculo), então 'Donatária' e 'DONATÁRIA' são o mesmo.
# CRÍTICO: recusa só quando o papel é o nome INTEIRO. 'DONATÁRIA ACME LTDA' NÃO é
# recusado — o papel é prefixo e há nome depois.
_PAPEIS = frozenset({
    "contratada", "contratado", "contratante",
    "donataria", "donatario", "donataria e donatario",
    "permissionaria", "permissionario",
    "concessionaria", "concessionario",
    "compromissaria", "compromissario",
    "convenente", "participe", "participes",
    "outorgada", "outorgado", "outorgante",
    "cessionaria", "cessionario", "cedente",
    "locadora", "locadoro", "locataria", "locatario",
    "fornecedora", "fornecedor", "beneficiaria", "beneficiario",
    "adjudicataria", "adjudicatario",
})

# Classe (b): tratamentos de pessoa física signatária. O span traz uma pessoa que
# assina pelo ente, não a parte jurídica. Detectado por prefixo de tratamento
# seguido de nome — 'SR. ', 'SRA. ', 'DR. '. Só o PREFIXO decide; uma pessoa
# nomeada nunca é a parte contratada nesta base.
_TRATAMENTOS = (
    "sr", "sra", "srta", "dr", "dra", "sr(a)", "exmo", "exma",
    "ilmo", "ilma", "prof", "profa", "eng", "enga", "arq",
)
_RE_SIGNATARIO = re.compile(
    r"^(?:" + "|".join(re.escape(t) for t in _TRATAMENTOS) + r")\.?\s+\w",
)

# Classe (c): identificador administrativo. Número de processo/protocolo/ajuste
# travestido de nome. A heurística: a string tem ao menos um componente numérico E
# todos os componentes são numéricos ou siglas curtas em caixa alta (≤4 caracteres).
# Um nome real que contenha número ('SUPERMERCADO 3 IRMÃOS') tem palavras longas e
# não casa. Exemplos que ISTO recusa: '37/2024-SGM', '050.SEME.2026'.
_RE_SO_ID = re.compile(r"^[0-9/.\-]+$")           # '37/2024', '050.2026'
_RE_TEM_DIGITO = re.compile(r"\d")

# Classe (d): conectivos que podem ficar pendurados na borda depois de o corte das
# partes separar o polo público do privado. Removidos do FIM, repetidamente até
# estabilizar — porque remover um pode expor outro ('ACME COMERCIO E DE' →
# 'ACME COMERCIO E' → 'ACME COMERCIO' exige duas passadas).
_CONECTIVOS_FINAIS = ("e", "de", "da", "do", "das", "dos", "e a", "e o")

# Classe (e): cauda com número do ajuste grudada no fim. 'ACME LTDA - CONTRATO
# 123/2024', 'ACME LTDA, PROCESSO 6011.2024/0003602-4'. O padrão casa um separador
# (traço, vírgula, 'nº') seguido de uma palavra de ajuste e um número.
#
# A cauda não pode cortar um nome que contenha a palavra no meio. Um padrão como
# `.*\d.*$` não serve: a alternância casa a primeira ocorrência da palavra e `.*$`
# engole o resto, de modo que 'CASA DO EMPENHO 10 LTDA' viraria 'CASA'. A delimitação
# usada é `[\d./\s-]{0,40}$`: depois do primeiro dígito só podem vir dígitos, barra,
# ponto, espaço ou traço, e no máximo quarenta caracteres até o fim. Razão social que
# siga a palavra com outra palavra alfabética não é cauda, é nome.
_RE_CAUDA = re.compile(
    r"\s*[-–,;]?\s*"
    r"\b(?:CONTRATO|ADITIVO|ATA|PROCESSO|PROTOCOLO|TERMO|CONVENIO|EMPENHO|AJUSTE|"
    r"N[º°]\.?|N[O]\.|SEI|SIMPROC|EDITAL)\b"
    r"[\s:n°ºNO.-]*\d[\d./\s-]{0,40}$",
    re.IGNORECASE,
)

# Classe (f): qualificação civil/societária pendurada no fim do nome.
#
# O extrator da camada 2 (`formulario.contratada_da_prosa`) corta a prosa no
# documento fiscal, e o que sobra é o nome MAIS a abertura da qualificação:
#
#     'CONTRATADA: Alfa Climatização Ltda-ME, inscrita no CNPJ sob nº …'
#       → 'Alfa Climatização Ltda-ME, inscrita no'
#
# Sem aparar, a qualificação entra no nome — e, se usada como gabarito de treino,
# ensinaria o modelo a reproduzi-la.
#
# Dois padrões, não um, porque o separador protege contra cortar nome legítimo mas
# não pode ser exigido sempre:
#
# * `_RE_QUALIF_COM_SEPARADOR` exige vírgula, ponto-e-vírgula ou travessão antes. É
#   o que impede que `ASSOCIAÇÃO DE PORTADORES DE DEFICIÊNCIA` — nome de entidade do
#   terceiro setor — seja cortada em `PORTADORES`.
# * `_RE_QUALIF_SEM_SEPARADOR` dispensa o separador porque `inscrita`, `doravante`
#   e `neste ato` são particípio e locução adverbial que não ocorrem dentro de
#   razão social, e a fonte às vezes omite a vírgula.
#
# Ambos ancorados no FIM e aplicados até estabilizar, porque a fonte empilha:
# '…, inscrita no CNPJ …, doravante designada CONTRATADA'.
_RE_QUALIF_SEM_SEPARADOR = re.compile(
    r"\s*[-–,;]?\s*\b(?:inscrit[ao]s?|doravante|neste\s+ato)\b.*$",
    re.IGNORECASE,
)
_RE_QUALIF_COM_SEPARADOR = re.compile(
    r"\s*[-–,;]\s*"
    r"(?:portador\w*|representad\w*|com\s+sede|sediad\w*|estabelecid\w*|"
    r"C\.?N\.?P\.?J\.?|C\.?P\.?F\.?|inscri[cç][aã]o)\b.*$",
    re.IGNORECASE,
)

# Marcadores de que um segmento é o lado PÚBLICO (Administração), não a
# contraparte. Conservador de propósito — só termos inequívocos de ente público
# municipal/estadual.
#
# Recusar `municipio de sao paulo` em qualquer posição recusaria a **PRODAM** —
# cuja razão social é "EMPRESA DE TECNOLOGIA DA INFORMAÇÃO E COMUNICAÇÃO DO
# MUNICÍPIO DE SÃO PAULO - PRODAM-SP S/A". Ela é sociedade de economia mista,
# contratada legítima do Município.
#
# Duas regras, ambas gerais e não casos especiais:
#
# 1. `municipio de sao paulo` e `prefeitura` só recusam quando **iniciam** o nome.
#    "MUNICÍPIO DE SÃO PAULO, por meio da SECRETARIA..." é o polo público; "... DO
#    MUNICÍPIO DE SÃO PAULO" é genitivo dentro de uma razão social.
# 2. **sufixo societário absolve.** Um nome que termina em S/A, LTDA, EIRELI ou
#    congênere é pessoa jurídica de direito privado — inclusive a empresa
#    estatal, que é privada na forma e contratável. A Administração direta não
#    usa sufixo societário, então o sufixo é evidência positiva de
#    contratabilidade. Isso absolve a PRODAM, a SPTrans, a SPObras e qualquer
#    estatal sem precisar nomeá-las.
_RE_LADO_PUBLICO_EM_QUALQUER_POSICAO = re.compile(
    r"\b(?:secretaria municipal|secretaria estadual|"
    r"governo do estado|camara municipal|"
    r"tribunal de contas|subprefeitura|gabinete do prefeito)\b"
)
_RE_LADO_PUBLICO_NO_INICIO = re.compile(
    r"^(?:prefeitura|municipio de sao paulo|estado de sao paulo|uniao)\b"
)

# Classe (h): formas que não são de razão social e que as classes anteriores não
# pegam. Três famílias:
#
#   1. RÓTULO/QUALIFICAÇÃO que sobrou como palavra inteira — 'inscrita',
#      'Tipo de Pessoa'. Vêm de corte de campo mal posicionado.
#   2. ENDEREÇO — 'Rua …', 'Avenida …', 'Situada à', 'CEP'. A janela da âncora às
#      vezes alcança o endereço do estabelecimento.
#   3. FRAGMENTO DE ÓRGÃO solto — siglas e nomes de pasta: 'SIURB', 'MEIO AMBIENTE',
#      'Segurança Pública', 'Licenciamento'.
#
# As famílias 1 e 2 recusam quando o termo aparece em QUALQUER posição, porque uma
# razão social não contém 'inscrita' nem 'Rua' no meio. A família 3 é diferente:
# 'MEIO AMBIENTE' e 'LICENCIAMENTO' podem ocorrer DENTRO de nomes legítimos — por
# exemplo 'ALFA … Esporte e Meio Ambiente'. Por isso o
# fragmento de órgão só recusa quando é o nome INTEIRO ou o INÍCIO dele, nunca
# quando está embutido.
_RE_LIXO_EM_QUALQUER_POSICAO = re.compile(
    r"\b(?:inscrit[ao]|tipo de pessoa|situad[ao]|cep|logradouro)\b"
    r"|\brua\b|\bavenida\b|\balameda\b|\btravessa\b"
)
_RE_FRAGMENTO_DE_ORGAO = re.compile(
    r"^(?:siurb|svma|smul|smsub|smped|licenciamento|"
    r"meio ambiente|seguranca publica|zeladoria)\b"
)
_RE_COMECA_MINUSCULA = re.compile(r"^[a-zà-ÿ]")
# Nome que, depois de toda a limpeza, é SÓ um sufixo societário ('LTDA', 'S/A') ou SÓ
# uma variação de rótulo de documento ('CNPJ', 'C.N.P.J', 'CPF'). Não é razão social:
# é o que sobrou quando o corte perdeu o nome e guardou o rótulo. Testado sobre a chave
# com a pontuação dos pontos removida, para apanhar 'C.N.P.J' tanto quanto 'CNPJ'.
_RE_SO_ROTULO_DOC = re.compile(r"^(?:cnpj|cpf|c\s*n\s*p\s*j|c\s*p\s*f|mf|rg|rne)$")

# Sufixos societários que a normalização de nome remove para agrupar. NÃO alteram o
# nome de exibição (`contractor_name`); só a chave de agrupamento. A grafia
# publicada é preservada — a fidelidade à fonte é exigência do escopo do projeto.
_SUFIXOS_SOCIETARIOS = (
    "LTDA", "LTDA.", "S/A", "S.A", "S.A.", "SA", "EPP", "ME", "MEI",
    "EIRELI", "S/S", "SS", "CIA", "CIA.", "& CIA", "EI",
)

# Tamanho mínimo de nome aceitável. Abaixo disto o candidato é tratado como
# sigla solta ou fragmento. Conservador: 3 caracteres (existem MEIs com nome curto).
_MIN_NOME = 3


# ═══════════════════════════════════════════════════════════════════════════
# A PORTA ÚNICA
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ResultadoNome:
    """
    O que a porta devolve. `nome` é None quando o candidato foi recusado por
    inteiro; `ressalvas` sempre lista o que foi recusado OU corrigido, para que a
    base registre a decisão ao lado do dado.
    """

    nome: str | None
    ressalvas: tuple[str, ...]
    #: O documento fiscal que ancorou este nome, quando a camada que o produziu foi a
    #: âncora por documento. `None` nas camadas 1 e 2, que localizam o nome por rótulo e
    #: não por número. Existe para que o pipeline grave como documento da contratada o
    #: número que a identificou, e não o primeiro documento válido do texto.
    documento: str | None = None

    def __iter__(self):
        # Permite `nome, ressalvas = porta_de_nome(...)`.
        yield self.nome
        yield tuple(self.ressalvas)


def porta_de_nome(
    candidato: str | None, *, lado_publico_conhecido: str | None = None
) -> ResultadoNome:
    """
    A porta única por onde passa todo nome candidato a parte contratada.

    É a mesma função para as três camadas (formulário, rótulo em prosa, span do
    modelo). Devolve `(nome_limpo|None, ressalvas)`. Recusar devolvendo None com a
    ressalva nomeada — em vez de simplesmente descartar — é o que permite medir a
    taxa de recusa por classe e detectar regressão.

    A ordem das verificações não é arbitrária. Corrigir (tirar cauda, tirar
    conectivo) vem ANTES de recusar por conteúdo, porque a cauda pode ser o que faz
    a string parecer um identificador; e recusar o vazio vem primeiro porque
    normalizar '' não tem o que decidir.

    Args:
        candidato: a cadeia bruta vinda de qualquer camada.
        lado_publico_conhecido: quando as camadas determinísticas já sabem qual é o
            polo público do ajuste, passá-lo aqui impede que a própria
            Administração seja devolvida como contratada. É contexto, não regra nova.

    Returns:
        ResultadoNome(nome, ressalvas). Desempacotável como tupla.
    """
    ressalvas: list[str] = []

    if not candidato or not candidato.strip():
        return ResultadoNome(None, (RECUSA_VAZIO,))

    # (g) A primeira quebra de linha é fronteira dura: a quebra separa CAMPOS, então
    # nenhuma razão social a atravessa. Precisa vir
    # antes de `_colapsar_espacos`, que transforma `\n` em espaço e apaga a
    # informação. A camada 3 pode devolver
    # `'ALFA GAMES LTDA\nTipo de Pessoa\nJurídica\nCPF /CNPJ/ RNE'` porque sua
    # janela de caracteres atravessa as linhas do formulário renderizado.
    if "\n" in candidato or "\r" in candidato:
        primeira = re.split(r"[\r\n]", candidato, maxsplit=1)[0]
        if primeira.strip():
            candidato = primeira
            ressalvas.append(RECUSA_LINHA)

    nome = _colapsar_espacos(candidato)

    # (e) cauda com número do ajuste — corrige, repetindo enquanto houver cauda,
    # porque a fonte às vezes empilha 'ACME LTDA - CONTRATO 1 - PROCESSO 2'.
    nova = _RE_CAUDA.sub("", nome).strip(" -–,;")
    if nova and nova != nome:
        anterior = None
        while nova != anterior:
            anterior = nova
            nova = _RE_CAUDA.sub("", nova).strip(" -–,;")
        if nova:
            nome = nova
            ressalvas.append(RECUSA_CAUDA)

    # (f) qualificação civil/societária pendurada no fim — corrige, repetindo até
    # estabilizar, porque a fonte empilha ', inscrita no CNPJ …, doravante designada …'.
    # Vem ANTES do conectivo porque remover a qualificação pode expor um conectivo:
    # 'ACME E, inscrita no' → 'ACME E' → 'ACME'.
    qualificado = nome
    anterior = None
    while qualificado != anterior:
        anterior = qualificado
        qualificado = _RE_QUALIF_SEM_SEPARADOR.sub("", qualificado).strip(" -–,;")
        qualificado = _RE_QUALIF_COM_SEPARADOR.sub("", qualificado).strip(" -–,;")
    # Só aceita a aparação se ela não zerou o nome: um span que É só qualificação não
    # deve virar vazio aqui — deve seguir e ser recusado pelas classes seguintes, que
    # nomeiam a razão da recusa. Aparar até o vazio esconderia essa razão.
    if qualificado and qualificado != nome:
        nome = qualificado
        ressalvas.append(RECUSA_QUALIFICACAO)

    # (d) conectivo pendurado no fim — corrige, repetindo até estabilizar.
    nome_sem_conectivo = _tirar_conectivo_final(nome)
    if nome_sem_conectivo != nome:
        nome = nome_sem_conectivo
        ressalvas.append(RECUSA_CONECTIVO)

    nome = nome.strip(" -–,;.")
    if not nome:
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_VAZIO]))

    chave = _normalizar_chave(nome)

    # (a) papel contratual sozinho — o nome inteiro é um papel, sem razão social.
    if chave in _PAPEIS:
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_PAPEL]))

    # (b) signatário — tratamento de pessoa física. Não se aplica quando o nome
    # termina em sufixo societário: "DR ALFA EXPRESS COMERCIO LTDA" é razão social de
    # pessoa jurídica cujo nome fantasia começa por "DR", não um signatário "Dr. …".
    # O sufixo societário é a evidência positiva que distingue os dois.
    if _RE_SIGNATARIO.match(chave) and not _tem_sufixo_societario(nome):
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_SIGNATARIO]))

    # (c) identificador administrativo.
    if _eh_identificador(chave):
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_IDENTIFICADOR]))

    # lado público — a Administração não é a contratada. O sufixo societário absolve:
    # ver a nota longa em `_RE_LADO_PUBLICO_*`. `_tem_sufixo_societario` é consultado
    # antes das duas buscas para que a absolvição seja explícita e não implícita numa
    # negação embutida no padrão.
    if not _tem_sufixo_societario(nome):
        if _RE_LADO_PUBLICO_EM_QUALQUER_POSICAO.search(chave):
            return ResultadoNome(None, tuple(ressalvas + [RECUSA_LADO_PUBLICO]))
        if _RE_LADO_PUBLICO_NO_INICIO.search(chave):
            return ResultadoNome(None, tuple(ressalvas + [RECUSA_LADO_PUBLICO]))
    if lado_publico_conhecido and _normalizar_chave(lado_publico_conhecido) == chave:
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_LADO_PUBLICO]))

    # (h) formas que não são de razão social: rótulo/qualificação solta, endereço,
    # fragmento de órgão, ou início em minúscula. Vem depois das correções (a cauda e
    # a qualificação já foram tiradas) e depois do lado público (que tem ressalva
    # própria). O início minúscula é testado sobre o NOME publicado, não sobre a chave
    # — a chave já é minúscula —, porque razão social nesta fonte é caixa alta ou
    # Title-Case, e começar em minúscula denuncia fragmento de prosa corrida.
    if _RE_COMECA_MINUSCULA.match(nome.strip()):
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_LIXO]))
    if _RE_LIXO_EM_QUALQUER_POSICAO.search(chave):
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_LIXO]))
    if _RE_FRAGMENTO_DE_ORGAO.match(chave):
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_LIXO]))
    # nome que é SÓ sufixo societário ('LTDA') ou SÓ rótulo de documento ('CNPJ',
    # 'C.N.P.J') — sobrou o rótulo, perdeu-se o nome. `chave` sem pontos de abreviação
    # para que 'c.n.p.j' colapse em 'cnpj'.
    chave_sem_ponto = re.sub(r"\.", "", chave)
    if _tem_sufixo_societario(nome) and chave.replace(" ", "") in {
        _normalizar_chave(s).rstrip(" .").replace(" ", "") for s in _SUFIXOS_SOCIETARIOS
    }:
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_LIXO]))
    if _RE_SO_ROTULO_DOC.match(chave_sem_ponto.strip()):
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_LIXO]))

    # Pontuação de separação pendurada no fim ('ACME LTDA //', 'ACME LTDA -'). É resíduo do
    # recorte entre campos, não parte do nome, e sobrevive às classes anteriores porque elas
    # procuram palavra, não símbolo.
    nome = re.sub(r"[\s/\\|,;:.\-–]+$", "", nome).strip()
    if not nome:
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_VAZIO]))
    chave = _normalizar_chave(nome)

    # tamanho mínimo — depois das correções, para não recusar por causa de uma cauda
    # que já foi tirada. `chave` é minúscula e sem acento (API de normalizar_chave).
    letras = re.sub(r"[^a-z]", "", chave)
    if len(letras) < _MIN_NOME:
        return ResultadoNome(None, tuple(ressalvas + [RECUSA_CURTO]))

    return ResultadoNome(nome, tuple(ressalvas))


def _tem_sufixo_societario(nome: str) -> bool:
    """
    O nome termina em sufixo societário — logo é pessoa jurídica de direito privado.

    Existe para absolver a empresa estatal da regra do lado público (ver a nota em
    `_RE_LADO_PUBLICO_*`). O teste é sobre o **fim** do nome, e não sobre conter o
    sufixo em qualquer posição, porque "LTDA" no meio de uma frase não indica forma
    societária — indica que o span pegou texto demais, o que é problema de outra
    classe.

    O sufixo tem de ser uma PALAVRA inteira no fim, não um final de caractere: sem a
    fronteira, `endswith("sa")` casaria 'CASA' e 'EMPRESA' como se fossem 'S.A.', e
    'MESA'/'PESA' como 'SA'. A comparação é feita sobre a última palavra da chave
    normalizada (ou as duas últimas, para '& CIA' e 'S/S' que normalizam em uma só).

    Tolera pontuação e o hífen de nome fantasia no fim ("PRODAM-SP S/A."), porque a
    fonte publica com e sem ponto final.
    """
    fim = _normalizar_chave(nome).rstrip(" .,;-–—")
    palavras = fim.split()
    if not palavras:
        return False
    sufixos = {_normalizar_chave(s).rstrip(" .").strip() for s in _SUFIXOS_SOCIETARIOS}
    sufixos.discard("")
    # a última palavra inteira, ou as duas últimas unidas (cobre '& cia', 's s' etc.)
    ultima = palavras[-1]
    duas = " ".join(palavras[-2:]) if len(palavras) >= 2 else ""
    return ultima in sufixos or duas in sufixos


def _tirar_conectivo_final(nome: str) -> str:
    """
    Remove conectivos pendurados no fim, repetindo até estabilizar (classe (d)).

    Uma passada não basta: remover um conectivo pode expor o seguinte —
    'ACME COMERCIO E DE' → 'ACME COMERCIO E' → 'ACME COMERCIO'. Aplicar uma vez só
    deixaria o 'E' pendurado.
    """
    anterior = None
    atual = nome
    while atual != anterior:
        anterior = atual
        m = re.search(r"\s+(\S+)$", atual)
        if not m:
            break
        ultima = _normalizar_chave(m.group(1)).strip(".")
        if ultima in _CONECTIVOS_FINAIS:
            atual = atual[: m.start()].rstrip(" -–,;.")
    return atual


def _eh_identificador(chave: str) -> bool:
    """
    Indica se a string é um número de processo/protocolo/ajuste, não um nome (c).

    Exige ao menos um componente numérico E que TODOS os componentes sejam numéricos
    ou siglas curtas em caixa alta (≤4 caracteres). Um nome real que contenha número
    ('SUPERMERCADO 3 IRMÃOS') tem palavras longas e não casa. O limite de sigla em 4
    caracteres é o que separa sigla de palavra: um limite maior recusaria MEIs
    legítimos de nome curto que contenham número.
    """
    if not _RE_TEM_DIGITO.search(chave):
        return False
    componentes = re.split(r"[\s/.\-]+", chave.strip())
    componentes = [c for c in componentes if c]
    if not componentes:
        return False
    tem_numero = False
    for c in componentes:
        if _RE_SO_ID.match(c) or c.isdigit():
            tem_numero = tem_numero or bool(_RE_TEM_DIGITO.search(c))
            continue
        # componente alfabético (chave é minúscula): só passa se for sigla curta.
        # 'sgm', 'seme' são siglas; 'supermercado' é palavra e descaracteriza o id.
        if c.isalpha() and len(c) <= 4:
            continue
        # qualquer palavra longa, ou mista com dígitos internos, descaracteriza o id
        return False
    return tem_numero


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENTO FISCAL — CNPJ / CPF com dígito verificador módulo 11
# ═══════════════════════════════════════════════════════════════════════════
# O DV módulo 11 é a verificação mais forte por linha porque é aritmética pura: um
# número lido da posição errada (deslize de OCR, coluna trocada) quase nunca fecha
# os dois dígitos. Por isso `tipo` e `valido` são SEPARADOS: o
# tipo diz o que a FORMA sugere (14 dígitos → cnpj), o válido diz se a ARITMÉTICA
# confirma. A base guarda os dois: um CNPJ bem formado com DV que não fecha é dado
# publicado (não se joga fora) mas não serve para agrupar.

_RE_NAO_DIGITO = re.compile(r"\D")
_RE_MASCARA = re.compile(r"[Xx]")

TIPO_CNPJ = "cnpj"
TIPO_CPF = "cpf"


def _digitos(s: str) -> str:
    return _RE_NAO_DIGITO.sub("", s or "")


def validar_cnpj(documento: str) -> bool:
    """DV do CNPJ pelos dois dígitos verificadores (módulo 11)."""
    d = _digitos(documento)
    if len(d) != 14 or len(set(d)) == 1:
        return False
    def dv(nums, pesos):
        # `strict=True` porque o comprimento é garantido pelo recorte acima, e uma
        # edição futura que mudasse uma lista de pesos truncaria em silêncio e
        # produziria um dígito verificador que valida números errados — defeito pior
        # que exceção, porque passa por verificação.
        s = sum(int(n) * p for n, p in zip(nums, pesos, strict=True))
        r = s % 11
        return "0" if r < 2 else str(11 - r)
    p1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    p2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    return dv(d[:12], p1) == d[12] and dv(d[:13], p2) == d[13]


def validar_cpf(documento: str) -> bool:
    """DV do CPF pelos dois dígitos verificadores (módulo 11)."""
    d = _digitos(documento)
    if len(d) != 11 or len(set(d)) == 1:
        return False
    def dv(nums, ini):
        s = sum(int(n) * p for n, p in zip(nums, range(ini, 1, -1), strict=True))
        r = (s * 10) % 11
        return "0" if r == 10 else str(r)
    return dv(d[:9], 10) == d[9] and dv(d[:10], 11) == d[10]


@dataclass(frozen=True)
class Documento:
    """Resultado da classificação de um documento fiscal."""

    documento: str            # como publicado (grafia preservada)
    tipo: str | None          # 'cnpj', 'cpf' ou None
    valido: bool              # o DV fechou (módulo 11)

    @property
    def digitos(self) -> str:
        """
        O documento só com dígitos — a forma canônica, para armazenar e agrupar.

        A fonte escreve o mesmo documento de várias formas (`12.345.678/0001-95`,
        `12345678000195`, com espaço ou traço no lugar da barra). Guardar a chave só
        com dígitos evita que duas grafias do mesmo número virem dois fornecedores
        no agregado. A grafia original continua disponível em `documento`.
        """
        return "".join(c for c in self.documento if c.isdigit())

    def __iter__(self):
        yield self.documento
        yield self.tipo
        yield self.valido


def classificar_documento(documento: str | None) -> Documento:
    """
    Classifica um documento fiscal, devolvendo tipo e validade SEPARADOS.

    Regras:
      - 14 dígitos → tenta CNPJ; `valido` diz se os dois DV fecharam;
      - 11 dígitos → tenta CPF; idem;
      - documento parcialmente mascarado pela fonte ('XXX.123.456-XX') é preservado
        como está e marcado 'cpf' PELO FORMATO — não há dígitos suficientes para
        validar, então `valido=False`. Marcar pelo formato é honesto: a fonte
        mascara CPF, não CNPJ, e o consumidor precisa saber que ali havia pessoa
        física ainda que o número não seja verificável.

    Returns:
        Documento(documento, tipo, valido).
    """
    if not documento or not documento.strip():
        return Documento("", None, False)

    bruto = documento.strip()
    d = _digitos(bruto)

    # mascarado: tem 'X' e menos de 11 dígitos legíveis, com forma de CPF.
    if _RE_MASCARA.search(bruto) and len(d) < 11:
        # a máscara da fonte é de CPF (protege pessoa física)
        return Documento(bruto, TIPO_CPF, False)

    if len(d) == 14:
        return Documento(bruto, TIPO_CNPJ, validar_cnpj(d))
    if len(d) == 11:
        return Documento(bruto, TIPO_CPF, validar_cpf(d))

    return Documento(bruto, None, False)


def documentos_no_texto(texto: str | None) -> list[str]:
    """
    Localiza CNPJ, CPF e CPF mascarado no texto, na ordem em que aparecem.

    É o recurso da camada 2 quando o documento não veio num campo do formulário.
    Não valida aqui — só localiza; a validação é `classificar_documento`, para que
    achar e validar continuem sendo decisões separadas (o texto pode citar um CNPJ
    de terceiro que não é a contratada).
    """
    if not texto:
        return []
    # CNPJ formatado ou nu; CPF formatado, nu, ou mascarado com X.
    #
    # A borda esquerda usa `(?<![\d.])` em vez de `\b`: `\b` não existe entre `º` e
    # dígito (ambos são caractere de palavra no Unicode), então
    # `CNPJ nº12.345.678/0001-95` colado não seria capturado com `\b`. O lookbehind exige
    # apenas que o dígito inicial NÃO seja precedido de outro dígito ou de ponto —
    # assim um documento colado a `nº`/`:`/letra é capturado, mas um trecho no
    # MEIO de um número pontuado mais longo não é.
    #
    # A borda direita `(?![\d.]*\d)` recusa um casamento que seja PARTE de uma
    # sequência pontuada mais longa: o código de dotação `2.180.33903600.00.1` não
    # deve render o falso CPF `180.33903600`, cujo DV fecha por acaso.
    borda_e = r"(?<![\d.])"
    borda_d = r"(?![\d.]*\d)"
    padrao = re.compile(
        borda_e + r"(?:"
        r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}"              # CNPJ
        r"|\d{3}\.?\d{3}\.?\d{3}-?\d{2}"                   # CPF
        r"|[X\d]{3}\.?[X\d]{3}\.?[X\d]{3}-?[X\d]{2}"       # CPF mascarado
        r")" + borda_d
    )
    return [m.group(0) for m in padrao.finditer(texto)]


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZAÇÃO DE NOME E RAIZ DE CNPJ — peças da chave, nunca a chave sozinhas
# ═══════════════════════════════════════════════════════════════════════════


def normalizar_nome(nome: str | None) -> str:
    """
    Chave canônica do nome para agrupar variações do mesmo fornecedor.

    Remove acento, pontuação de borda e sufixos societários, que são fonte de
    duplicata ('ACME LTDA', 'ACME LTDA.', 'Acme Ltda ME' → 'ACME'). NÃO
    altera a grafia publicada guardada em `contractor_name`; opera só sobre a chave.
    """
    if not nome:
        return ""
    chave = _normalizar_chave(nome)
    chave = re.sub(r"[^\w\s&]", " ", chave)
    palavras = chave.split()
    # remove sufixos societários do fim (podem estar empilhados: 'ACME LTDA ME')
    sufixos = {_normalizar_chave(s).strip(".") for s in _SUFIXOS_SOCIETARIOS}
    while palavras and palavras[-1] in sufixos:
        palavras.pop()
    return " ".join(palavras).strip()


def raiz_cnpj(documento: str | None) -> str | None:
    """
    Raiz (8 primeiros dígitos) de um CNPJ — o que identifica a EMPRESA.

    Os quatro dígitos seguintes distinguem filial; os dois últimos são o DV. Para
    medir concentração a unidade correta é a raiz: duas filiais da mesma empresa
    contratando com o mesmo órgão são um fornecedor, não dois.

    Devolve None para CPF, documento mascarado ou número que não tem 14 dígitos —
    casos em que não há raiz e a chave recai sobre o nome.
    """
    if not documento or _RE_MASCARA.search(documento):
        return None
    d = _digitos(documento)
    if len(d) != 14:
        return None
    return d[:8]


# ═══════════════════════════════════════════════════════════════════════════
# A CHAVE DE AGRUPAMENTO — O ÚNICO PONTO QUE A PRODUZ
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Chave:
    """A chave de agrupamento e a origem que ela declara."""

    chave: str | None         # 'cnpj:11222333' ou 'nome:ACME' ou None
    origem: str | None        # 'cnpj' ou 'nome' ou None

    def __iter__(self):
        yield self.chave
        yield self.origem


def chave_de_agrupamento(
    documento: str | None,
    tipo_documento: str | None,
    documento_valido: bool,
    nome: str | None,
) -> Chave:
    """
    A chave de agrupamento híbrida — e o ÚNICO ponto que aplica a regra.

    Prefere a RAIZ do CNPJ (8 dígitos, que reúne matriz e filiais) quando o
    documento existe E o DV fechou; recai no nome normalizado quando não. A chave
    declara sua origem no prefixo ('cnpj:' ou 'nome:') porque quem consulta precisa
    saber se o agrupamento se apoia em registro público (CNPJ verificado) ou em
    semelhança de grafia (nome).

    Assinatura por CAMPO, não por objeto de registro: permite ao job Spark montar
    a mesma decisão como expressão nativa sobre colunas, e garante que exista UM
    só ponto de aplicação — as camadas determinísticas e a camada de modelo chamam
    esta função; nenhuma monta a chave por conta própria.

    O CNPJ só vira base da chave quando `documento_valido` é True: um número de 14
    dígitos com DV inválido não é confiável para agrupar.

    LIMITAÇÃO (repetida do cabeçalho porque viaja com o resultado): grupo
    econômico com raízes de CNPJ distintas continua fragmentado. A concentração
    publicada a partir desta chave é um PISO.

    Returns:
        Chave(chave, origem). `chave` é None só quando não há nem documento válido
        nem nome — registro sem parte identificável, que não deve ser agrupado.
    """
    if tipo_documento == TIPO_CNPJ and documento_valido:
        raiz = raiz_cnpj(documento)
        if raiz:
            return Chave(f"cnpj:{raiz}", TIPO_CNPJ)

    canonico = normalizar_nome(nome)
    if canonico:
        return Chave(f"nome:{canonico}", "nome")

    return Chave(None, None)


# ═══════════════════════════════════════════════════════════════════════════
# FERRAMENTA DE MEDIÇÃO — a guarda contra a regra de prefixo 'EMPRESA'
# ═══════════════════════════════════════════════════════════════════════════


def medir_prefixo_contra_corpus(prefixo: str, razoes_sociais: list[str]) -> dict:
    """
    Mede quantas razões sociais LEGÍTIMAS uma regra de prefixo recusaria.

    Serve para testar uma regra de prefixo hipotética (ex.: remover 'EMPRESA' do
    início) antes de adicioná-la ao módulo: corrigir poucos erros corrompendo
    muitas razões sociais legítimas (Correios/ECT, PRODAM) não compensa. Uma regra
    de prefixo deve ser avaliada por aqui contra um conjunto de razões sociais
    distintas, e só deve entrar se `atingidas` for zero.

    Args:
        prefixo: a palavra que a regra hipotética removeria do início.
        razoes_sociais: as razões sociais DISTINTAS contra as quais a regra é testada.

    Returns:
        {'prefixo', 'total_razoes', 'atingidas', 'exemplos'} — `atingidas` é quantas
        razões sociais começam pelo prefixo (logo seriam corrompidas), e `exemplos`
        traz até 10 delas para inspeção humana.
    """
    alvo = _normalizar_chave(prefixo)
    atingidas = []
    for rs in razoes_sociais:
        chave = _normalizar_chave(rs)
        if chave.startswith(alvo + " ") or chave == alvo:
            atingidas.append(rs)
    return {
        "prefixo": prefixo,
        "total_razoes": len(razoes_sociais),
        "atingidas": len(atingidas),
        "exemplos": atingidas[:10],
    }


# A chave declara sua origem por prefixo, para que quem consulta distinga
# agrupamento apoiado em registro público (raiz de CNPJ) de agrupamento apoiado
# em semelhança de grafia (nome). Esta guarda confere que os valores de `origem`
# produzidos aqui estão dentro do vocabulário publicado em `vocabulario.ORIGENS_CHAVE`.
_origens = {c.origem for c in (
    chave_de_agrupamento("04.160.147/0012-52", "cnpj", True, "EXEMPLO LTDA"),
    chave_de_agrupamento(None, None, False, "EXEMPLO LTDA"),
)}
if not _origens <= set(V.ORIGENS_CHAVE):
    raise AssertionError(
        f"origem de chave fora do vocabulário publicado: "
        f"{sorted(_origens - set(V.ORIGENS_CHAVE))}"
    )
del _origens
