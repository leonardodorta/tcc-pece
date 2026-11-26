"""
triagem.py — decide se uma publicação do DOM-SP é ato de contratação em escopo
ou ruído editorial, e por quê. O caderno "Negócios" mistura contrato firmado,
licitação (fase de disputa, fora de escopo) e ruído administrativo; esta triagem
separa os três.

A decisão roda em dois estágios: a maior parte do volume é decidida só pelo
metadado (caderno, série, veículo), que é barato; só a série genérica
("Extrato"/"Termo") sem palavra confirmadora precisa abrir o texto do HTML, que é
caro. `REGRAS_TRIAGEM` declara essa divisão como dado, e `OPERADORES_METADADO` diz
quais operadores decidem por metadado — assim `pipeline.silver` aplica as regras de
metadado antes do parsing e as de texto depois.

O caderno é derivado do `veiculo` (não do prefixo de série, que pode variar)
para recusar os cadernos de controle externo (Câmara e TCM).

O vocabulário de motivos é fechado e é o mesmo que `contrato/esquema.py` promete
na coluna `reason` de `silver_discard`: `caderno_fora_de_escopo`,
`serie_de_licitacao`, `serie_de_ruido`, `veiculo_fora_de_escopo`,
`sem_palavra_confirmadora`, `sem_atos_publicados`. Série não reconhecida, num
veículo em escopo, é recusada como `serie_de_ruido` (postura conservadora: não há
`serie_desconhecida`). `sem_atos_publicados` é o dia sem edição (art. 5º/7º do
Decreto 62.177/2023), reconhecido por marcador no texto.

Função escalar pura sobre `str`: não importa pyspark, torch, boto3 nem pandas.
De `domsp.contrato` importa apenas `contrato.vocabulario`.
"""
from __future__ import annotations

from .. import texto as T
from ..contrato import vocabulario as V

# Os sete cadernos do art. 13 § único do Decreto 62.177/2023; a fonte não devolve o
# caderno como campo, devolve o `veiculo`. Este mapa é a ponte, com os nomes de
# veículo que a fonte usa.

CADERNO_EXECUTIVO = "Atos do Executivo"
CADERNO_SERVIDORES = "Servidores"
CADERNO_CONCURSOS = "Concursos"
CADERNO_EDITAIS = "Editais"
CADERNO_NEGOCIOS = "Negócios"
CADERNO_CAMARA = "Atos da Câmara"
CADERNO_TCM = "Atos do TCM"

CADERNO_DO_VEICULO: dict[str, str] = {
    "Atos do Executivo": CADERNO_EXECUTIVO,
    "Servidores": CADERNO_SERVIDORES,
    "Concursos": CADERNO_CONCURSOS,
    "Editais": CADERNO_EDITAIS,
    "Negócios": CADERNO_NEGOCIOS,
    "Atos da CMSP": CADERNO_CAMARA,
    "Atos do TCM-SP": CADERNO_TCM,
}

# Cadernos de controle externo: são atos da Câmara e do Tribunal de Contas do
# Município, não contratações do Executivo. Recusados por caderno, o que é robusto
# à inconstância do prefixo de série.
CADERNOS_CONTROLE_EXTERNO: frozenset[str] = frozenset({CADERNO_CAMARA, CADERNO_TCM})

# Só estes veículos publicam atos de contratação. O veículo é a segunda barreira
# contra homonímia de série: "Apostilamento" e "Termo" no veículo "Servidores"
# são apostila de aposentadoria e termo de posse — atos de pessoal, não contratos.
VEICULOS_CONTRATACAO: frozenset[str] = frozenset({
    "Negócios", "Atos do Executivo", "Editais",
})


def caderno_do_veiculo(veiculo: str | None) -> str | None:
    """O caderno do art. 13 a que o veículo pertence, ou `None` se desconhecido."""
    return CADERNO_DO_VEICULO.get((veiculo or "").strip())


# ═══════════════════════════════════════════════════════════════════════════
# TAXONOMIA DE SÉRIES
# ═══════════════════════════════════════════════════════════════════════════
# Séries com sufixo "(NP)" vêm do Painel de Negócios e trazem formulário HTML
# estruturado; as demais são texto semi-estruturado com rótulos em CAPS.

# --- Séries que representam contratação (série exata → tipo canônico) ---
# Os tipos à direita são os da coluna `act_kind` de silver_contract, além de
# `extrato_generico` (TIPO_GENERICO), que é refinado depois pelo texto. Não existe
# `compra` no vocabulário: "Extrato de Compras" é despesa efetiva e entra como
# `contrato`.
SERIES_CONTRATO: dict[str, str] = {
    # Painel de Negócios (formulário estruturado)
    "Extrato de Contrato/Nota de empenho (NP)": "contrato",
    "Extrato de Contratação (NP)": "contrato",
    "Extrato de Aditamento (NP)": "aditivo",
    "Extrato Aditamento Ata Registro de Preço (NP)": "aditivo",
    "Extrato de Ata de Registro de Preço (NP)": "ata_registro_preco",
    "Extrato de Apostilamento (NP)": "apostilamento",
    "Extrato de Compras (NP)": "contrato",   # compra é despesa efetiva; não há tipo `compra`
    "Emissão Nota de Empenho (NP)": "empenho",
    "Extrato de Termo de Doação (NP)": "doacao",
    "Extrato de Convênio (NP)": "convenio",
    "Extrato de Autorização de Uso (NP)": "autorizacao_uso",
    "Penalidade (NP)": "penalidade",
    "Rescisão (NP)": "rescisao",
    # Fora do Painel de Negócios (texto semi-estruturado). Filtrar só pelo veículo
    # "Negócios" descartaria contratos: este bloco, as séries genéricas e GARP
    # aparecem noutros veículos.
    "Extrato do Termo Aditivo": "aditivo",
    "Extrato de Termo de Doação": "doacao",
    "Extrato de Convênio": "convenio",
    "Extrato de Contrato": "contrato",
    "Extrato de Ata de Registro de Preços": "ata_registro_preco",
    "Extrato de Aditamento": "aditivo",
    "Apostilamento": "apostilamento",
    "Extrato de Acordo de Cooperação Técnica": "acordo_cooperacao",
    # Gestão de Atas de Registro de Preços (SEGES): formulário próprio, com o valor
    # da ata em campo dedicado.
    "Extrato de ARP - GARP": "ata_registro_preco",
    "Extrato de Apostilamento - GARP": "apostilamento",
    # Séries de nome genérico: só entram se o TEXTO confirmar contratação (segundo
    # estágio). Ver KEYWORDS_CONFIRMACAO e o tipo TIPO_GENERICO.
    "Extrato": "extrato_generico",
    "Termo": "extrato_generico",
}

# --- Séries explicitamente de licitação (fase de disputa, fora de escopo) ---
# Descartadas com motivo próprio: há contrato à frente, mas ainda não firmado.
SERIES_LICITACAO: frozenset[str] = frozenset({
    "Abertura (NP)", "Homologação (NP)", "Julgamento (NP)", "Revogação (NP)",
    "Anulação (NP)", "Deserto/Fracassado (NP)", "Suspensão (NP)",
    "Retificação/Alteração (NP)", "Ata da Licitação (NP)", "Recurso (NP)",
    "Impugnação (NP)", "Prejudicada (NP)", "Cancelamento  (NP)",
    "Registro de alteração (NP)", "Registro de adjudicação (NP)",
    "Comunicado  (NP)", "Comunicado (NP)", "Despacho autorizatório (NP)",
    "Outras (NP)", "Chamamento Público - Homologação  (NP)",
    "Inexigibilidade de Chamamento Público (NP)",
    "Aviso - Edital de Chamamento Público (NP)",
    "Aviso de Licitação", "Edital", "Edital de Seleção Pública",
    "Despacho Autorizatório de Abertura de Licitação",
})

# --- Séries de ruído administrativo/pessoal — nunca contratação ---
SERIES_RUIDO: frozenset[str] = frozenset({
    "Despacho deferido", "Despacho indeferido", "Comunique-se", "Despacho",
    "Despacho Autorizatório", "Despacho Autorização", "Decisão Tributária",
    "Portaria", "Notificação", "Notificação Orientativa", "Intimação",
    "Convocação", "Auto de Infração", "Comunicado", "Decisão", "Resolução",
    "Ata de Reunião", "Ata", "Extrato de Ata", "Designação de Fiscal",
    "Título de Nomeação", "Movimentação de Pessoal", "Errata",
    "Despacho Documental", "Despacho Rerratificação", "Despacho de Retificação",
    "Despacho Parcialmente Deferido", "Despacho Pagamento", "Adiantamento",
    "GeoInfra - Despacho de Certif. Conclusão de Obras",
    "Despacho de Edificações - Súmula", "Despacho de Poda/Remoção de Árvore",
    "Despacho Autorizatório-Ação Judicial", "SP-156: Indeferido",
    "SP-156: Deferido", "Balancetes dos Fundos Municipais", "Deliberação",
    "Atos Normativos e Despachos", "Autorização", "Ato", "Retificação",
    "Demonstrativo de Compras", "SIMPROC - Editais",
    "Edital de ciência de eliminação de documentos",
    "Revisão Numérica de Logradouro Simples",
    # Séries de pessoal: entram como ruído por serem inequivocamente atos de
    # servidor.
    "Despacho de Apostilamento",   # apostila de aposentadoria
    "Licença", "Férias", "Edital de Licenças Médicas", "Pauta",
    "Despacho de Acolhimento",
})

# Para a série genérica "Extrato"/"Termo", exigimos evidência textual de
# contratação; sem ela a publicação é descartada como `sem_palavra_confirmadora`.
KEYWORDS_CONFIRMACAO: tuple[str, ...] = (
    "extrato de contrato", "extrato do contrato", "termo de contrato",
    "extrato de acordo de cooperação", "termo de apostilamento",
    "extrato do aditivo", "termo aditivo", "extrato de aditamento",
    "termo de colaboração", "termo de fomento", "termo de doação",
    "extrato de convênio", "contrato de gestão", "ata de registro de preço",
    "contratada:", "contratado:", "contratado(a):", "detentora:",
)

# Marcador de edição sem atos (art. 5º e 7º do Decreto 62.177/2023). É procurado
# no texto normalizado; por isso `sem_atos_publicados` é regra de texto.
MARCADORES_SEM_ATOS: tuple[str, ...] = (
    "sem atos oficiais publicados",
    "nao ha atos oficiais",
)

# Série de nome genérico: o único tipo cujo conteúdo precisa ser aberto.
TIPO_GENERICO = "extrato_generico"

# Janela do conteúdo (em caracteres) em que a evidência textual é procurada. É onde
# os rótulos de cabeçalho do ato aparecem.
JANELA_CONFIRMACAO = 1500


# ═══════════════════════════════════════════════════════════════════════════
# TRIAGEM — a ordem dos testes, declarada como dado
# ═══════════════════════════════════════════════════════════════════════════
# A ordem é parte da regra, não detalhe de implementação: caderno e série são
# testados antes de qualquer leitura de conteúdo. Cada regra é
# (motivo, operador, argumento); os motivos são EXATAMENTE os da coluna `reason`
# de silver_discard. Os operadores são poucos de propósito — cada um tem tradução
# direta em expressão Spark.

REGRAS_TRIAGEM: tuple[tuple[str, str, object], ...] = (
    ("caderno_fora_de_escopo",   "caderno_em",        CADERNOS_CONTROLE_EXTERNO),
    ("veiculo_fora_de_escopo",   "veiculo_fora_de",   VEICULOS_CONTRATACAO),
    ("serie_de_licitacao",       "serie_em",          SERIES_LICITACAO),
    ("serie_de_ruido",           "serie_em",          SERIES_RUIDO),
    # Série não reconhecida como contrato, num veículo em escopo, é ruído (ver o
    # cabeçalho: não há `serie_desconhecida`, e a postura é conservadora).
    ("serie_de_ruido",           "serie_fora_de",     frozenset(SERIES_CONTRATO)),
    # A partir daqui é preciso o texto:
    ("sem_atos_publicados",      "sem_atos",          MARCADORES_SEM_ATOS),
    ("sem_palavra_confirmadora", "exige_confirmacao", KEYWORDS_CONFIRMACAO),
)

# Quais operadores decidem só por metadado. A distinção é o que faz a triagem ser
# barata: estas regras são aplicadas ANTES de converter o HTML em texto. A
# partição sai do nome do operador — acrescentar uma regra em REGRAS_TRIAGEM a
# coloca no estágio certo.
OPERADORES_METADADO: frozenset[str] = frozenset({
    "caderno_em", "veiculo_fora_de", "serie_em", "serie_fora_de",
})

ESTAGIO_METADADO = "metadado"
ESTAGIO_TEXTO = "texto"


def _norm(s: str | None) -> str:
    """Normaliza para comparação sem acento/caixa. Usa a API de `texto`."""
    return T.remover_acentos(s or "").lower()


def keyword_confirmadora(conteudo_texto: str | None) -> str | None:
    """
    A primeira palavra-chave de contratação presente na janela inicial do texto.

    "Primeira" é na ordem de KEYWORDS_CONFIRMACAO (a mais específica antes), não na
    ordem em que aparecem no texto — a keyword escolhida decide o tipo do ato em
    `ato.py`, então a ordem importa. Comparação sem acento para robustez à fonte
    ("convênio"/"convenio").
    """
    cabeca = _norm((conteudo_texto or "")[:JANELA_CONFIRMACAO])
    return next((k for k in KEYWORDS_CONFIRMACAO if _norm(k) in cabeca), None)


def _tem_marcador_sem_atos(conteudo_texto: str | None) -> bool:
    cabeca = _norm((conteudo_texto or "")[:JANELA_CONFIRMACAO])
    return any(m in cabeca for m in MARCADORES_SEM_ATOS)


def triar(
    serie: str | None,
    veiculo: str | None,
    conteudo_texto: str | None = None,
) -> tuple[str | None, str]:
    """
    Decide o destino de uma publicação. Devolve `(motivo, estagio)`.

    - Se a publicação é um ato de contratação em escopo, `motivo` é `None` e
      `estagio` é o estágio em que a decisão de ACEITAR foi tomada (`metadado` se
      nenhuma regra de texto precisou rodar, `texto` se a confirmação textual foi
      consultada).
    - Se foi recusada, `motivo` é um dos seis valores do contrato e `estagio` diz
      se a recusa saiu do metadado ou do texto — é a medida do que a triagem
      barata economizou (a coluna `stage` de silver_discard).

    `conteudo_texto` é o texto já extraído do HTML (por `texto.html_para_texto`) e
    só é consultado nas regras de texto. Quem chama pode passar `None` para obter
    apenas a decisão por metadado — que é o primeiro estágio do job. Passar `None`
    não é "ignorar as regras de texto": é declarar que elas ainda não podem ser
    decididas, e o job as aplica na segunda passada, sobre o que sobrou.
    """
    s = (serie or "").strip()
    v = (veiculo or "").strip()
    caderno = caderno_do_veiculo(v)
    consultou_texto = False

    for motivo, operador, arg in REGRAS_TRIAGEM:
        # O estágio é DERIVADO de `OPERADORES_METADADO`, não escrito em cada ramo —
        # assim uma regra de metadado nova nunca é reportada por engano como estágio
        # `texto`.
        estagio = ESTAGIO_METADADO if operador in OPERADORES_METADADO else ESTAGIO_TEXTO

        if operador == "caderno_em":
            if caderno in arg:
                return motivo, estagio
        elif operador == "veiculo_fora_de":
            if v not in arg:
                return motivo, estagio
        elif operador == "serie_em":
            if s in arg:
                return motivo, estagio
        elif operador == "serie_fora_de":
            if s not in arg:
                return motivo, estagio
        elif operador == "sem_atos":
            # Regra de texto: só decide se o texto foi fornecido.
            if conteudo_texto is None:
                continue
            consultou_texto = True
            if _tem_marcador_sem_atos(conteudo_texto):
                return motivo, estagio
        elif operador == "exige_confirmacao":
            # Só as séries genéricas exigem confirmação textual.
            if SERIES_CONTRATO.get(s) != TIPO_GENERICO:
                continue
            if conteudo_texto is None:
                continue
            consultou_texto = True
            if keyword_confirmadora(conteudo_texto) is None:
                return motivo, estagio
        else:  # pragma: no cover - guarda contra regra sem operador conhecido
            raise ValueError(f"operador de triagem desconhecido: {operador!r}")

    # Passou por todas as regras: é ato de contratação.
    return None, (ESTAGIO_TEXTO if consultou_texto else ESTAGIO_METADADO)


def motivo_recusa(
    serie: str | None,
    veiculo: str | None,
    conteudo_texto: str | None = None,
) -> str | None:
    """Só o motivo (ou `None`). Conveniência para quem não precisa do estágio."""
    return triar(serie, veiculo, conteudo_texto)[0]


def eh_metadado_suficiente(serie: str | None, veiculo: str | None) -> bool:
    """
    `True` se o metadado sozinho já decide (recusa ou série não genérica).

    É o predicado que o job usa para saber se precisa abrir o HTML desta linha:
    uma série genérica que passou pelo metadado ainda depende do texto; qualquer
    outra decisão já está tomada. Existe para tornar explícito, e testável fora do
    Spark, o critério que decide quem paga o parsing.
    """
    motivo, _ = triar(serie, veiculo, conteudo_texto=None)
    if motivo is not None:
        return True
    return SERIES_CONTRATO.get((serie or "").strip()) != TIPO_GENERICO


# MOTIVOS e ESTAGIOS reexportam o vocabulário único publicado em
# `contrato/vocabulario`, para que quem lê este módulo encontre o conjunto sem
# precisar saber onde ele mora.

MOTIVOS: tuple[str, ...] = V.MOTIVOS_DESCARTE
ESTAGIOS: tuple[str, ...] = V.ESTAGIOS

if ESTAGIO_METADADO != V.ESTAGIO_METADADO or ESTAGIO_TEXTO != V.ESTAGIO_TEXTO:
    raise AssertionError(
        "os nomes dos estágios divergem do vocabulário publicado: "
        f"{(ESTAGIO_METADADO, ESTAGIO_TEXTO)} != "
        f"{(V.ESTAGIO_METADADO, V.ESTAGIO_TEXTO)}"
    )
