"""
silver — triagem e as camadas determinísticas. Uma linha por ato.

    publicações do bronze
        │
        ▼  triagem — recusa o que não é ato de contratação, em geral só por metadado
      atos de contratação
        │
        ├── camada 1  campo do formulário do Painel de Negócios
        ├── camada 2  rótulo na prosa (CONTRATADA:, DETENTORA:)
        ├── camada 3  âncora por documento fiscal (CNPJ com DV válido)
        │
        ▼
      resíduo → é a entrada do modelo, e a única

## A regra que ordena as camadas

A camada menos precisa nunca sobrescreve a mais precisa, e as quatro — as três daqui
mais o span do modelo — passam por uma única porta de validação de nome
(`dominio.parte.porta_de_nome`). Como o modelo só vê o que o determinístico não
resolveu, ele não pode degradar o dado que a fonte declarou — garantia que não
depende de nenhuma regra deste repositório estar correta.

## O que este estágio decide, e o modelo não

Validade de documento fiscal pelo módulo 11, grandeza monetária, somabilidade,
natureza financeira, anulação, normalização de data e prazo em meses. O modelo
entra depois, só no span de nome, e o `apply_inference` o faz passar pela mesma
porta.

O `object` também é determinístico: não há gabarito possível para treinar objeto
por redundância intradocumento, e a prosa já rotula o objeto.
"""
from __future__ import annotations

import datetime as dt
import decimal
import hashlib
from dataclasses import dataclass, field
from typing import Any

from .. import MODELO_AUSENTE, VERSAO_REGRAS
from .. import texto as T
from ..contrato import financeiro
from ..dominio import ato, dinheiro, objeto, parte, tempo, triagem
from ..dominio import formulario as F

# OS NOMES DAS CAMADAS
# Em inglês porque são valor de dado, gravados em `field_provenance` e consultados por
# Athena. O esquema público do lago é em inglês; a prosa deste repositório é em português.

CAMADA_FORMULARIO = "form"
CAMADA_ROTULO = "prose_label"
CAMADA_ANCORA = "tax_id_anchor"
CAMADA_MODELO = "model"
CAMADA_PROSA = "prose"

CAMADAS_DETERMINISTICAS: tuple[str, ...] = (
    CAMADA_FORMULARIO, CAMADA_ROTULO, CAMADA_ANCORA,
)

# A FAIXA DE PLAUSIBILIDADE DAS DATAS
#
# O Parquet (via Spark) recusa data anterior a 1582-10-15, a transição do calendário
# Juliano para o Gregoriano, com o erro `WRITE_ANCIENT_DATETIME`. Configurar
# `datetimeRebaseModeInWrite=CORRECTED` evitaria o erro, mas gravaria a data implausível
# (por exemplo, o ano 15 vindo de "15/03" lido sem ano), indistinguível de dado válido.
#
# Por isso a faixa é verificada aqui, e a data implausível vira NULO com a razão
# registrada na proveniência: assim se distingue "a fonte não declarou" de "não foi
# possível ler".
#
# 1980 como piso: aceita aditamento de contrato antigo e recusa anos de dois dígitos.
ANO_MINIMO = 1980
ANOS_A_FRENTE = 2  # tolera assinatura com data futura, até este número de anos

RECUSA_DATA = "recusada_implausivel"

# A data de assinatura que não veio de rótulo, e sim da primeira data do texto.
# `tempo.vigencia` marca essa queda com a ressalva abaixo; aqui ela vira
# proveniência, para que o lago distinga "a fonte rotulou" de "o parser adivinhou".
RESSALVA_PRIMEIRA_DATA = "assinatura_por_primeira_data"
CAMADA_PROSA_PRIMEIRA_DATA = "prose_primeira_data"


def data_plausivel(data: dt.date | None, hoje: dt.date | None = None) -> bool:
    """A data cai numa faixa que um ato de contratação municipal admite."""
    if data is None:
        return False
    referencia = hoje or dt.date.today()
    return ANO_MINIMO <= data.year <= referencia.year + ANOS_A_FRENTE


# A IDENTIDADE DO ATO


def identidade(publicacao: str, numero: str | None, processo: str | None) -> str:
    """
    Identificador estável de um ato: SHA-256 truncado da identidade canônica.

    Hash e não sequencial: sequencial exigiria estado entre execuções, e o pipeline
    não tem onde guardar estado que sobreviva a um `overwrite` de partição. O mesmo
    insumo produz o mesmo identificador em qualquer execução, em qualquer processo.

    A identidade é composta só de campos estáveis da fonte: o identificador da
    publicação, o número do ajuste e o número do processo. O nome da contratada não
    entra na semente: o nome vem da extração (camadas 2, 3 e modelo), e incluí-lo
    tornaria o `contract_id` — e portanto o `ocid` e os ids OCDS derivados — instável
    entre versões do código. Um identificador de ato não pode depender de um campo que
    o próprio pipeline infere.

    A publicação entra sempre; número e processo entram quando existem, porque uma
    publicação pode conter mais de um ato distinguível por eles. O separador `|` é
    literal e não pode aparecer nos componentes.

    32 hexadígitos são 128 bits. Pela aproximação do paradoxo do aniversário, p ≈ n²/2¹²⁹;
    para p = 1e-9 seriam necessários cerca de 8e14 atos.
    """
    semente = "|".join(p for p in (publicacao, numero, processo) if p)
    return hashlib.sha256(semente.encode("utf-8")).hexdigest()[:32]


# O QUE O ESTÁGIO DEVOLVE


@dataclass(frozen=True)
class Descarte:
    """Uma linha de `silver_discard`: a publicação recusada, e por quê."""

    publication_id: str
    section: str | None
    title: str | None
    reason: str
    stage: str
    rule_version: str
    model_version: str
    run_id: str | None
    source_hash: str
    processed_at: dt.datetime
    dt: str

    def como_dicionario(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Ato:
    """
    Uma linha de `silver_contract`.

    `alvo_texto` não é coluna da prata: é o recorte de prosa que o resíduo leva para o modelo,
    e viaja aqui para que `residuo_de` não precise reabrir o bronze. Fica fora de
    `como_dicionario`.
    """

    contract_id: str
    publication_id: str
    edition_date: dt.date | None
    agency: str | None
    act_kind: str | None
    financial_nature: str | None
    is_annulment: bool
    is_republication: bool
    contractor_name: str | None
    contractor_tax_id: str | None
    object: str | None
    contract_number: str | None
    process_number: str | None
    amount_brl: decimal.Decimal | None
    amount_magnitude: str | None
    is_summable: bool
    summability_reason: str
    signature_date: dt.date | None
    term_months: int | None
    legal_basis: str | None
    field_provenance: dict[str, str]
    rule_version: str
    model_version: str
    run_id: str | None
    source_hash: str
    processed_at: dt.datetime
    dt: str
    alvo_texto: str = field(default="", repr=False, compare=False)

    def como_dicionario(self) -> dict[str, Any]:
        """A linha como o Parquet a recebe — sem `alvo_texto`, que não é coluna."""
        d = dict(self.__dict__)
        d.pop("alvo_texto", None)
        return d

    @property
    def resolvido(self) -> bool:
        """A parte contratada foi resolvida por alguma camada determinística."""
        return self.contractor_name is not None


@dataclass(frozen=True)
class Residuo:
    """Uma linha de `silver_residual`: o ato que vai ao modelo, com o texto que ele lê."""

    contract_id: str
    publication_id: str
    reason: str
    target_text: str
    rule_version: str
    model_version: str
    run_id: str | None
    source_hash: str
    processed_at: dt.datetime
    dt: str

    def como_dicionario(self) -> dict[str, Any]:
        return dict(self.__dict__)


# O ESTÁGIO


def de_publicacao(
    linha: dict[str, Any],
    *,
    run_id: str | None = None,
    agora: dt.datetime | None = None,
    hoje: dt.date | None = None,
) -> Ato | Descarte:
    """
    Aplica triagem e as camadas determinísticas a uma linha de bronze.

    Devolve `Ato` quando a publicação é ato de contratação, e `Descarte` quando não é. Nunca
    `None`: o descarte é dado, não ausência de dado. As publicações descartadas não podem
    desaparecer em silêncio; é a tabela de descarte que torna a triagem auditável.

    `hoje` é injetável para que a faixa de plausibilidade não dependa do calendário.
    """
    instante = agora or dt.datetime.now(dt.UTC)
    bruto = linha.get("body_html") or ""
    serie = linha.get("title") or ""
    veiculo = linha.get("section") or ""
    orgao = linha.get("agency") or ""
    digest = linha.get("source_hash") or ""
    # Ao ler o bronze particionado, o Spark infere o tipo da coluna de partição
    # `dt=AAAA-MM-DD` e a devolve como objeto `date`, não como string. Regravar esse objeto
    # como valor de partição produz o `toString()` de um GregorianCalendar, incompatível com
    # a partition projection do Athena. Formatar para ISO `AAAA-MM-DD` aqui garante a string
    # em qualquer caso.
    particao = _dt_iso(linha.get("dt"))
    identificador = linha.get("publication_id") or ""

    def descartar(motivo: str, estagio: str) -> Descarte:
        return Descarte(
            publication_id=identificador, section=veiculo, title=serie,
            reason=motivo, stage=estagio,
            rule_version=VERSAO_REGRAS, model_version=MODELO_AUSENTE, run_id=run_id,
            source_hash=digest, processed_at=instante, dt=particao,
        )

    # triagem: `triar` devolve (motivo_da_recusa, estágio). Motivo None significa aceito.
    #
    # Quando série e veículo bastam para decidir (`eh_metadado_suficiente`), a recusa
    # dispensa converter o HTML, que é a etapa cara; só as demais publicações são convertidas
    # antes de triar.
    if triagem.eh_metadado_suficiente(serie, veiculo):
        motivo, estagio = triagem.triar(serie, veiculo)
        if motivo is not None:
            return descartar(motivo, estagio)
        conteudo = T.html_para_texto(bruto)
    else:
        conteudo = T.html_para_texto(bruto)
        motivo, estagio = triagem.triar(serie, veiculo, conteudo)
        if motivo is not None:
            return descartar(motivo, estagio)

    proveniencia: dict[str, str] = {}
    form = F.ler(bruto)

    # O recorte onde procurar rótulo, valor e documento: o campo de prosa quando existe, o
    # texto todo quando não. Procurar "R$" no texto completo de um layout de formulário acha o
    # valor do extrato e qualquer número que outro campo declare, competindo com dado que não
    # é do ajuste.
    alvo = F.alvo_da_analise(form, conteudo)

    classificacao = ato.classificar(serie, conteudo)

    # a parte contratada, pelas três camadas
    nome, camada, documento_da_ancora = _contratada(form, alvo)
    if nome is not None:
        proveniencia["contractor_name"] = camada  # type: ignore[assignment]

    # documento fiscal
    documento, origem_documento = _documento(form, conteudo, documento_da_ancora)
    if documento is not None:
        proveniencia["contractor_tax_id"] = origem_documento  # type: ignore[assignment]

    # objeto
    texto_objeto, camada_objeto, _ = objeto.extrair(
        form.campos.get(F.CAMPO_OBJETO), alvo
    )
    if texto_objeto is not None:
        proveniencia["object"] = camada_objeto  # type: ignore[assignment]

    # valor e grandeza
    # Dois parsers independentes na mesma frase: numeral e extenso. `valor_do_ajuste` escolhe
    # entre as ocorrências concorrentes por, nesta ordem, peso do rótulo, confirmação por
    # extenso, magnitude e posição no texto (ver `dominio.dinheiro.escolher_valor`).
    eleito = dinheiro.valor_do_ajuste(alvo)
    valor: decimal.Decimal | None = None
    grandeza: str | None = None
    if eleito is not None and eleito.valor is not None:
        # Decimal e não float: o Parquet grava decimal(18,2) e o float introduz erro de
        # representação que aparece na soma do agregado mensal.
        valor = decimal.Decimal(str(eleito.valor)).quantize(decimal.Decimal("0.01"))
        grandeza = eleito.grandeza
        proveniencia["amount_brl"] = CAMADA_PROSA
        proveniencia["amount_magnitude"] = grandeza or "indefinido"

    # tempo
    assinatura, prazo = _tempo(form, alvo, proveniencia, hoje)

    # a regra financeira
    anulacao = bool(classificacao.get("eh_anulacao"))
    somavel, motivo_soma = financeiro.avaliar(
        valor=valor,
        natureza=classificacao.get("natureza"),  # type: ignore[arg-type]
        grandeza=grandeza,
        anulado=anulacao,
    )

    return Ato(
        contract_id=identidade(
            identificador,
            form.campos.get(F.CAMPO_NUMERO_AJUSTE),
            form.campos.get(F.CAMPO_PROCESSO),
        ),
        publication_id=identificador,
        edition_date=_data_iso(particao),
        agency=orgao or form.campos.get(F.CAMPO_ORGAO),
        act_kind=classificacao.get("tipo"),  # type: ignore[arg-type]
        financial_nature=classificacao.get("natureza"),  # type: ignore[arg-type]
        is_annulment=anulacao,
        is_republication=bool(classificacao.get("eh_republicacao")),
        contractor_name=nome,
        contractor_tax_id=documento,
        object=texto_objeto,
        contract_number=form.campos.get(F.CAMPO_NUMERO_AJUSTE),
        process_number=form.campos.get(F.CAMPO_PROCESSO),
        amount_brl=valor,
        amount_magnitude=grandeza,
        is_summable=somavel,
        summability_reason=motivo_soma,
        signature_date=assinatura,
        term_months=prazo,
        legal_basis=form.campos.get(F.CAMPO_FUNDAMENTO),
        field_provenance=proveniencia,
        rule_version=VERSAO_REGRAS,
        model_version=MODELO_AUSENTE,
        run_id=run_id,
        source_hash=digest,
        processed_at=instante,
        dt=particao,
        alvo_texto=alvo,
    )


def _contratada(
    form: F.Formulario, alvo: str
) -> tuple[str | None, str | None, str | None]:
    """
    A parte contratada pelas três camadas, na ordem de precisão.

    Cada camada só é tentada se a anterior não resolveu. Inverter duas delas seria deixar a
    menos precisa sobrescrever a mais precisa.

    Returns:
        `(nome, camada, documento_da_ancora)`. O terceiro item vem preenchido quando a camada
        2 ou a 3 devolve um documento junto com o nome; é o documento associado àquele nome,
        a informação que `_documento` precisa para não gravar o documento de outra parte.
    """
    achado = F.contratada_do_formulario(form, alvo)
    if achado.nome:
        return achado.nome, CAMADA_FORMULARIO, None

    achado = F.contratada_da_prosa(alvo)
    if achado.nome:
        return achado.nome, CAMADA_ROTULO, achado.documento

    achado = F.contratada_por_ancora(alvo)
    if achado.nome:
        return achado.nome, CAMADA_ANCORA, achado.documento

    return None, None, None


def _documento(
    form: F.Formulario, conteudo: str, documento_da_ancora: str | None = None
) -> tuple[str | None, str | None]:
    """
    O documento fiscal da CONTRATADA, só com dígito verificador válido.

    O DV é conferido: um documento que falha no módulo 11 é erro de digitação
    da fonte ou captura da posição errada, e nos dois casos não serve como chave de
    agrupamento no ouro.

    A ordem prioriza o documento devolvido junto com o nome (camadas 2 e 3): é o documento
    daquele nome, encontrado junto dele no texto. Varrer o texto inteiro e gravar o primeiro
    documento válido, independentemente de qual parte ele identifica, atribuiria o CNPJ do
    órgão ao fornecedor num ato que cita o órgão antes da empresa; o DV seria válido, então
    a verificação de dígito não detecta o erro.
    """
    if documento_da_ancora:
        classificado = parte.classificar_documento(documento_da_ancora)
        if classificado.valido:
            return classificado.digitos, CAMADA_ANCORA

    campo = form.campos.get(F.CAMPO_DOCUMENTO_FISCAL)
    if campo:
        classificado = parte.classificar_documento(campo)
        if classificado.valido:
            return classificado.digitos, CAMADA_FORMULARIO

    for bruto in parte.documentos_no_texto(conteudo):
        classificado = parte.classificar_documento(bruto)
        if classificado.valido:
            return classificado.digitos, CAMADA_PROSA

    return None, None


def _tempo(
    form: F.Formulario,
    alvo: str,
    proveniencia: dict[str, str],
    hoje: dt.date | None,
) -> tuple[dt.date | None, int | None]:
    """
    Assinatura e prazo, com precedência formulário → prosa.

    A mesma precedência das camadas de contratada, e pela mesma razão: o campo é declaração da
    fonte, a prosa é leitura nossa.
    """
    do_form = tempo.do_formulario(
        assinatura=form.campos.get(F.CAMPO_ASSINATURA),
        inicio=form.campos.get(F.CAMPO_INICIO),
        fim=form.campos.get(F.CAMPO_FIM),
        prazo=form.campos.get(F.CAMPO_PRAZO),
        # A unidade vem separada no formulário: "12" e "meses" são dois campos. Sem ela, um
        # prazo em dias seria lido como meses.
        prazo_unidade=form.campos.get(F.CAMPO_PRAZO_UNIDADE),
    )
    da_prosa = tempo.vigencia(alvo)

    assinatura = do_form.data_assinatura or da_prosa.data_assinatura
    if assinatura is not None and not data_plausivel(assinatura, hoje):
        # Fora da faixa é erro de leitura, não data antiga. Registrar a recusa com o ano lido
        # permite medir depois quantas datas o parser erra.
        proveniencia["signature_date"] = f"{RECUSA_DATA}_{assinatura.year}"
        assinatura = None
    elif assinatura is not None:
        # A proveniência distingue rótulo de heurística. `tempo.vigencia` procura um rótulo de
        # assinatura e, quando não acha, cai para a primeira data do texto — heurística
        # declarada, marcada com a ressalva `assinatura_por_primeira_data`. Sem registrar essa
        # ressalva, quem consulta o lago não teria como separar uma data que a fonte rotulou de
        # uma que o parser adivinhou; a heurística pode pegar a primeira data de outra cláusula
        # (por exemplo, uma prorrogação) em vez da data de assinatura.
        if do_form.data_assinatura:
            proveniencia["signature_date"] = CAMADA_FORMULARIO
        elif RESSALVA_PRIMEIRA_DATA in da_prosa.ressalvas:
            proveniencia["signature_date"] = CAMADA_PROSA_PRIMEIRA_DATA
        else:
            proveniencia["signature_date"] = CAMADA_PROSA

    prazo = do_form.prazo_meses or da_prosa.prazo_meses
    if prazo:
        proveniencia["term_months"] = (
            CAMADA_FORMULARIO if do_form.prazo_meses else CAMADA_PROSA
        )
    return assinatura, prazo


def _data_iso(valor: str) -> dt.date | None:
    """A partição `aaaa-mm-dd` como `date`, ou None se não for uma data."""
    try:
        return dt.date.fromisoformat(valor)
    except (ValueError, TypeError):
        return None


def _dt_iso(valor: Any) -> str:
    """
    A partição `dt` como string ISO `AAAA-MM-DD`, venha ela como `date`, `datetime` ou `str`.

    O Spark infere o tipo da coluna de partição ao ler o bronze e devolve `dt` como `date`.
    Sem esta normalização, gravar o objeto de volta como valor de partição produziria o dump
    de um `GregorianCalendar` na chave, incompatível com a partition projection do Athena.
    """
    if valor is None or valor == "":
        return ""
    if isinstance(valor, dt.datetime):
        return valor.date().isoformat()
    if isinstance(valor, dt.date):
        return valor.isoformat()
    return str(valor)[:10]


# O RESÍDUO

MOTIVO_SEM_NOME = "sem_nome_nas_tres_camadas"


def residuo_de(atos: list[Ato]) -> list[Residuo]:
    """
    Os atos que nenhuma camada determinística resolveu — a entrada do modelo, e a única.

    `target_text` leva o recorte de prosa, e não o HTML: é o mesmo recorte que o gabarito usa
    como entrada, e é isso que faz o que o modelo vê em produção ser o que ele viu no treino.
    Passar o HTML aqui criaria uma diferença silenciosa entre treino e inferência.

    Resíduo com `target_text` vazio é descartado: não há o que o modelo leia, e mandá-lo ao
    Batch Transform gastaria uma inferência para receber nada.
    """
    return [
        Residuo(
            contract_id=a.contract_id,
            publication_id=a.publication_id,
            reason=MOTIVO_SEM_NOME,
            target_text=a.alvo_texto,
            rule_version=a.rule_version,
            model_version=a.model_version,
            run_id=a.run_id,
            source_hash=a.source_hash,
            processed_at=a.processed_at,
            dt=a.dt,
        )
        for a in atos
        if not a.resolvido and a.alvo_texto.strip()
    ]
