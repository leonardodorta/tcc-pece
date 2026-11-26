"""
ocds.py — exporta a prata consolidada no Open Contracting Data Standard 1.1 e valida o resultado.

O OCDS é um esquema publicado pela Open Contracting Partnership para que informação de
contratação pública seja comparável entre jurisdições. Exportar para ele serve a dois
propósitos neste projeto.

O primeiro é de publicação: uma base em esquema próprio só é útil a quem aprende o esquema,
enquanto uma base em padrão internacional entra direto em ferramentas que já existem.

O segundo é de verificação. O esquema do OCDS **não foi escrito por este projeto**.
Submeter a base a ele é pedir que um terceiro diga se o dado está bem formado. As cópias dos
esquemas oficiais estão em `infraestrutura/esquemas/ocds-1.1.5/`, com origem e hash
registrados.

## O que é exportado, e o que não é

O OCDS modela o processo de contratação inteiro: planejamento, edital, propostas, julgamento,
adjudicação, contrato e execução. O DOM-SP publica o **ato**, não o processo — o que chega é o
extrato do contrato ou do aditamento, já assinado. Então a exportação popula o que o ato
declara e deixa vazio o que ele não declara. Em vocabulário do OCDS: `contracts` e `awards`
são preenchidos; `tender` e `planning` não, porque a fonte não os traz.

Isso é limite da fonte, não do mapeamento, e fica declarado em vez de preenchido com valor
inventado — um `tender` fabricado passaria na validação e mentiria.

## O identificador do processo (ocid)

O OCDS exige que cada processo tenha um `ocid` globalmente único, formado por um **prefixo
registrado** junto à Open Contracting Partnership mais um identificador interno. Este
projeto não registrou prefixo, então usa `PREFIXO_OCID`, marcado abaixo como
não registrado. Publicação real exige registrar o prefixo; o resto do mapeamento não muda.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Prefixo do ocid. O padrão pede algo registrado; este não é, e o nome diz isso para que
# ninguém publique a base achando que está conforme. Trocar por um prefixo registrado é a
# única alteração necessária para publicar.
PREFIXO_OCID = "ocds-naoregistrado"

# Versão do padrão contra a qual a exportação é validada, e onde estão as cópias.
VERSAO_OCDS = "1.1"
DIRETORIO_ESQUEMAS = Path("infraestrutura/esquemas/ocds-1.1.5")
URL_ESQUEMA_RELEASE = (
    "https://standard.open-contracting.org/schema/1__1__5/release-schema.json"
)

# Moeda. O DOM-SP publica em reais e não declara moeda; o OCDS exige o código ISO 4217.
MOEDA = "BRL"

# O ato pode ser contrato novo ou alteração de um existente. O OCDS tem uma lista fechada de
# etiquetas (`releaseTag`), e as duas que cabem aqui são estas.
TAG_CONTRATO = "contract"
TAG_ALTERACAO = "contractAmendment"

# Mapa de campos, declarado para que a correspondência seja auditável sem ler o código.
# Cada entrada é: campo da prata -> caminho no release do OCDS.
MAPA_DE_CAMPOS: Mapping[str, str] = {
    "contract_id": "releases[].id e releases[].contracts[].id",
    "process_number": "releases[].ocid (compõe, após o prefixo)",
    "signature_date": "releases[].date e releases[].contracts[].dateSigned",
    "agency": "releases[].buyer.name e releases[].parties[roles=buyer].name",
    "contractor_name": "releases[].parties[roles=supplier].name",
    "contractor_tax_id": "releases[].parties[roles=supplier].identifier.id",
    "object": "releases[].contracts[].title",
    "amount_brl": "releases[].contracts[].value.amount",
    "term_months": "releases[].contracts[].period.durationInDays (convertido)",
    "contract_number": "releases[].contracts[].description (quando presente)",
    "act_kind": "releases[].tag (contract ou contractAmendment)",
    "legal_basis": "releases[].contracts[].description (quando presente)",
}

# Campos da prata que NÃO vão para o OCDS, com a razão. Declarar a ausência evita a leitura
# de que foram esquecidos.
CAMPOS_NAO_MAPEADOS: Mapping[str, str] = {
    "field_provenance": "metadado deste pacote; o OCDS não tem campo de proveniência por campo",
    "rule_version": "metadado de processamento, não do contrato",
    "model_version": "metadado de processamento, não do contrato",
    "run_id": "metadado de processamento, não do contrato",
    "source_hash": "metadado de processamento, não do contrato",
    "processed_at": "metadado de processamento, não do contrato",
    "is_summable": "derivado da regra financeira deste pacote, não do padrão",
    "summability_reason": "idem",
    "amount_magnitude": "idem; o OCDS assume que value.amount é o valor do contrato",
    "financial_nature": "idem",
    "is_annulment": "o OCDS não tem etiqueta de anulação; o ato anulado é omitido da exportação",
    "is_republication": "metadado editorial da fonte",
    "publication_id": "identificador interno da publicação de origem",
    "edition_date": "data da edição do diário, distinta da data do ato",
}

DIAS_POR_MES = 30


def _texto(valor: Any) -> str | None:
    """Devolve texto não vazio, ou None. Campo vazio é omitido, nunca exportado como ''."""
    if valor is None:
        return None
    s = str(valor).strip()
    return s or None


def _data_hora(valor: Any) -> str | None:
    """
    Converte data para o `date-time` ISO 8601 que o OCDS exige.

    A prata guarda `signature_date` como data, sem hora. O OCDS pede data-hora com fuso. A
    conversão fixa meia-noite em UTC e não inventa precisão: a hora do ato não é publicada,
    e qualquer outro horário seria invenção.
    """
    if valor is None:
        return None
    if isinstance(valor, dt.datetime):
        d = valor.date()
    elif isinstance(valor, dt.date):
        d = valor
    else:
        s = str(valor).strip()[:10]
        if not s:
            return None
        try:
            d = dt.date.fromisoformat(s)
        except ValueError:
            return None
    return f"{d.isoformat()}T00:00:00Z"


def _valor(amount: Any) -> dict[str, Any] | None:
    """O objeto `Value` do OCDS: montante mais moeda. Sem montante, não há objeto."""
    if amount is None:
        return None
    if isinstance(amount, Decimal):
        montante = float(amount)
    else:
        try:
            montante = float(amount)
        except (TypeError, ValueError):
            return None
    return {"amount": montante, "currency": MOEDA}


def _ocid(linha: Mapping[str, Any]) -> str:
    """
    Monta o `ocid` a partir do número do processo, com queda para o identificador do ato.

    O OCDS quer um identificador do **processo**, e o número do processo administrativo é o
    que mais se aproxima disso na fonte: atos diferentes do mesmo processo compartilham-no, o
    que é exatamente a agregação que o padrão espera. Quando o ato não declara processo, o
    identificador do ato é usado, e aí cada ato vira um processo de um ato só.
    """
    processo = _texto(linha.get("process_number"))
    interno = processo or _texto(linha.get("contract_id")) or "sem-identificador"
    # O padrão proíbe '#' no identificador e recomenda evitar '/' e espaço, comuns em número
    # de processo administrativo.
    seguro = interno.replace("#", "-").replace("/", "-").replace(" ", "")
    return f"{PREFIXO_OCID}-{seguro}"


def _identificador_estavel(prefixo: str, nome: str) -> str:
    """
    Identificador de parte derivado do nome, estável entre execuções.

    O OCDS exige que cada parte tenha `id` e que `buyer` e `suppliers` o referenciem. Quando
    a fonte não publica documento fiscal, o nome é a única base disponível.

    Usa SHA-256 e não `hash()` da linguagem: `hash()` de texto é aleatorizado por processo,
    então duas exportações do mesmo dado produziriam identificadores diferentes.
    """
    digest = hashlib.sha256(nome.encode("utf-8")).hexdigest()[:12]
    return f"{prefixo}-{digest}"


def _partes(linha: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    As organizações do ato: o órgão que contrata e a empresa contratada.

    O OCDS usa `parties` com papéis, e referencia essas partes por `id` em `buyer` e em
    `awards[].suppliers`. O documento fiscal é o identificador natural da contratada; quando
    ele falta, a parte é exportada só com nome.
    """
    partes: list[dict[str, Any]] = []

    orgao = _texto(linha.get("agency"))
    if orgao:
        partes.append({
            "id": _identificador_estavel("orgao", orgao),
            "name": orgao,
            "roles": ["buyer"],
        })

    contratada = _texto(linha.get("contractor_name"))
    if contratada:
        documento = _texto(linha.get("contractor_tax_id"))
        # O documento fiscal (CNPJ ou CPF) é o identificador natural da parte. É dado público
        # — o Diário Oficial publica o contrato com o documento da contratada, inclusive de
        # pessoa física — então entra como está, sem redação nem pseudonimização. Quando a
        # fonte não traz documento, cai no identificador derivado do nome.
        parte: dict[str, Any] = {
            "id": documento or _identificador_estavel("fornecedor", contratada),
            "name": contratada,
            "roles": ["supplier"],
        }
        if documento:
            # `scheme` identifica a autoridade emissora. CNPJ e CPF são emitidos pela Receita
            # Federal; o OCDS pede o código da lista `organizationIdentifierScheme`, que para
            # o Brasil é BR-RFB.
            parte["identifier"] = {"scheme": "BR-RFB", "id": documento}
        partes.append(parte)

    return partes


def _periodo(linha: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    O período do contrato, a partir do prazo em meses.

    O OCDS aceita `durationInDays` ou datas de início e fim. A fonte publica prazo em meses,
    então a conversão usa 30 dias por mês — a mesma aproximação que o restante do código usa
    para prazo. Quando há data de assinatura, ela entra como início.
    """
    meses = linha.get("term_months")
    inicio = _data_hora(linha.get("signature_date"))
    if meses is None and inicio is None:
        return None
    periodo: dict[str, Any] = {}
    if inicio:
        periodo["startDate"] = inicio
    if meses is not None:
        try:
            periodo["durationInDays"] = int(meses) * DIAS_POR_MES
        except (TypeError, ValueError):
            pass
    return periodo or None


def release_de_ato(linha: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    Converte uma linha da prata consolidada em um release do OCDS.

    Devolve None para ato anulado: o OCDS não tem etiqueta para anulação, e exportar o ato
    como contrato faria a base somar um valor que foi desfeito. Também devolve None quando
    o ato não tem `contract_id` ou não tem nenhuma data (assinatura ou edição).

    O release traz `awards` além de `contracts` porque o OCDS exige que todo contrato aponte
    para a adjudicação que o originou (`Contract.awardID` é obrigatório). A fonte não publica
    a adjudicação em separado, então ela é derivada do próprio ato, com o mesmo identificador.
    """
    if linha.get("is_annulment"):
        return None

    identificador = _texto(linha.get("contract_id"))
    if not identificador:
        return None

    data = _data_hora(linha.get("signature_date")) or _data_hora(linha.get("edition_date"))
    if not data:
        return None

    tipo = (_texto(linha.get("act_kind")) or "").lower()
    etiqueta = TAG_ALTERACAO if "aditamento" in tipo or "aditivo" in tipo else TAG_CONTRATO

    valor = _valor(linha.get("amount_brl"))
    objeto = _texto(linha.get("object"))
    partes = _partes(linha)
    fornecedores = [
        {"id": p["id"], "name": p["name"]} for p in partes if "supplier" in p["roles"]
    ]
    comprador = next((p for p in partes if "buyer" in p["roles"]), None)

    adjudicacao: dict[str, Any] = {"id": identificador, "status": "active"}
    if fornecedores:
        adjudicacao["suppliers"] = fornecedores
    if valor:
        adjudicacao["value"] = valor
    if objeto:
        adjudicacao["description"] = objeto

    contrato: dict[str, Any] = {"id": identificador, "awardID": identificador}
    if objeto:
        contrato["title"] = objeto
    if valor:
        contrato["value"] = valor
    assinatura = _data_hora(linha.get("signature_date"))
    if assinatura:
        contrato["dateSigned"] = assinatura
    periodo = _periodo(linha)
    if periodo:
        contrato["period"] = periodo
    numero = _texto(linha.get("contract_number"))
    fundamento = _texto(linha.get("legal_basis"))
    descricao = " | ".join(x for x in (numero, fundamento) if x)
    if descricao:
        contrato["description"] = descricao

    release: dict[str, Any] = {
        "ocid": _ocid(linha),
        "id": identificador,
        "date": data,
        "tag": [etiqueta],
        # Lista fechada do OCDS; `tender` é o único valor do padrão 1.1. O DOM-SP não publica
        # a modalidade de origem no extrato, então o valor obrigatório é usado como está.
        "initiationType": "tender",
        "awards": [adjudicacao],
        "contracts": [contrato],
    }
    if partes:
        release["parties"] = partes
    if comprador:
        release["buyer"] = {"id": comprador["id"], "name": comprador["name"]}
    return release


def pacote(
    releases: Sequence[Mapping[str, Any]],
    *,
    uri: str,
    publicador: str,
    publicado_em: dt.datetime | None = None,
) -> dict[str, Any]:
    """
    Embrulha os releases no `release package`, que é a unidade de publicação do OCDS.

    Os cinco campos obrigatórios do pacote são `uri`, `publisher`, `publishedDate`,
    `releases` e `version`. O `uri` identifica o próprio arquivo publicado; como ele só
    existe depois de publicado, quem chama informa.

    `publicado_em` é injetável para fixar o carimbo de tempo. Sem ele, usa
    a hora corrente.
    """
    quando = publicado_em or dt.datetime.now(dt.timezone.utc)
    return {
        "uri": uri,
        "version": VERSAO_OCDS,
        "publishedDate": quando.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "publisher": {"name": publicador},
        "releases": list(releases),
    }


def de_atos(
    linhas: Iterable[Mapping[str, Any]],
    *,
    uri: str,
    publicador: str,
    publicado_em: dt.datetime | None = None,
) -> tuple[dict[str, Any], int]:
    """
    Converte atos em um pacote OCDS. Devolve o pacote e a contagem de atos omitidos.

    Omitido é o ato que `release_de_ato` recusa: anulação, ou ato sem identificador ou sem
    data. A contagem é devolvida em vez de registrada em log para que quem exporta possa
    conferir o balanço — atos que entraram mais atos omitidos tem de fechar com o total.
    """
    releases: list[dict[str, Any]] = []
    omitidos = 0
    for linha in linhas:
        r = release_de_ato(linha)
        if r is None:
            omitidos += 1
        else:
            releases.append(r)
    return pacote(releases, uri=uri, publicador=publicador, publicado_em=publicado_em), omitidos


def _carregar_esquemas(diretorio: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lê as cópias locais dos esquemas oficiais."""
    pkg = json.loads((diretorio / "release-package-schema.json").read_text(encoding="utf-8"))
    rel = json.loads((diretorio / "release-schema.json").read_text(encoding="utf-8"))
    return pkg, rel


def validar(
    pacote_ocds: Mapping[str, Any], diretorio_esquemas: Path | str = DIRETORIO_ESQUEMAS
) -> list[str]:
    """
    Valida o pacote contra o esquema oficial do OCDS. Devolve a lista de erros, vazia se válido.

    A validação usa `jsonschema` contra as cópias em `infraestrutura/esquemas/ocds-1.1.5/`.
    O esquema do pacote referencia o esquema do release por URL absoluta; o registro abaixo
    resolve essa URL para a cópia local, de modo que a validação não dependa de rede.

    Os erros vêm ordenados por caminho, e não por ordem de descoberta, para que duas execuções
    sobre o mesmo dado produzam a mesma lista.
    """
    from jsonschema import Draft4Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT4

    pkg, rel = _carregar_esquemas(Path(diretorio_esquemas))

    # Os esquemas do OCDS 1.1 declaram draft-04. O registro associa a URL que o esquema do
    # pacote referencia à cópia local do esquema do release.
    registro = Registry().with_resource(
        URL_ESQUEMA_RELEASE, Resource.from_contents(rel, default_specification=DRAFT4)
    )
    validador = Draft4Validator(pkg, registry=registro)
    erros = sorted(validador.iter_errors(dict(pacote_ocds)),
                   key=lambda e: (list(e.absolute_path), e.message))
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '(raiz)'}: {e.message}" for e in erros
    ]
