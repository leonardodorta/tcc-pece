"""
esquema — a declaração autoritativa das colunas. Fonte única.

Deste módulo saem, por geração e nunca por transcrição:

    esquema Python      `campos(CONJUNTO)` — o que o pipeline local usa
    esquema Spark       `struct_type(CONJUNTO)` — o que os jobs Glue usam
    tabela Glue         `colunas_catalogo(CONJUNTO)` — o que o CloudFormation usa
    DDL Athena          `ddl(CONJUNTO, ...)` — o que a documentação publica
    dicionário de dados `markdown(CONJUNTO)` — a documentação publicada

## Por que uma fonte só

Declarar o esquema em mais de um lugar — um `StructType` no job, uma lista de `Columns` no
template, um `ESQUEMA` à parte — exige lembrar de todos a cada mudança, e o modo de falhar é o
pior possível: acrescentar uma coluna faz o Parquet ganhá-la, o catálogo não, e o Athena
devolve a tabela **sem** a coluna nova, silenciosamente, para sempre. Nada quebra. O dado
simplesmente não aparece.

O `StructType`, as colunas do catálogo e o DDL não divergem por construção, porque os três saem daqui.

## Por que os nomes são em inglês

São **valor de dado consultado por Athena**: o esquema público do lago. A prosa deste
repositório é em português porque documenta decisão; o esquema é em inglês porque é
interface. As duas escolhas são deliberadas e não se misturam.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import lago
from . import vocabulario as V

# Os tipos
#
# Um vocabulário próprio, e não os nomes do Spark nem os do Athena, porque o mesmo tipo
# lógico tem nomes diferentes nos dois e a tradução tem de morar num lugar só.

TEXTO = "texto"
DATA = "data"
MOMENTO = "momento"
INTEIRO = "inteiro"
LONGO = "longo"
DECIMAL = "decimal"
REAL = "real"
LOGICO = "logico"
MAPA_TEXTO = "mapa_texto"

# `DECIMAL` e não `REAL` para dinheiro. O float introduz erro de representação que aparece na
# soma: 0,1 + 0,2 não é 0,3 em binário, e um agregado mensal de milhares de linhas acumula a
# diferença até ela ser visível no relatório. `decimal(18,2)` comporta R$ 9,99 quatrilhões.
_ATHENA: dict[str, str] = {
    TEXTO: "string",
    DATA: "date",
    MOMENTO: "timestamp",
    INTEIRO: "int",
    LONGO: "bigint",
    DECIMAL: "decimal(18,2)",
    REAL: "double",
    LOGICO: "boolean",
    MAPA_TEXTO: "map<string,string>",
}

_SPARK: dict[str, str] = {
    TEXTO: "StringType",
    DATA: "DateType",
    MOMENTO: "TimestampType",
    INTEIRO: "IntegerType",
    LONGO: "LongType",
    DECIMAL: "DecimalType",
    REAL: "DoubleType",
    LOGICO: "BooleanType",
    MAPA_TEXTO: "MapType",
}

PRECISAO_DECIMAL = 18
ESCALA_DECIMAL = 2


@dataclass(frozen=True)
class Coluna:
    """
    Uma coluna: nome, tipo, e o que ela responde.

    `nota` vai para o `Comment` da coluna no Glue Data Catalog. É o único texto deste
    repositório que chega a quem consulta pelo Athena sem ter lido nada — então ela diz *o
    que o campo significa*, não como foi calculado.

    `valores` é o vocabulário fechado da coluna, quando ela tem um, e vem sempre de
    `vocabulario.py` — nunca escrito à mão aqui. A nota publicada é composta a partir
    dele, e é essa composição que elimina a divergência entre o que a coluna promete e
    o que o domínio produz.
    """

    nome: str
    tipo: str
    nota: str
    valores: tuple[str, ...] = ()
    obrigatoria: bool = False

    @property
    def athena(self) -> str:
        return _ATHENA[self.tipo]

    @property
    def nota_completa(self) -> str:
        """A nota como o catálogo a recebe: a descrição, e o vocabulário quando houver."""
        if not self.valores:
            return self.nota
        return f"{self.nota}. Valores: {', '.join(self.valores)}"


# Procedência — as colunas que toda linha de toda camada carrega
#
# Cada execução registra ID, versão das regras, versão do modelo, hash da origem e instante.
# Sem isso, quando um número do relatório divergir do lago não há como saber se o lago foi
# reprocessado por regra mais nova.

PROCEDENCIA: tuple[Coluna, ...] = (
    Coluna("rule_version", TEXTO,
           "versão do pacote domsp que produziu a linha; muda quando a regra muda",
           obrigatoria=True),
    Coluna("model_version", TEXTO,
           "versão do modelo que participou da linha, ou 'nenhum' quando nenhum participou",
           obrigatoria=True),
    Coluna("run_id", TEXTO,
           "identificador da execução da máquina de estados que produziu a linha"),
    Coluna("source_hash", TEXTO,
           "SHA-256 do HTML de origem; muda se a fonte republicar conteúdo diferente",
           obrigatoria=True),
    Coluna("processed_at", MOMENTO,
           "instante UTC em que a linha foi gravada", obrigatoria=True),
)

# `processed_at` é UTC e o nome não diz. Limitação conhecida: `timestamp` do Athena é sem fuso,
# e gravar horário de São Paulo faria a coluna saltar uma hora duas vezes por ano dentro de
# uma série temporal. Converter para exibição é decisão de apresentação, não de armazenamento.

PARTICAO = Coluna(
    lago.PARTICAO, DATA,
    "data da EDIÇÃO em que a publicação saiu — não a da assinatura. "
    "É a chave de partição e de reprocessamento",
    obrigatoria=True,
)


# Bronze — o que a fonte devolveu
#
# Tipado: a bronze é Parquet com colunas declaradas, e não JSON cru; `body_html` guarda o HTML
# **intocado** — a fidelidade está no conteúdo da coluna, não na ausência de esquema.

PUBLICACAO: tuple[Coluna, ...] = (
    Coluna("publication_id", TEXTO,
           "identificador da publicação na fonte, ou SHA-256 do corpo quando a fonte não o "
           "declara", obrigatoria=True),
    Coluna("edition_date", TEXTO,
           "data da edição como a fonte a imprimiu, sem normalizar"),
    Coluna("edition_number", TEXTO, "número da edição, quando a fonte o declara"),
    Coluna("section", TEXTO,
           "caderno em que a publicação saiu; o art. 13, § único do Decreto 62.177/2023 "
           "lista sete, e o V é Negócios"),
    Coluna("agency", TEXTO, "órgão publicador, como declarado pela fonte"),
    Coluna("title", TEXTO, "série do ato, o campo que a fonte chama de 'serie'"),
    Coluna("body_html", TEXTO,
           "corpo da publicação em HTML, exatamente como a fonte devolveu: sem limpeza, sem "
           "normalizar espaço, sem decodificar entidade", obrigatoria=True),
    Coluna("encoding_regime", TEXTO,
           "regime de codificação detectado na edição: 'direto' quando o UTF-8 veio correto, "
           "'dupla' quando veio gravado e relido como latin-1 e foi desfeito"),
)


# Prata — o ato, tipado, com proveniência por campo
#
# As 21 colunas próprias do ato, seguidas das 5 de `PROCEDENCIA`: 26 no total.

CONTRATO: tuple[Coluna, ...] = (
    Coluna("contract_id", TEXTO,
           "SHA-256 truncado da identidade canônica do ato; estável entre reexecuções",
           obrigatoria=True),
    Coluna("publication_id", TEXTO,
           "liga de volta a bronze_publication", obrigatoria=True),
    Coluna("edition_date", DATA, "data da edição, normalizada"),
    Coluna("agency", TEXTO, "órgão contratante"),
    Coluna("act_kind", TEXTO, "espécie do ato", valores=V.TIPOS),
    Coluna("financial_nature", TEXTO,
           "natureza financeira do ato — o que decide se o valor pode somar",
           valores=V.NATUREZAS),
    Coluna("is_annulment", LOGICO,
           "o ato anula outro ato; um valor anulado nunca foi gasto"),
    Coluna("is_republication", LOGICO,
           "o ato republica outro; o art. 12, § 2º do decreto manda que revisão seja nova "
           "publicação, então republicação não é correção e sim linha nova"),
    Coluna("contractor_name", TEXTO,
           "parte contratada, validada pela porta única de nome"),
    Coluna("contractor_tax_id", TEXTO,
           "CNPJ ou CPF só com dígitos, com dígito verificador conferido pelo módulo 11"),
    Coluna("object", TEXTO,
           "objeto contratado como a fonte o escreveu, truncado em 2.000 caracteres; "
           "determinístico, nunca inferido por modelo"),
    Coluna("contract_number", TEXTO, "número do contrato ou do ajuste"),
    Coluna("process_number", TEXTO, "número do processo administrativo"),
    Coluna("amount_brl", DECIMAL,
           "valor em reais. NÃO somar sem o filtro de is_summable"),
    Coluna("amount_magnitude", TEXTO,
           "grandeza semântica do valor lido — total vence mensal e unitário na eleição",
           valores=V.GRANDEZAS),
    Coluna("is_summable", LOGICO,
           "o valor satisfaz a regra financeira: natureza gasto efetivo, grandeza "
           "total ou estimada, e ato não anulado. Aditivo e apostilamento têm natureza "
           "alteracao e não são somáveis", obrigatoria=True),
    Coluna("summability_reason", TEXTO,
           "por que o valor não pode somar, quando não pode",
           valores=None or ()),  # preenchido abaixo, depois do import de financeiro
    Coluna("signature_date", DATA,
           "data de assinatura; nula quando a fonte não declarou OU quando a data lida caiu "
           "fora da faixa de plausibilidade — a proveniência distingue os dois casos"),
    Coluna("term_months", INTEIRO,
           "prazo normalizado em meses; dias convertidos pela regra declarada de 30"),
    Coluna("legal_basis", TEXTO,
           "artigo citado da Lei 14.133/2021, quando a fonte o declara"),
    Coluna("field_provenance", MAPA_TEXTO,
           "de que camada veio cada campo preenchido. Sem este mapa a cobertura por camada "
           "não é recalculável",
           obrigatoria=True),
    *PROCEDENCIA,
)

DESCARTE: tuple[Coluna, ...] = (
    Coluna("publication_id", TEXTO, "a publicação recusada", obrigatoria=True),
    Coluna("section", TEXTO, "caderno"),
    Coluna("title", TEXTO, "série"),
    Coluna("reason", TEXTO, "por que foi recusada", valores=V.MOTIVOS_DESCARTE),
    Coluna("stage", TEXTO,
           "em que estágio a recusa foi decidida; 'metadado' significa que não foi preciso "
           "abrir o HTML", valores=V.ESTAGIOS),
    *PROCEDENCIA,
)

RESIDUO: tuple[Coluna, ...] = (
    Coluna("contract_id", TEXTO, "o ato sem parte contratada", obrigatoria=True),
    Coluna("publication_id", TEXTO, "liga de volta ao bronze, de onde vem o texto"),
    Coluna("reason", TEXTO,
           "por que nenhuma camada determinística resolveu", valores=V.MOTIVOS_RESIDUO),
    Coluna("target_text", TEXTO,
           "o texto que o modelo deve ler: a prosa recortada, SEM o formulário. É o mesmo "
           "recorte que o gabarito usa como entrada", obrigatoria=True),
    *PROCEDENCIA,
)

INFERENCIA: tuple[Coluna, ...] = (
    Coluna("contract_id", TEXTO, "o ato inferido", obrigatoria=True),
    Coluna("entity_class", TEXTO, "classe do span", valores=("CONTRATADA", "CONTRATANTE")),
    Coluna("span_text", TEXTO, "o texto do span como o modelo o delimitou",
           obrigatoria=True),
    Coluna("span_start", INTEIRO, "posição inicial do span em target_text"),
    Coluna("span_end", INTEIRO, "posição final, exclusiva"),
    Coluna("score", REAL,
           "escore do span: o mínimo (não a média) das probabilidades das palavras que o "
           "compõem — o span só é aceito como certo se todas as suas palavras pertencem à "
           "entidade; entre 0 e 1, não é probabilidade calibrada"),
    Coluna("accepted", LOGICO,
           "o span passou pela porta de validação de nome — a mesma das camadas "
           "determinísticas", obrigatoria=True),
    Coluna("rejection", TEXTO, "a ressalva da porta, quando ela recusou"),
    Coluna("truncated", LOGICO,
           "target_text passou de 512 tokens e foi processado em janelas"),
    *PROCEDENCIA,
)


# Ouro — agregado com significado declarado

FORNECEDOR_MES: tuple[Coluna, ...] = (
    Coluna("year_month", TEXTO, "mês da EDIÇÃO no formato aaaa-mm", obrigatoria=True),
    Coluna("contractor_tax_id", TEXTO,
           "agrupa por documento e não por nome: a fonte escreve a mesma empresa de várias "
           "formas e agrupar por nome a multiplicaria", obrigatoria=True),
    Coluna("contractor_name", TEXTO, "a grafia mais frequente do mesmo documento"),
    Coluna("contract_count", LONGO, "quantos atos, somáveis ou não"),
    Coluna("summable_count", LONGO, "quantos atos entraram no total"),
    Coluna("total_amount_brl", DECIMAL,
           "soma apenas dos valores que satisfazem a regra financeira"),
    Coluna("agency_count", LONGO, "em quantos órgãos distintos"),
    *PROCEDENCIA,
)

ORGAO_MES: tuple[Coluna, ...] = (
    Coluna("year_month", TEXTO, "mês da edição no formato aaaa-mm", obrigatoria=True),
    Coluna("agency", TEXTO, "órgão contratante", obrigatoria=True),
    Coluna("contract_count", LONGO, "quantos atos"),
    Coluna("summable_count", LONGO, "quantos entraram no total"),
    Coluna("total_amount_brl", DECIMAL, "soma sob a regra financeira"),
    Coluna("supplier_count", LONGO, "de quantos fornecedores distintos"),
    *PROCEDENCIA,
)

QUALIDADE: tuple[Coluna, ...] = (
    Coluna("metric_name", TEXTO, "o que foi medido", obrigatoria=True),
    Coluna("stratum", TEXTO,
           "o estrato em que foi medido; 'agregado' é o corpus todo", valores=V.ESTRATOS),
    Coluna("numerator", LONGO, "quantos satisfizeram"),
    Coluna("denominator", LONGO, "de quantos", obrigatoria=True),
    Coluna("proportion", REAL, "numerador sobre denominador"),
    Coluna("ci_lower", REAL, "limite inferior do intervalo de 95%"),
    Coluna("ci_upper", REAL, "limite superior"),
    Coluna("ci_method", TEXTO,
           "método do intervalo; Wilson e não normal, porque com proporção perto de 0 ou 1 o "
           "normal produz limite fora de [0,1]", valores=V.METODOS_INTERVALO),
    Coluna("measured_at", MOMENTO, "instante UTC da medição", obrigatoria=True),
    *PROCEDENCIA,
)


# O mapa conjunto → colunas

_CAMPOS: dict[str, tuple[Coluna, ...]] = {
    lago.PUBLICACAO.tabela: PUBLICACAO,
    lago.CONTRATO.tabela: CONTRATO,
    lago.CONTRATO_FINAL.tabela: CONTRATO,
    lago.DESCARTE.tabela: DESCARTE,
    lago.RESIDUO.tabela: RESIDUO,
    lago.INFERENCIA_BRUTA.tabela: INFERENCIA,
    lago.INFERENCIA_ACEITA.tabela: INFERENCIA,
    lago.FORNECEDOR_MES.tabela: FORNECEDOR_MES,
    lago.ORGAO_MES.tabela: ORGAO_MES,
    lago.QUALIDADE.tabela: QUALIDADE,
}

# `silver_contract` e `silver_contract_final` compartilham as colunas de propósito: a
# consolidação não muda o contrato, só preenche onde havia nulo.


def campos(conjunto: lago.Conjunto) -> tuple[Coluna, ...]:
    """As colunas de dado do conjunto, sem a coluna de partição."""
    try:
        return _CAMPOS[conjunto.tabela]
    except KeyError:
        raise KeyError(
            f"conjunto {conjunto.tabela} não tem colunas declaradas em esquema.py. "
            f"Declarar aqui é o que mantém a fonte única"
        ) from None


def nomes(conjunto: lago.Conjunto) -> tuple[str, ...]:
    """Os nomes das colunas, incluindo a partição quando o conjunto é particionado."""
    base = tuple(c.nome for c in campos(conjunto))
    return (*base, PARTICAO.nome) if conjunto.particionado else base


def obrigatorias(conjunto: lago.Conjunto) -> tuple[str, ...]:
    """As colunas que nunca podem ser nulas."""
    return tuple(c.nome for c in campos(conjunto) if c.obrigatoria)


# Os geradores


def struct_type(conjunto: lago.Conjunto):
    """
    O `StructType` do Spark para o conjunto.

    Importa pyspark **dentro** da função, e não no topo: `esquema.py` é importado pelo
    pipeline local, que roda num laptop sem pyspark instalado. Um
    import no topo tornaria a fonte única do esquema indisponível justamente onde ela é mais
    verificada.
    """
    from pyspark.sql.types import (  # noqa: PLC0415
        ArrayType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        IntegerType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    construtor = {
        TEXTO: StringType,
        DATA: DateType,
        MOMENTO: TimestampType,
        INTEIRO: IntegerType,
        LONGO: LongType,
        REAL: DoubleType,
        LOGICO: BooleanType,
    }

    def tipo_de(coluna: Coluna):
        if coluna.tipo == DECIMAL:
            return DecimalType(PRECISAO_DECIMAL, ESCALA_DECIMAL)
        if coluna.tipo == MAPA_TEXTO:
            return MapType(StringType(), StringType())
        if coluna.tipo not in construtor:
            raise KeyError(f"tipo {coluna.tipo} sem construtor Spark")
        return construtor[coluna.tipo]()

    campos_spark = [
        StructField(c.nome, tipo_de(c), nullable=not c.obrigatoria)
        for c in campos(conjunto)
    ]
    if conjunto.particionado:
        # A partição entra como StringType e não DateType: o Spark grava o valor da partição
        # no NOME do diretório (`dt=2026-05-04`), e o caminho é texto. Declarar DateType faz
        # o Spark reescrever o diretório como `dt=2026-05-04 00:00:00`, que a projeção de
        # partição do Athena não casa. O catálogo declara `date`, e é ele que interpreta.
        campos_spark.append(StructField(PARTICAO.nome, StringType(), nullable=False))
    assert ArrayType  # noqa: S101  — só referencia o import `ArrayType`, que o mapa de tipos não usa
    return StructType(campos_spark)


def colunas_catalogo(conjunto: lago.Conjunto) -> list[dict[str, str]]:
    """
    As colunas no formato que `AWS::Glue::Table` espera, sem a partição.

    A partição fica de fora porque no Glue ela é `PartitionKeys` e não `Columns`, e declarar
    nos dois lugares faz a criação da tabela falhar com "duplicate column".
    """
    return [
        {"Name": c.nome, "Type": c.athena, "Comment": c.nota_completa}
        for c in campos(conjunto)
    ]


def particao_catalogo() -> list[dict[str, str]]:
    """A chave de partição no formato do Glue."""
    return [{"Name": PARTICAO.nome, "Type": _ATHENA[PARTICAO.tipo],
             "Comment": PARTICAO.nota}]


def propriedades_projecao(inicio: str) -> dict[str, str]:
    """
    As propriedades de projeção de partição do Athena.

    Projeção e não crawler: o esquema é conhecido e declarado aqui, então um crawler só
    poderia descobrir errado — e cobra por descoberta. A projeção também dispensa
    `MSCK REPAIR`, que é o passo que todo mundo esquece de rodar depois de gravar.

    `inicio` é a data da primeira edição do corpus. A ressalva da projeção: uma consulta a
    uma data fora da faixa devolve vazio em vez de erro, então a faixa tem de cobrir o corpus
    com folga no futuro — daí `NOW`.
    """
    return {
        "projection.enabled": "true",
        f"projection.{PARTICAO.nome}.type": "date",
        f"projection.{PARTICAO.nome}.format": "yyyy-MM-dd",
        f"projection.{PARTICAO.nome}.range": f"{inicio},NOW",
        f"projection.{PARTICAO.nome}.interval": "1",
        f"projection.{PARTICAO.nome}.interval.unit": "DAYS",
    }


def _escapar(s: str) -> str:
    """Escapa a nota para caber num comentário SQL entre apóstrofos."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def ddl(conjunto: lago.Conjunto, banco: str, raiz: str, inicio: str) -> str:
    """
    O `CREATE EXTERNAL TABLE` do conjunto, para a documentação e para o Athena.

    Gerado e não redigido: um DDL escrito à mão divergiria do `StructType` na primeira coluna
    acrescentada, e a divergência não tem sintoma.
    """
    corpo = ",\n".join(
        f"  `{c.nome}` {c.athena} COMMENT '{_escapar(c.nota_completa)}'"
        for c in campos(conjunto)
    )
    linhas = [
        f"CREATE EXTERNAL TABLE IF NOT EXISTS `{banco}`.`{conjunto.tabela}` (",
        corpo,
        ")",
        f"COMMENT '{_escapar(conjunto.descricao)}'",
    ]
    if conjunto.particionado:
        linhas.append(
            f"PARTITIONED BY (`{PARTICAO.nome}` {_ATHENA[PARTICAO.tipo]})"
        )
    linhas.append("STORED AS PARQUET")
    linhas.append(f"LOCATION '{conjunto.caminho(raiz)}'")

    propriedades = {"classification": "parquet", "parquet.compression": "SNAPPY"}
    if conjunto.particionado:
        propriedades.update(propriedades_projecao(inicio))
    tabela = ", ".join(f"'{k}'='{v}'" for k, v in propriedades.items())
    linhas.append(f"TBLPROPERTIES ({tabela})")
    return "\n".join(linhas) + ";"


def markdown(conjunto: lago.Conjunto) -> str:
    """O dicionário de dados do conjunto em Markdown, para a documentação publicada."""
    cabecalho = (
        f"### `{conjunto.tabela}`\n\n{conjunto.descricao}\n\n"
        f"| coluna | tipo | nulo? | significado |\n|---|---|---|---|\n"
    )
    corpo = "\n".join(
        f"| `{c.nome}` | {c.athena} | {'não' if c.obrigatoria else 'sim'} "
        f"| {c.nota_completa} |"
        for c in campos(conjunto)
    )
    if conjunto.particionado:
        corpo += (
            f"\n| `{PARTICAO.nome}` | {_ATHENA[PARTICAO.tipo]} | não "
            f"| {PARTICAO.nota} (partição) |"
        )
    return cabecalho + corpo + "\n"


def dicionario_completo(banco: str) -> str:
    """O dicionário de dados do lago inteiro, camada por camada."""
    partes = [f"# Dicionário de dados — `{banco}`\n"]
    for camada in lago.CAMADAS:
        partes.append(f"\n## Camada {camada}\n")
        for conjunto in lago.da_camada(camada):
            partes.append(markdown(conjunto))
    return "\n".join(partes)


# A coluna `summability_reason` só pode declarar seu vocabulário depois que `financeiro` é
# importado, e `financeiro` não pode ser importado no topo porque nada aqui precisa dele
# antes deste ponto — importar no topo criaria um ciclo se `financeiro` algum dia precisar de
# um nome de coluna. A substituição em tempo de importação mantém a fonte única: o
# vocabulário continua vindo de um lugar só.
def _preencher_vocabulario_de_somabilidade() -> None:
    from . import financeiro  # noqa: PLC0415

    global CONTRATO, _CAMPOS
    atualizadas = tuple(
        Coluna(c.nome, c.tipo, c.nota, financeiro.MOTIVOS, c.obrigatoria)
        if c.nome == "summability_reason"
        else c
        for c in CONTRATO
    )
    CONTRATO = atualizadas
    _CAMPOS[lago.CONTRATO.tabela] = atualizadas
    _CAMPOS[lago.CONTRATO_FINAL.tabela] = atualizadas


_preencher_vocabulario_de_somabilidade()
