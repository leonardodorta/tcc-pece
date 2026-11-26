"""
edicao.py — a fronteira do bronze: como os bytes da fonte se tornam dado

Este módulo é a única parte autorizada a interpretar bytes crus da fonte. A jusante,
tudo recebe uma `Edicao` já decodificada e com o metadado já normalizado.

`fonte` e `texto` são fronteiras independentes de propósito: `fonte` interpreta os
bytes e o metadado de uma edição (decodificação, regime de codificação, correção de
dupla codificação nos sete campos curtos); `texto` interpreta o HTML do corpo
(`conteudo`) de uma publicação. A correção de dupla codificação é aplicada só aos sete
campos de metadado; o corpo não é alterado por ela.

Este módulo não faz HTTP: a obtenção da resposta viva é de `fonte.baixar`. O que este módulo
expõe é `carregar_bytes(bruto, data) -> Edicao`; a única obtenção de bytes que ele faz é
a de arquivo local.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# A FORMA DA FONTE

# A raiz do JSON é `{"edicao": [ ... ]}`. Não se aceita chave alternativa: tolerar
# uma chave que a fonte nunca usou esconderia uma mudança real de contrato.
CHAVE_RAIZ = "edicao"

# Os oito campos na ordem em que a fonte os emite. Os sete primeiros são metadado —
# curtos, sem HTML — e é neles, e só neles, que a dupla codificação bate. O oitavo é
# o corpo, que é HTML e é tratado pelo pacote `texto`.
CAMPOS_METADADOS: tuple[str, ...] = (
    "veiculo", "orgao", "unidade", "serie", "processo", "documento", "link",
)
CAMPO_CORPO = "conteudo"
CAMPOS: tuple[str, ...] = (*CAMPOS_METADADOS, CAMPO_CORPO)


# DECODIFICAÇÃO

# Ordem de tentativa. UTF-8 primeiro porque é o que as edições salvas são; latin-1
# depois porque é o que a resposta HTTP declara e às vezes cumpre. `latin-1` nunca
# levanta exceção — toda sequência de bytes é latin-1 válido —, então ele é
# necessariamente o último e não existe o caso "nenhum funcionou".
ENCODINGS: tuple[str, ...] = ("utf-8", "latin-1")

# A marca de ordem de bytes (BOM), por `chr` e não por literal: U+FEFF é de largura
# zero e seria um caractere invisível entre aspas no código-fonte.
_BOM = chr(0xFEFF)


def decodificar(bruto: bytes) -> tuple[str, str]:
    """
    Bytes -> (texto, nome_do_encoding).

    O nome do encoding é devolvido para virar coluna de proveniência: se uma edição só
    decodificar em latin-1, isso aparece na conferência.
    """
    for enc in ENCODINGS:
        try:
            return bruto.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # Inalcançável: latin-1 aceita qualquer byte.
    raise AssertionError("latin-1 deveria aceitar qualquer byte")


# REGIME DE CODIFICAÇÃO

REGIME_DIRETO = "direto"        # UTF-8 uma vez, acento correto
REGIME_DUPLA = "dupla"          # UTF-8 escrito e relido como latin-1

# A assinatura da dupla codificação: um byte-guia latin-1 (Â=0xC2 ou Ã=0xC3) seguido
# de um byte de continuação UTF-8 (0x80–0xBF). "ã" (U+00E3) em UTF-8 são os bytes
# C3 A3; lidos como latin-1 viram "Ã£" — exatamente este par. Nenhum português
# legítimo produz "Â"/"Ã" seguido dessa faixa.
#
# Montado por `chr()` de propósito: escrever a faixa 0x80–0xBF como caractere literal
# encheria o código-fonte de controles invisíveis.
_GUIAS = chr(0xC2) + chr(0xC3)
_CONTINUACAO = chr(0x80) + "-" + chr(0xBF)
_RE_ASSINATURA = re.compile("[" + _GUIAS + "][" + _CONTINUACAO + "]")

# Fração de publicações com a assinatura acima da qual a edição é declarada em dupla
# codificação. Espera-se uma taxa próxima de 0% ou próxima de 100%;
# o meio do caminho entre elas é o limiar.
LIMIAR_REGIME = 0.5

# Quantas vezes desfazer a codificação. Uma passada basta no caso normal. O limite fica
# como margem, e cada passada é contada: se uma edição precisar de mais de uma, isso
# aparece na proveniência em vez de sumir num laço.
MAX_PASSADAS = 3


def tem_assinatura(s: str | None) -> bool:
    """Verdadeiro quando o texto carrega a marca de UTF-8 relido como latin-1."""
    return bool(s) and _RE_ASSINATURA.search(s) is not None


def desfazer_dupla_codificacao(s: str) -> tuple[str, int]:
    """
    Reverte UTF-8 relido como latin-1, e devolve também quantas passadas gastou.

    A contagem permite distinguir uma edição em tripla codificação de uma em dupla
    sem precisar reprocessar.

    Duas razões para uma passada não acontecer, ambas devolvidas como estão em vez de
    mascaradas:

    - `encode("latin-1")` falha quando o texto tem caractere acima de U+00FF (um
      hífen inseparável U+2011, um `€`): não é dupla codificação, e insistir
      corromperia o texto.
    - `decode("utf-8")` falha quando os bytes resultantes não formam UTF-8 válido: a
      assinatura era coincidência, e o texto original é o certo.

    O ponto fixo (candidato == s) também encerra o laço.
    """
    passadas = 0
    for _ in range(MAX_PASSADAS):
        if not tem_assinatura(s):
            break
        try:
            candidato = s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if candidato == s:
            break
        s, passadas = candidato, passadas + 1
    return s, passadas


def detectar_regime(publicacoes: list[dict]) -> tuple[str, float]:
    """
    O regime de codificação da edição e a taxa de assinatura que o sustenta.

    Olha só os sete campos de metadado: o corpo já está escapado em entidade HTML nos
    dois regimes, então incluí-lo só adicionaria ruído.

    Levanta exceção na faixa do meio (entre 0 e `LIMIAR_REGIME`): uma edição com,
    digamos, 30% das publicações marcadas contradiz a premissa de que o regime é
    propriedade da edição, e nesse caso nenhuma das duas respostas está certa — o
    certo é parar e olhar, não escolher em silêncio.

    Partição vazia é `direto` com taxa 0,0 — dia legítimo sem atos, não falha.
    """
    if not publicacoes:
        return REGIME_DIRETO, 0.0

    marcadas = sum(
        1 for p in publicacoes
        if any(tem_assinatura(_como_texto(p.get(c))) for c in CAMPOS_METADADOS)
    )
    taxa = marcadas / len(publicacoes)

    if taxa == 0.0:
        return REGIME_DIRETO, taxa
    if taxa >= LIMIAR_REGIME:
        return REGIME_DUPLA, taxa
    raise ValueError(
        f"regime ambíguo: {marcadas} de {len(publicacoes)} publicações "
        f"({taxa:.1%}) com assinatura de dupla codificação. Esperava-se "
        f"taxa próxima de 0% ou de 100%; uma taxa intermediária significa que a codificação "
        f"deixou de ser propriedade da edição e a correção precisa ser revista antes "
        f"de aplicada."
    )


def _como_texto(valor: object) -> str | None:
    """
    O valor de um campo como texto, ou None.

    A fonte pode emitir `documento` como número e `link` como null; a assinatura
    e a reversão operam sobre `str`, e passar um `int` ao regex é `TypeError`
    silencioso — pior que um None explícito.
    """
    if valor is None:
        return None
    return valor if isinstance(valor, str) else str(valor)


def normalizar_metadados(
    publicacoes: list[dict], regime: str,
) -> tuple[list[dict], int, int, int]:
    """
    Aplica a correção do regime aos sete metadados, devolvendo (publicações,
    corrigidos, resistentes, max_passadas).

    No regime `direto` não há o que fazer e a lista volta como veio — os mesmos
    objetos, sem cópia.

    No regime `dupla`, os sete campos de metadado passam pela reversão e o corpo não
    é tocado (é tratado pelo pacote `texto`). `corrigidos` conta os campos que a reversão
    de fato mudou; `resistentes` conta os que mantiveram a assinatura depois das
    passadas; `max_passadas` é o maior número de passadas que uma correção exigiu.
    """
    if regime == REGIME_DIRETO:
        return publicacoes, 0, 0, 0

    saida: list[dict] = []
    corrigidos = resistentes = max_passadas = 0
    for pub in publicacoes:
        novo = dict(pub)
        for campo in CAMPOS_METADADOS:
            valor = _como_texto(pub.get(campo))
            if not valor or not tem_assinatura(valor):
                continue
            corrigido, passadas = desfazer_dupla_codificacao(valor)
            if passadas:
                novo[campo] = corrigido
                corrigidos += 1
                max_passadas = max(max_passadas, passadas)
            if tem_assinatura(_como_texto(novo[campo])):
                resistentes += 1
        saida.append(novo)
    return saida, corrigidos, resistentes, max_passadas


# A EDIÇÃO


@dataclass(frozen=True)
class Edicao:
    """
    Uma edição lida, com a proveniência de *como* ela foi lida.

    `frozen` porque uma edição do bronze é imutável (revisão é nova publicação, não
    alteração da anterior).

    Os campos de diagnóstico permitem responder "por que este metadado está assim?"
    sem reprocessar:

    - `numero_edicao` é campo, não a data, porque pode haver edição extraordinária no
      mesmo dia da ordinária (numeração distinta); a data sozinha não identifica a
      edição. Pode ser vazio quando a fonte não o traz no arquivo — a data continua
      sendo a chave de partição, o número desambigua ordinária × extraordinária.
    - `regime`, `encoding`, `ponto_virgula` e `taxa_assinatura` registram a
      correlação entre sinal e codificação como dado, para que a quebra dessa
      correlação seja visível no dia em que acontecer.
    - `campos_corrigidos` e `campos_resistentes` tornam a correção auditável por
      edição.
    """

    data: str
    numero_edicao: str
    publicacoes: list[dict]
    regime: str
    encoding: str
    campos_corrigidos: int
    bytes_origem: int
    taxa_assinatura: float
    ponto_virgula: bool = True
    campos_resistentes: int = 0
    passadas: int = 0

    @property
    def divergencia_de_sinal(self) -> bool:
        """
        Verdadeiro quando o `;` final e a codificação discordam.

        A correlação esperada é: ausência do `;` anda com dupla codificação,
        presença do `;` anda com regime direto. "Concordam" significa
        `ponto_virgula == (regime é direto)`.

        É um sinal esperado, não invariante da fonte: por isso é uma propriedade lida, não
        um `assert` que derrubaria o pipeline se deixasse de valer.
        """
        return self.ponto_virgula != (self.regime == REGIME_DIRETO)


# A proveniência vive nos atributos de `Edicao`, não num método agregador: o bronze é
# uma linha por publicação, não uma tabela por edição, e quem consome a `Edicao` lê
# `regime`, `campos_corrigidos` e `bytes_origem` direto dos atributos.


# MONTAGEM

_RE_DATA_ARQUIVO = re.compile(r"(\d{4}-\d{2}-\d{2})")

# O número da edição, quando a fonte o traz no nome do arquivo depois da data. É
# opcional: arquivos nomeados só por data (`dom_2026-05-04.json`) não trazem número,
# então o padrão só captura se houver um sufixo `_<n>`. Sem ele o número fica vazio, e
# a data continua sendo a chave de partição.
_RE_NUMERO_ARQUIVO = re.compile(r"\d{4}-\d{2}-\d{2}[_-](\d+)")


def _extrair_lista(dados: object) -> list[dict]:
    """
    Extrai a lista de publicações da raiz do JSON.

    Recusa forma inesperada em vez de devolver lista vazia: devolver vazio faria uma
    mudança de contrato da fonte virar uma partição de zero linha — indistinguível de
    um dia sem atos, que não dispara nada.
    """
    if isinstance(dados, dict):
        lista = dados.get(CHAVE_RAIZ)
        if isinstance(lista, list):
            return lista
        raise ValueError(
            f"raiz do JSON sem a chave '{CHAVE_RAIZ}' como lista; "
            f"chaves presentes: {sorted(dados)[:10]}"
        )
    if isinstance(dados, list):
        return dados
    raise ValueError(
        f"raiz do JSON é {type(dados).__name__}, esperado objeto ou lista"
    )


def carregar_bytes(bruto: bytes, data: str, numero_edicao: str = "") -> Edicao:
    """
    Uma edição, dos bytes crus à lista de publicações com metadado normalizado.

    A ordem dos passos é obrigatória e cada um depende do anterior:

    1. decodificar — bytes viram texto; o `latin-1` é o último recurso.
    2. limpar as bordas — o BOM só é um caractere (não byte) depois de decodificado,
       e por isso o `lstrip(_BOM)` vem aqui e não sobre os bytes.
    3. registrar o `;` final — antes de removê-lo, porque a presença dele é o sinal
       independente que a proveniência guarda ao lado do regime.
    4. parsear — o `;` final quebra o `json.loads`, então é removido só para o
       parse; a informação de que ele existia já foi guardada no passo 3.
    5. detectar o regime — olhando campo por campo do metadado.
    6. normalizar — a correção é decisão da edição inteira, tomada no passo 5, não
       da string isolada.

    `data` e `numero_edicao` vêm de fora (do nome do arquivo, ou do pedido HTTP na
    coleta) porque o JSON da fonte não os traz em campo algum. A data é a chave de
    partição; o número desambigua ordinária × extraordinária.
    """
    texto, encoding = decodificar(bruto)
    texto = texto.lstrip(_BOM).strip()
    ponto_virgula = texto.endswith(";")
    dados = json.loads(texto.rstrip(";").rstrip())
    publicacoes = _extrair_lista(dados)
    regime, taxa = detectar_regime(publicacoes)
    publicacoes, corrigidos, resistentes, passadas = normalizar_metadados(
        publicacoes, regime,
    )

    return Edicao(
        data=data,
        numero_edicao=numero_edicao,
        publicacoes=publicacoes,
        regime=regime,
        encoding=encoding,
        campos_corrigidos=corrigidos,
        bytes_origem=len(bruto),
        taxa_assinatura=taxa,
        ponto_virgula=ponto_virgula,
        campos_resistentes=resistentes,
        passadas=passadas,
    )


# LEITURA DE ARQUIVO LOCAL
# A única obtenção de bytes que este módulo faz. HTTP é de `fonte.baixar` (ver cabeçalho).


def data_do_arquivo(caminho: Path) -> str:
    """
    A data no nome do arquivo (`dom_2026-05-04.json` -> `2026-05-04`).

    A data vem do nome e não do conteúdo porque o JSON da fonte não a traz em campo
    algum. Ela é a chave de partição, então um nome sem data é erro de ingestão.
    """
    achado = _RE_DATA_ARQUIVO.search(caminho.name)
    if not achado:
        raise ValueError(
            f"'{caminho.name}' não traz data em AAAA-MM-DD; a data da edição é a "
            f"chave de partição e não pode ser inferida do conteúdo"
        )
    return achado.group(1)


def numero_do_arquivo(caminho: Path) -> str:
    """
    O número da edição no nome do arquivo, se houver (`..._12345.json` -> `12345`).

    Vazio quando ausente: arquivos nomeados só por data não trazem número.
    """
    achado = _RE_NUMERO_ARQUIVO.search(caminho.name)
    return achado.group(1) if achado else ""


def carregar(caminho: str | Path) -> Edicao:
    """Lê uma edição de arquivo local, com data e número tirados do nome."""
    caminho = Path(caminho)
    return carregar_bytes(
        caminho.read_bytes(),
        data_do_arquivo(caminho),
        numero_do_arquivo(caminho),
    )


def arquivos(diretorio: str | Path) -> list[Path]:
    """Os arquivos de edição de um diretório, em ordem de data (pelo nome)."""
    return sorted(Path(diretorio).glob("dom_*.json"))


def iterar(diretorio: str | Path) -> Iterator[Edicao]:
    """
    Percorre as edições de um diretório, uma por vez.

    Gerador, não lista: materializar todas as edições juntas custa muita memória em
    dicionários Python. Também é o que permite a quem chama distribuir caminhos entre
    processos em vez de serializar dados já carregados.
    """
    for arq in arquivos(diretorio):
        yield carregar(arq)
