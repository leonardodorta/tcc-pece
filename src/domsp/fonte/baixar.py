"""
baixar.py — como o DOM-SP entrega uma edição, e o que ele responde quando não há

O DOM-SP não publica API documentada: o endereço usado é o mesmo `POST` que o botão
"download" do site faz, `md_epubli_controlador.php?acao=edicao_download`, com a data em
três campos de formulário.

Sábado, domingo e feriado municipal não têm edição, e a fonte responde com 500 e corpo
vazio. Quem decide se é "não houve edição" ou falha real é o TAMANHO do corpo, não o
código HTTP — a distinção está em `SEM_EDICAO`.

Só biblioteca padrão: nada de `requests` nem `boto3`. Quem grava é o job `infraestrutura/jobs/download.py`.
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

BASE = "https://diariooficial.prefeitura.sp.gov.br"
CAMINHO = "/md_epubli_controlador.php"
ACAO = "edicao_download"

# Abaixo disto o corpo não é uma edição. A fonte responde 500 com corpo vazio (ou uma página
# de erro curta) quando a data não tem edição, então o tamanho é o que separa os dois casos —
# não o código HTTP.
TAMANHO_MINIMO = 2_048

# O endpoint recusa cliente sem `User-Agent` de navegador; `python-urllib/3.x` recebe 403.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

OK = "ok"
SEM_EDICAO = "sem_edicao"
FALHA = "falha"


@dataclass(frozen=True)
class Resposta:
    """
    O que a fonte devolveu para uma data.

    `situacao` é um dos três valores acima; distinguir `SEM_EDICAO` de `FALHA` evita que o
    pipeline alarme em todo sábado ou engula uma indisponibilidade real — as duas devolvem
    zero bytes.
    """

    data: str
    situacao: str
    conteudo: bytes = b""
    detalhe: str = ""

    @property
    def tem_edicao(self) -> bool:
        return self.situacao == OK


def nome_do_arquivo(data: str, numero_edicao: str = "") -> str:
    """
    O nome sob o qual a edição é gravada: `dom_AAAA-MM-DD.json`.

    O prefixo `dom_` e a data no nome não são estética: `fonte.edicao.arquivos` filtra por
    `dom_*.json` e `fonte.edicao.data_do_arquivo` extrai a data daí.
    """
    if numero_edicao:
        return f"dom_{data}_{numero_edicao}.json"
    return f"dom_{data}.json"


def _url() -> str:
    return f"{BASE}{CAMINHO}?acao={ACAO}"


def _campos(data: str) -> bytes:
    """
    O corpo do POST. A data vai em `dd/mm/aaaa`, que é o formato do formulário.

    `hdnBolEdicaoGerada=true` é o campo que o site envia ao pedir a edição já publicada.
    """
    ano, mes, dia = data.split("-")
    return urllib.parse.urlencode({
        "hdnDtaEdicao": f"{dia}/{mes}/{ano}",
        "hdnFormato": "json",
        "hdnBolEdicaoGerada": "true",
    }).encode("ascii")


def baixar(data: str, tempo_limite: int = 120) -> Resposta:
    """
    Pede a edição de `data` (`AAAA-MM-DD`) e classifica o que voltou.

    Não retenta: a decisão de retentar é de quem chama.

    O corpo é devolvido em bytes, sem decodificar — a correção de codificação é de
    `fonte.edicao.decodificar` e `fonte.edicao.carregar_bytes`, que distinguem os regimes de codificação da fonte.
    """
    requisicao = urllib.request.Request(
        _url(),
        data=_campos(data),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": f"{BASE}/",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=tempo_limite) as resposta:
            corpo = resposta.read()
    except urllib.error.HTTPError as erro:
        # A fonte sinaliza "não há edição nesta data" com HTTP 500 e corpo pequeno (fim de
        # semana, feriado). Só esse caso é SEM_EDICAO. Outros códigos — 403 (bloqueio), 429
        # (throttling), 503 (indisponível) — são FALHA, mesmo com corpo pequeno, para que o
        # chamador os trate como falha em vez de registrar silenciosamente "sem edição" e seguir.
        corpo = erro.read() or b""
        if erro.code == 500 and len(corpo) < TAMANHO_MINIMO:
            return Resposta(data, SEM_EDICAO, detalhe=f"HTTP 500, {len(corpo)} bytes")
        return Resposta(data, FALHA, detalhe=f"HTTP {erro.code}, {len(corpo)} bytes")
    except (urllib.error.URLError, TimeoutError) as erro:
        return Resposta(data, FALHA, detalhe=f"{type(erro).__name__}: {erro}")

    if len(corpo) < TAMANHO_MINIMO:
        return Resposta(data, SEM_EDICAO, detalhe=f"{len(corpo)} bytes")
    return Resposta(data, OK, conteudo=corpo, detalhe=f"{len(corpo)} bytes")
