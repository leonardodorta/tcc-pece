"""formulario.py — camadas 1 a 3 de extração: lê o campo estruturado do formulário, o rótulo na prosa e o documento fiscal como âncora, antes de recorrer ao modelo (camada 4, em domsp/modelo/)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import texto as T
from . import parte

# O FORMULÁRIO NO HTML

# O par canônico: `<p><strong>rótulo</strong></p><p>valor</p>`.
#
# `<b>` entra como alternativa a `<strong>` porque a fonte usa as duas tags, e
# `[^>]*` nos atributos porque o `<p>` do formulário às vezes carrega `style`. O
# `.*?` do valor é preguiçoso e o `</p>` final não é ancorado no fim da linha: o
# campo de prosa contém `<p>` aninhado (o valor de `sintese` vem embrulhado num
# segundo `<p>`), e um casamento guloso engoliria os campos seguintes.
#
# `(?:<!--[^>]*?-->\s*)*` entre a tag de abertura e a tag de rótulo/valor: a fonte pode gravar `<p><!--style="margin-left: 10px;"--><strong>Rótulo</strong></p>` —
# o gerador comenta o `style` em vez de atribuí-lo. Como cada par do documento tem a
# mesma marca, o comentário precisa ser tolerado aqui; do contrário o documento inteiro
# resultaria em `n_pares = 0`, e a camada 1 e o recurso de rótulo-sozinho-na-linha da
# camada 2 não reconheceriam um layout que é, de resto, idêntico. O comentário é descartado nesta
# regra (não entra em `rotulo`/`valor`); a limpeza de comentário em `html_para_texto`
# continua existindo para o que sobra dentro do valor.
#
# A classe é `[^>]` e não `.`, e por isso NÃO leva `re.DOTALL`: o comentário real desta
# fonte é sempre um atributo de estilo numa linha só (`<!--style="..."-->`), nunca
# comentário multilinha. Com `.` e DOTALL, um `.*?` que não achasse o `-->` mais próximo
# atravessaria para o `-->` de um comentário bem mais adiante no documento, varrendo
# campos inteiros por engano. `[^>]` impede a travessia porque o comentário real nunca
# contém `>` no meio.
_RE_COMENTARIO_OPCIONAL = r"(?:\s*<!--[^>]*?-->)*\s*"
_RE_PAR = re.compile(
    r"<p[^>]*>" + _RE_COMENTARIO_OPCIONAL + r"<(?:strong|b)>(?P<rotulo>[^<]*?)</(?:strong|b)>"
    + _RE_COMENTARIO_OPCIONAL + r"</p>\s*"
    r"<p[^>]*>" + _RE_COMENTARIO_OPCIONAL + r"(?P<valor>.*?)</p>",
    re.DOTALL | re.IGNORECASE,
)

# O mesmo gerador intercala, por vezes, um `<p>` vazio (às vezes só com o comentário de
# estilo, sem conteúdo algum) entre o rótulo e o `<p>` que de fato traz o valor — visto
# em "Nome do Contratado (entidade parceira)" seguido de um `<p></p>` espaçador antes do
# nome. Removê-lo ANTES de `_RE_PAR` rodar (e não como alternativa opcional dentro de
# `_RE_PAR`) evita a ambiguidade de backtracking que uma alternativa opcional
# introduziria no meio de dois grupos não-greedy: com o espaçador fora da gramática do
# par, `rotulo` e `valor` continuam podendo ser preguiçosos sem risco de atravessar
# campos vizinhos. Pela mesma razão do comentário acima, sem `re.DOTALL`.
_RE_PARAGRAFO_VAZIO = re.compile(
    r"<p[^>]*>" + _RE_COMENTARIO_OPCIONAL + r"</p>", re.IGNORECASE
)

# O cabeçalho de seção do formulário (`<h5>Principal</h5>`). Não é par rótulo/valor e
# não deve entrar como campo; é removido antes para não virar ruído.
_RE_SECAO = re.compile(r"<h[1-6][^>]*>.*?</h[1-6]>", re.DOTALL | re.IGNORECASE)

# Mínimo de pares para considerar que há formulário. Três, e não um, porque uma
# publicação em prosa pode ter um `<strong>` seguido de `<p>` por acidente de
# formatação, como numa intimação em negrito. Três pares consecutivos no
# padrão do gerador é assinatura de sistema, não de redação.
MINIMO_PARES = 3


# OS RÓTULOS: DE GRAFIA DA FONTE PARA CAMPO CANÔNICO
# A chave da esquerda é o rótulo **normalizado** (`texto.normalizar_chave`:
# minúscula, sem acento, sem pontuação de borda), e é por isso que "Órgão" e "Orgao"
# não precisam de duas entradas. As grafias que precisam de entrada própria são as
# que diferem em palavra, não em acento — cada uma abaixo é uma grafia usada pela fonte.

CAMPO_PROSA = "prosa"
CAMPO_CONTRATADA = "contratada"
CAMPO_CONTRATANTE = "contratante"
CAMPO_DOCUMENTO_FISCAL = "documento_fiscal"
CAMPO_TIPO_PESSOA = "tipo_pessoa"
CAMPO_NUMERO_AJUSTE = "numero_ajuste"
CAMPO_PROCESSO = "processo"
CAMPO_OBJETO = "objeto"
CAMPO_MODALIDADE = "modalidade"
CAMPO_ORGAO = "orgao"
CAMPO_FUNDAMENTO = "fundamento_legal"
CAMPO_JUSTIFICATIVA = "justificativa"
CAMPO_ASSINATURA = "data_assinatura"
CAMPO_INICIO = "vigencia_inicio"
CAMPO_FIM = "vigencia_fim"
CAMPO_PRAZO = "prazo"
CAMPO_PRAZO_UNIDADE = "prazo_unidade"
CAMPO_PUBLICACAO = "data_publicacao"
CAMPO_NATUREZA_DECLARADA = "natureza_declarada"
CAMPO_LOCAL = "local_execucao"

ROTULOS: dict[str, str] = {
    # ── o campo de prosa: três grafias ──
    "sintese (texto do despacho)": CAMPO_PROSA,
    "texto do despacho": CAMPO_PROSA,
    "conteudo do despacho": CAMPO_PROSA,
    # ── a parte contratada ──
    "contratado(a)": CAMPO_CONTRATADA,
    "nome do contratado (entidade parceira)": CAMPO_CONTRATADA,
    "contratada": CAMPO_CONTRATADA,
    "detentora": CAMPO_CONTRATADA,
    "nome do contratado": CAMPO_CONTRATADA,
    # ── o lado público: só existe no layout de parceria ──
    "nome do contratante": CAMPO_CONTRATANTE,
    "contratante": CAMPO_CONTRATANTE,
    # ── documento fiscal ──
    "cpf /cnpj/ rne": CAMPO_DOCUMENTO_FISCAL,
    "cpf/cnpj/rne": CAMPO_DOCUMENTO_FISCAL,
    "cnpj do contratado (entidade parceira)": CAMPO_DOCUMENTO_FISCAL,
    "cnpj": CAMPO_DOCUMENTO_FISCAL,
    "cnpj/cpf": CAMPO_DOCUMENTO_FISCAL,
    "tipo de pessoa": CAMPO_TIPO_PESSOA,
    # ── identificação do ajuste ──
    "numero do contrato": CAMPO_NUMERO_AJUSTE,
    "numero da ata": CAMPO_NUMERO_AJUSTE,
    "numero do termo aditivo": CAMPO_NUMERO_AJUSTE,
    "numero": CAMPO_NUMERO_AJUSTE,
    "processo": CAMPO_PROCESSO,
    "numero de processo interno do orgao/unidade": CAMPO_PROCESSO,
    # ── objeto ──
    "objeto do contrato": CAMPO_OBJETO,
    "objeto da licitacao": CAMPO_OBJETO,
    "objeto do aditamento": CAMPO_OBJETO,
    "descricao detalhada do objeto": CAMPO_OBJETO,
    "objeto": CAMPO_OBJETO,
    # ── enquadramento ──
    "modalidade": CAMPO_MODALIDADE,
    "orgao": CAMPO_ORGAO,
    "fundamento legal": CAMPO_FUNDAMENTO,
    "justificativa": CAMPO_JUSTIFICATIVA,
    "natureza": CAMPO_NATUREZA_DECLARADA,
    "descricao da natureza": CAMPO_NATUREZA_DECLARADA,
    "local de execucao": CAMPO_LOCAL,
    # ── tempo ──
    "data da assinatura": CAMPO_ASSINATURA,
    "data da assinatura do termo aditivo": CAMPO_ASSINATURA,
    "data de inicio": CAMPO_INICIO,
    "data de fim": CAMPO_FIM,
    "prazo do contrato": CAMPO_PRAZO,
    "tipo do prazo": CAMPO_PRAZO_UNIDADE,
    # `data de publicacao` e `data da publicacao` são a mesma coisa com duas
    # preposições. Não são sinônimo acidental: vêm de layouts diferentes. Registradas as duas porque normalizar acento não resolve troca de
    # preposição.
    "data de publicacao": CAMPO_PUBLICACAO,
    "data da publicacao": CAMPO_PUBLICACAO,
}

# Os campos que este módulo não mapeia, de propósito:
#
#   `anexo i/ii (numero do documento sei)`, `arquivo (...)`, `integra do contrato
#   (...)` — são identificadores de documento no SEI, não dado do ato.
#   `especificacao de outras`, `cotacao eletronica`, `registro de preco`, `tipo` —
#   existem na fonte, mas o vocabulário de cada um não está catalogado, e mapear campo cujo
#   domínio de valores não se conhece produz coluna que ninguém sabe interpretar.
#
# Registrar a recusa é o ponto: sem esta lista, "não mapeamos" e "esquecemos" ficam
# indistinguíveis para quem lê o módulo depois.
NAO_MAPEADOS: frozenset[str] = frozenset({
    "anexo i (numero do documento sei)", "anexo ii (numero do documento sei)",
    "arquivo (numero do documento sei)", "integra do contrato (numero do documento sei)",
    "especificacao de outras", "cotacao eletronica", "registro de preco", "tipo",
})


# OS LAYOUTS


@dataclass(frozen=True)
class Layout:
    """
    Um layout de formulário: como reconhecê-lo, e o que ele garante.

    `assinatura` são os campos canônicos cuja presença simultânea identifica o
    layout. Reconhecer por assinatura, e não por série editorial, é deliberado: a
    série é metadado que o publicador escolhe e que varia, enquanto o conjunto de
    campos é produzido pelo sistema que gera o HTML — o layout é propriedade do
    gerador.

    `tem_prosa` é a informação operacionalmente mais importante: onde é falso, a
    camada 2 não tem onde procurar, e a ausência de rótulo na prosa não é sinal de
    que o dado falta — é sinal de que ele está em campo próprio.
    """

    nome: str
    assinatura: frozenset[str]
    tem_prosa: bool
    nota: str


LAYOUTS: tuple[Layout, ...] = (
    Layout(
        "contrato", frozenset({CAMPO_CONTRATADA, CAMPO_DOCUMENTO_FISCAL, CAMPO_PRAZO,
                               CAMPO_PROSA}),
        tem_prosa=True,
        nota="o layout mais completo com prosa: parte, documento, prazo e o extrato "
             "integral no campo de síntese. Não traz o lado público",
    ),
    Layout(
        "parceria", frozenset({CAMPO_CONTRATANTE, CAMPO_CONTRATADA, CAMPO_INICIO,
                               CAMPO_FIM}),
        tem_prosa=False,
        nota="totalmente estruturado, sem prosa alguma. É o único que declara o lado "
             "público, e o único em que vigência vem em dois campos próprios",
    ),
    Layout(
        "licitacao", frozenset({CAMPO_OBJETO, CAMPO_NATUREZA_DECLARADA, CAMPO_LOCAL,
                               CAMPO_PROSA}),
        tem_prosa=True,
        nota="objeto, natureza e local de execução com prosa em 'texto do despacho'. "
             "Em geral é de licitação, que a triagem descarta, mas o layout "
             "também pode aparecer em ata de registro de preço, que não é descartada",
    ),
    Layout(
        "despacho", frozenset({CAMPO_PROSA, CAMPO_PUBLICACAO}),
        tem_prosa=True,
        nota="mínimo: prosa e data. Todo o dado do ato está na prosa, "
             "então a camada 2 é a que lê este layout",
    ),
)


def layout_de(campos: dict[str, str]) -> Layout | None:
    """
    O layout mais específico cuja assinatura os campos satisfazem.

    "Mais específico" = maior assinatura, e a ordem importa: `despacho` exige apenas
    prosa e data, então ele casaria com quase tudo. Testando os maiores primeiro,
    `contrato` e `parceria` ganham de `despacho` quando ambos casam, o que é o
    desejado — `despacho` é o caso residual, não o caso geral.
    """
    presentes = {k for k, v in campos.items() if v}
    candidatos = [lay for lay in LAYOUTS if lay.assinatura <= presentes]
    if not candidatos:
        return None
    return max(candidatos, key=lambda lay: len(lay.assinatura))


# CAMADA 1 — LER O FORMULÁRIO


@dataclass(frozen=True)
class Formulario:
    """O que a camada 1 leu, e o suficiente para saber o quanto confiar nisso."""

    campos: dict[str, str] = field(default_factory=dict)
    layout: Layout | None = None
    n_pares: int = 0
    rotulos_ignorados: tuple[str, ...] = ()

    @property
    def existe(self) -> bool:
        return self.n_pares >= MINIMO_PARES

    @property
    def tem_prosa(self) -> bool:
        return bool(self.campos.get(CAMPO_PROSA))


def ler(html: str | None) -> Formulario:
    """
    Os pares rótulo/valor do HTML, mapeados para campo canônico.

    Quatro decisões nesta função:

    - **Primeiro valor ganha, repetição é ignorada.** Um layout pode repetir o
      mesmo rótulo; o primeiro é o principal, o último sobrescreveria com o
      acessório.
    - **Rótulo desconhecido é registrado, não descartado.** `rotulos_ignorados`
      guarda o que apareceu e não está em `ROTULOS` nem em `NAO_MAPEADOS`, para
      detectar se a fonte criou campo novo.
    - **O valor passa por `html_para_texto`, não por remoção de tag.** O campo de
      prosa contém HTML aninhado com fronteira de bloco, e é essa fronteira que
      impede "...CAMPO LIMPO</p><p>EXTRATO..." de virar "LIMPOEXTRATO".
    - **O `<p>` vazio espaçador é removido antes de `_RE_PAR` rodar, não casado por
      ela.** Ele pode intercalar rótulo e valor (ver `_RE_PARAGRAFO_VAZIO`);
      removê-lo primeiro evita que uma alternativa opcional no meio da regra force
      backtracking nos grupos preguiçosos de rótulo e valor, o que atravessaria `<p>`
      vizinhos e desalinharia campos inteiros.
    """
    if not html:
        return Formulario()

    limpo = _RE_SECAO.sub(" ", html)
    limpo = _RE_PARAGRAFO_VAZIO.sub("", limpo)
    pares = _RE_PAR.findall(limpo)
    if len(pares) < MINIMO_PARES:
        return Formulario(n_pares=len(pares))

    campos: dict[str, str] = {}
    ignorados: list[str] = []
    for rotulo_bruto, valor_bruto in pares:
        chave = T.normalizar_chave(T.html_para_texto(rotulo_bruto))
        if not chave:
            continue
        canonico = ROTULOS.get(chave)
        if canonico is None:
            if chave not in NAO_MAPEADOS:
                ignorados.append(chave)
            continue
        valor = T.html_para_texto(valor_bruto).strip()
        if valor and canonico not in campos:
            campos[canonico] = valor

    return Formulario(
        campos=campos, layout=layout_de(campos), n_pares=len(pares),
        rotulos_ignorados=tuple(dict.fromkeys(ignorados)),
    )


# O CORTE DA FONTE NO CAMPO DO FORMULÁRIO

# O Painel de Negócios corta o valor do campo em 100 caracteres. Alguns atos têm o campo de
# contratada com exatamente 100 caracteres terminando no meio de uma palavra, como em
# contratos de consórcio, cujos nomes listam várias empresas e por isso estouram o limite.
#
# O campo é usado em DOIS lugares: é a camada 1 do pipeline, e é o gabarito da avaliação.
# Como gabarito truncado, torna a resposta correta inalcançável — o modelo lê a prosa,
# acerta o nome inteiro, e é contado como errado por não reproduzir o corte no meio da
# palavra. Como camada 1, grava nome incompleto na base entregue.
CORTE_DA_FONTE = 100

# Rótulos de campo do formulário. Se a extensão alcançar um deles, ela passou do fim do nome
# e entrou no campo seguinte — a completação é descartada em vez de gravar lixo.
_RE_ROTULO_DE_CAMPO = re.compile(
    r"(?i)\b(valor|objeto|prazo|vig[êe]ncia|processo|contrato|cnpj|data|assinatura|"
    r"dota[çc][ãa]o|empenho|modalidade|tipo de pessoa)\b"
)


def completar_corte_da_fonte(nome: str, texto_completo: str) -> tuple[str, bool]:
    """
    Completa, pela prosa, o nome que a FONTE cortou no limite do campo.

    Devolve `(nome, foi_completado)`. Sem completação possível, devolve o nome intacto — a
    função nunca degrada o valor recebido.

    A identidade da entidade continua vindo do formulário: são 100 caracteres de evidência
    independente da prosa, e o casamento do prefixo é exato, não por similaridade. Só a
    cauda do token cortado vem da prosa, pela mesma `parte.porta_de_nome` que delimita nome
    em todas as camadas.

    Duas guardas protegem a completação de dois modos de erro: uma extensão que engole o
    rótulo do campo seguinte (`Objeto`, `Valor`), e uma extensão que para num segundo corte,
    devolvendo nome ainda incompleto:

    1. se a cauda acrescentada contém rótulo de campo, a extensão passou do nome — descarta;
    2. se o nome completado termina colado a um caractere alfanumérico, ele próprio está
       cortado — descarta.

    Quando a completação é recusada, o nome permanece truncado em vez de virar invenção.
    """
    if len(nome) != CORTE_DA_FONTE:
        return nome, False
    inicio = texto_completo.find(nome)
    if inicio < 0:
        # O valor do campo não ocorre na prosa: não há o que completar, e tampouco há
        # evidência para fazê-lo.
        return nome, False
    if len(texto_completo) <= inicio + len(nome):
        return nome, False

    janela = texto_completo[inicio:inicio + CORTE_DA_FONTE + 500]
    candidato = parte.porta_de_nome(janela).nome
    if not candidato or len(candidato) <= len(nome):
        return nome, False
    if not candidato.startswith(nome[:60]):
        # A porta devolveu outra coisa, não uma extensão deste nome.
        return nome, False

    cauda = candidato[len(nome):]
    if _RE_ROTULO_DE_CAMPO.search(cauda):          # guarda 1
        return nome, False

    fim = texto_completo.find(candidato)
    if fim >= 0 and len(texto_completo) > fim + len(candidato):
        if texto_completo[fim + len(candidato)].isalnum():   # guarda 2
            return nome, False

    return candidato, True


def contratada_do_formulario(
    form: Formulario, texto_completo: str | None = None
) -> parte.ResultadoNome:
    """
    A parte contratada declarada no formulário, pela **mesma** porta das outras camadas.

    O ponto desta função é não ter regra própria: ela delega a
    `parte.porta_de_nome`, que é a porta única. O campo do formulário também traz
    papel contratual sozinho ("DONATÁRIA"), nome com cauda de identificador, e
    conectivo pendurado, do mesmo jeito que a prosa — a diferença entre as camadas
    é a precisão, não a dispensa de validação.

    O lado público conhecido é passado adiante porque é o que impede o polo
    contratante de ser aceito como contratada quando os dois nomes estão no mesmo
    campo.
    """
    bruto = form.campos.get(CAMPO_CONTRATADA)
    if bruto and texto_completo:
        bruto, _ = completar_corte_da_fonte(bruto, texto_completo)
    return parte.porta_de_nome(
        bruto,
        lado_publico_conhecido=form.campos.get(CAMPO_CONTRATANTE),
    )


# O ALVO DA ANÁLISE × O TEXTO PARA AUDITORIA


def alvo_da_analise(form: Formulario, texto_completo: str) -> str:
    """
    **Onde procurar** rótulo, valor e documento.

    Quando o formulário tem campo de prosa, o alvo é esse campo — porque ele contém o
    extrato integral do ato, e o HTML completo contém, além dele, os rótulos do
    próprio formulário e os valores dos outros campos. Procurar "R$" no HTML completo
    de um layout `contrato` acha o valor do extrato **e** qualquer número que outro
    campo declare, e a eleição do valor passa a competir com dado que não é do ajuste.

    Sem formulário, o alvo é o texto completo, que é o caso das publicações em prosa.
    """
    prosa = form.campos.get(CAMPO_PROSA)
    return prosa if prosa else (texto_completo or "")


def texto_para_auditoria(texto_completo: str) -> str:
    """
    **O que gravar** como texto de auditoria.

    Sempre o texto completo, mesmo quando o alvo da análise foi um recorte. Dois
    consumidores exigem isso: o modelo, que precisa do contexto inteiro para
    delimitar span, e a auditoria humana, que precisa poder conferir a extração
    contra a publicação como ela foi publicada.
    """
    return texto_completo or ""


# CAMADA 2 — O RÓTULO NA PROSA

# Os rótulos que a prosa usa para nomear a parte contratada.
#
# A ordem é a de especificidade. `DETENTORA` está aqui porque é a forma das atas de
# registro de preço, que não usam "contratada" — e sem ela uma classe inteira de ato
# perde a camada 2 e cai no resíduo, aumentando o custo do modelo sem necessidade.
#
# `COLABORADORA` é a forma do termo de colaboração (Lei 13.019/2014) — o mesmo motivo:
# sem ela, a camada 3 (âncora por documento) pode ancorar no primeiro CNPJ do texto,
# que pode pertencer a uma filial mencionada no OBJETO do ato, e não à colaboradora.
_ROTULOS_PROSA: tuple[tuple[str, str], ...] = (
    (CAMPO_CONTRATADA, r"CONTRATAD[AO]\s*(?:\(A\))?"),
    (CAMPO_CONTRATADA, r"DETENTORA"),
    (CAMPO_CONTRATADA, r"PERMISSION[AÁ]RI[AO]"),
    (CAMPO_CONTRATADA, r"DONAT[AÁ]RI[AO]"),
    (CAMPO_CONTRATADA, r"COLABORADORA"),
    (CAMPO_CONTRATANTE, r"CONTRATANTE"),
)

# O limite do valor rotulado
#
# O trecho rotulado tem de terminar em algum lugar: curto demais corta a razão
# social ("ALFA - ESPORTES COMÉRCIO" tem hífen no meio do nome), longo demais
# entrega uma frase à porta de limpeza.
#
# Terminar só em rótulo CAIXA ALTA longo ou fim de linha não basta: a fonte escreve
# em caixa mista (Title Case, não só maiúsculas), e o separador de campo é ` - `,
# não fim de linha, com o rótulo seguinte podendo ter poucos caracteres (`Objeto:`,
# `CNPJ:`).
#
# O limite termina no primeiro de cinco sinais, cada um usado pela fonte:
# (a) ` - ` seguido de palavra capitalizada com dois-pontos, que é o separador do
# gerador; (b) `CNPJ`/`CPF` com ou sem dois-pontos, campo que costuma seguir o nome;
# (c) rótulo do vocabulário fechado, em caixa alta ou Title-Case, com dois-pontos;
# (d) o mesmo rótulo colado a uma palavra em caixa alta; (e) fim de linha.
#
# O que não se faz é cortar em qualquer ` - `: isso partiria "ALFA - ESPORTES
# COMÉRCIO E SERVIÇOS LTDA" no primeiro hífen. Exigir que o hífen seja seguido de
# rótulo com dois-pontos é o que distingue separador de campo de hífen de razão
# social — e é a razão de este limite ser uma alternação e não um caractere.
#
# O vocabulário de rótulos de campo em prosa
#
# O extrato em prosa separa campos por rótulo em caixa alta seguido de dois-pontos:
# `OBJETO:`, `VALOR TOTAL:`, `DATA DA ASSINATURA:`. Para saber onde o valor de um campo
# termina, é preciso saber onde o PRÓXIMO rótulo começa — e isso exige reconhecer o rótulo,
# não apenas a caixa alta.
#
# A razão de ser um vocabulário fechado, e não um padrão de forma: razão social também é
# escrita em caixa alta. Um padrão que aceite qualquer corrida de maiúsculas seguida de
# dois-pontos casa o fim do nome mais o rótulo seguinte, e o valor é truncado no terceiro
# caractere — `ALFA ENGENHARIA ELETRICA LTDA OBJETO:` devolveria `ALF`, porque
# ` ENGENHARIA ELETRICA LTDA OBJETO:` satisfaz a forma. Ancorar na primeira palavra de um
# rótulo conhecido resolve isso de maneira que não depende da caixa do nome.
#
# As entradas vêm de duas origens: as primeiras palavras dos rótulos que `ROTULOS` declara
# para o formulário, e as usadas no extrato em prosa.
_PALAVRAS_DE_ROTULO = (
    "OBJETO", "VALOR", "DATA", "DATAS", "PRAZO", "VIG[ÊE]NCIA", "PROCESSO", "PROTOCOLO",
    "CONTRATANTE", "CONTRATADA", "CONTRATADO", "DETENTORA", "DOADOR", "DONAT[ÁA]RI[OA]",
    "SIGNAT[ÁA]RI[OA]S?", "PARTES", "MODALIDADE", "MODELO", "MARCA", "MEDIDAS",
    "DOTA[ÇC][ÃA]O", "OR[ÇC]AMENT[ÁA]RIA", "NOTA", "EMPENHO", "FUNDAMENTO", "NATUREZA",
    "DESCRI[ÇC][ÃA]O", "JUSTIFICATIVA", "LOCAL", "ENDERE[ÇC]O", "NOME", "N[ÚU]MERO",
    "[ÓO]RG[ÃA]O", "S[ÍI]NTESE", "TEXTO", "ANEXO", "ARQUIVO", "TIPO", "SEI", "SIMPROC",
    "ATENDIMENTO", "CONSUMO", "FAIXA", "PER[ÍI]ODO", "TERMO", "ADITAMENTO", "REGISTRO",
    "ESPECIFICA[ÇC][ÃA]O", "COTA[ÇC][ÃA]O", "INTEGRA", "CONTE[ÚU]DO", "QUANTIDADE",
    "UNIDADE", "LOTE", "ITEM", "ITENS", "REAJUSTE", "GARANTIA", "FISCAL", "GESTOR",
)

#: Onde um rótulo pode começar. Ou há fronteira de palavra, ou há a transição de minúscula
#: para maiúscula que denuncia rótulo colado à palavra anterior — `…São PauloVALOR:`. O
#: rótulo também cola a uma palavra em CAIXA ALTA (`…-SMTCONTRATADA:`), transição que `\b`
#: não vê — por isso o lookbehind permanece sensível só a "não ser início de palavra", e a
#: distinção entre um prefixo colado legítimo (sigla de órgão) e um prefixo que MODIFICA o
#: papel (`SUBCONTRATADA`, `RECONTRATADA`) é feita por `_RE_PREFIXO_QUE_NEGA_PAPEL`, abaixo.
_ABRE_ROTULO = r"(?:\b|(?<=[a-zà-ú0-9]))"

#: Prefixos que, colados ao papel, mudam o papel e portanto NÃO devem casar como o rótulo
#: base. `SUBCONTRATADA`/`RECONTRATADA`/`DESCONTRATADA` não são a contratada;
#: `CODETENTORA` não é a detentora. São prefixos de formação de palavra (sub-, re-, des-,
#: co-), ao contrário de uma sigla de órgão colada (`SMTCONTRATADA`), que é concatenação de
#: duas palavras.
_RE_PREFIXO_QUE_NEGA_PAPEL = r"(?<!SUB)(?<!RE)(?<!DES)(?<!CO)(?<!PRE)"

#: O caso `…LTDAOBJETO:`, em que o rótulo cola numa palavra que também é caixa alta e não
#: deixa transição de caixa para detectar. A fronteira de palavra é abandonada de
#: propósito, e o preço é cobrado numa lista explícita: só estas palavras abrem rótulo
#: colado, e o rótulo tem de terminar em dois-pontos.
#:
#: A lista é a das palavras de rótulo que aparecem coladas ao valor, com uma exclusão
#: deliberada: `DATA`, `ITEM`, `NOTA`, `NOME`, `LOTE`, `TEXTO` e `UNIDADE` ficam fora
#: porque são subcadeia de palavra comum — `CANDIDATA`, `SUBITEM`, `ANOTA`, `RENOME`,
#: `PILOTE`, `CONTEXTO`, `COMUNIDADE` —, e aceitá-las cortaria nome pelo meio para ganhar
#: uma terminação (`COMUNIDADE:` viraria `COM`, porque `UNIDADE:` casaria a subcadeia).
_COLADAS = (
    "CONTRATADA", "CONTRATANTE", "CONTRATADO", "DETENTORA", "PROCESSO", "OBJETO",
    "MODALIDADE", "VALOR", "PRAZO", "VIG[ÊE]NCIA", "ATENDIMENTO", "FISCAL",
    "LOCAL", "TERMO", "EMPENHO", "DOTA[ÇC][ÃA]O", "SIGNAT[ÁA]RI[OA]S?", "MARCA", "MODELO",
)

#: O que pode vir entre a primeira palavra do rótulo e os dois-pontos: até cinco palavras
#: de até 20 letras, separadas por espaço, tabulação ou barra. O limite de número e de
#: tamanho evita que a alternativa varra prosa até achar um dois-pontos distante.
_RESTO_DO_ROTULO = r"(?:[ \t/]+(?:[A-ZÀ-Úa-zà-ú][A-ZÀ-Úa-zà-ú]{0,19})){0,5}"


def _title_case_de_rotulo(padrao: str) -> str:
    """
    A variante Title-Case de um padrão de rótulo, preservando as classes de caractere.

    `_PALAVRAS_DE_ROTULO` contém PADRÕES, não literais: `[ÓO]RG[ÃA]O`, `SIGNAT[ÁA]RI[OA]S?`.
    `str.capitalize()` sobre eles produz lixo — `"[ÓO]RG[ÃA]O".capitalize()` vira
    `"[óo]rg[ãa]o"`, minusculizando o interior das classes e quebrando o padrão.

    Title-Case significa: inicial maiúscula seguida de minúsculas. Mantém-se o primeiro
    átomo do padrão como está (já cobre a maiúscula inicial: `[ÓO]…` ou `O…`) e torna-se o
    resto insensível a caixa com `(?i:…)`, de modo que `RGÃO` passe a casar `rgão`. O
    primeiro átomo fora do `(?i:)` preserva a exigência de inicial maiúscula — é isso que
    distingue um rótulo (`Órgão:`) de uma palavra comum em minúscula no meio do valor.
    """
    corpo, resto = _primeiro_atomo(padrao)
    return corpo + (rf"(?i:{resto})" if resto else "")


def _primeiro_atomo(padrao: str) -> tuple[str, str]:
    """Separa o primeiro átomo de um padrão (uma classe `[...]` ou um caractere) do resto."""
    if padrao.startswith("["):
        fim = padrao.index("]") + 1
        return padrao[:fim], padrao[fim:]
    return padrao[:1], padrao[1:]

_LIMITE = (
    r"(?=\s*(?:"
    r"-\s+[A-ZÀ-Ú][A-Za-zÀ-ú]{2,}\s*:"              # a) ` - Assunto:`  ` - Objeto:`
    r"|-?\s*CNPJ|-?\s*CPF|-?\s*C\.N\.P\.J"           # b) ` - CNPJ Nº`
    # c) o próximo rótulo de campo: primeira palavra do vocabulário fechado, as palavras de
    #    rótulo que a seguem, e dois-pontos. Aceita o rótulo em CAIXA ALTA e em Title-Case,
    #    porque a fonte usa as duas grafias — mas só as palavras do vocabulário abrem
    #    rótulo, e é isso que impede o fim de uma razão social de passar por um.
    r"|(?-i:" + _ABRE_ROTULO + r"(?:"
    + "|".join(_PALAVRAS_DE_ROTULO) + r"|"
    + "|".join(_title_case_de_rotulo(p) for p in _PALAVRAS_DE_ROTULO)
    + r")\b" + _RESTO_DO_ROTULO + r"[ \t]*:)"
    # d) o mesmo rótulo, colado a uma palavra em caixa alta: `…LTDAOBJETO:`
    r"|(?-i:(?:" + "|".join(_COLADAS) + r")" + _RESTO_DO_ROTULO + r"[ \t]*:)"
    r"|\n|$"                                          # e) fim de linha
    r"))"
)
# A alternativa (c) tem o IGNORECASE desligado no escopo porque `_padrao` compila com
# `re.IGNORECASE` (a fonte escreve em caixa mista), e o efeito colateral seria a
# alternativa (c) — escrita para casar rótulo em CAIXA ALTA seguido de dois-pontos — casar
# qualquer sequência de letras seguida de dois-pontos, truncando o valor no primeiro ponto
# em que o resto até os dois-pontos seja só letras e espaço. `(?-i:…)` restaura a
# exigência de caixa alta só nesse grupo, sem desfazer a correção de caixa mista nos
# rótulos.


def _padrao(rotulo: str) -> re.Pattern[str]:
    """
    O padrão de um rótulo na prosa.

    `IGNORECASE` existe porque a fonte escreve em caixa mista, e tem um efeito colateral
    aceito: o padrão passa a casar `contratada:` em minúscula no meio de uma frase corrida.
    É um risco real, e a defesa não é aqui — é a porta de limpeza, que recusa o que não tem
    forma de nome. Preferir o falso positivo tratável ao falso negativo silencioso é a
    escolha certa quando existe uma porta única a jusante.
    """
    return re.compile(rf"{_ABRE_ROTULO}(?-i:{_RE_PREFIXO_QUE_NEGA_PAPEL}){rotulo}\s*[:\-–]\s*(?P<valor>.{{3,300}}?){_LIMITE}",
                      re.DOTALL | re.IGNORECASE)


_COMPILADOS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (campo, _padrao(rot)) for campo, rot in _ROTULOS_PROSA
)

# As duas construções sem dois-pontos
#
# A ata de registro de preço não rotula: escreve "tem como detentora a empresa X,
# inscrita no CNPJ". O apostilamento escreve "em favor da empresa contratada X CNPJ
# nº". Nenhuma das duas tem dois-pontos, então nenhuma é alcançada pelos padrões
# rotulados acima.
#
# O limite das duas é o mesmo: o nome termina onde começa o documento fiscal
# (`inscrita`, `CNPJ`, `portadora`).
_RE_DETENTORA_EM_PROSA = re.compile(
    r"\btem\s+como\s+detentora\s+(?:a\s+)?(?:empresa\s+)?(?P<valor>.{5,200}?)"
    r"\s*,?\s*(?:inscrita|CNPJ|C\.N\.P\.J|portadora)",
    re.IGNORECASE | re.DOTALL,
)

_RE_EMPRESA_CONTRATADA = re.compile(
    r"\b(?:em\s+favor\s+d[ae]\s+|com\s+[ao]\s+|firmado\s+com\s+[ao]\s+)?"
    r"empresa\s+contratad[ao]\s+(?P<valor>.{5,200}?)"
    r"\s*,?\s*(?:inscrita|CNPJ|C\.N\.P\.J|portadora)",
    re.IGNORECASE | re.DOTALL,
)

# A ordem é a de especificidade decrescente, e é a ordem de tentativa: a construção
# com "detentora" é mais específica que a com "empresa contratada", que por sua vez
# exige menos contexto.
_SEM_ROTULO_EXPLICITO: tuple[re.Pattern[str], ...] = (
    _RE_DETENTORA_EM_PROSA, _RE_EMPRESA_CONTRATADA,
)


# O rótulo sozinho na linha: o formulário renderizado em texto puro, um campo por
# linha (`Contratado(a)\nBETA INDÚSTRIA FARMACÊUTICA S/A\n...`). Os padrões
# rotulados acima exigem `[:\-–]` entre rótulo e valor, e aqui o separador é a
# quebra de linha. Exigir que a linha contenha apenas o rótulo distingue campo de
# formulário de menção em frase corrida (ex.: "obrigações da CONTRATADA" seguido de
# outro parágrafo não deve casar).
#
# É consultado apenas como recurso subsidiário, quando nenhum rótulo com dois-pontos
# foi achado.
_RE_ROTULO_EM_LINHA = re.compile(
    r"(?:^|\n)[ \t]*"
    r"(?:CONTRATAD[AO]\s?\(?A?\)?|DETENTORA|PERMISSION[AÁ]RI[AO]|DONAT[AÁ]RI[AO]|COLABORADORA)"
    r"[ \t]*\n[ \t]*"
    r"(?P<valor>[^\n]{3,200})",
    re.IGNORECASE,
)


def rotulos_na_prosa(alvo: str | None) -> dict[str, str]:
    """
    Os valores que a prosa rotulou, por campo canônico.

    Devolve o bruto: nenhuma limpeza é feita aqui. Encontrar o trecho e decidir se
    o trecho é um nome são decisões diferentes; quem valida é `parte.porta_de_nome`,
    a mesma porta das demais camadas.
    """
    if not alvo:
        return {}
    achados: dict[str, str] = {}
    for campo, padrao in _COMPILADOS:
        if campo in achados:
            continue
        m = padrao.search(alvo)
        if m:
            valor = T.colapsar_espacos(m.group("valor")).strip(" .,;:-–—")
            if valor:
                achados[campo] = valor
    if CAMPO_CONTRATADA not in achados:
        m = _RE_ROTULO_EM_LINHA.search(alvo)
        if m:
            valor = T.colapsar_espacos(m.group("valor")).strip(" .,;:-–—")
            if valor:
                achados[CAMPO_CONTRATADA] = valor
    if CAMPO_CONTRATADA not in achados:
        m = _RE_DETENTORA_EM_PROSA.search(alvo)
        if m:
            valor = T.colapsar_espacos(m.group("valor")).strip(" .,;:-–—")
            if valor:
                achados[CAMPO_CONTRATADA] = valor
    return achados


def contratada_da_prosa(alvo: str | None,
                        lado_publico: str | None = None) -> parte.ResultadoNome:
    """
    A parte contratada rotulada na prosa, pela porta única.

    Devolve também, quando existe, o documento fiscal que segue o nome no texto. A
    fonte escreve `CONTRATADA: RAZÃO SOCIAL, inscrita no CNPJ nº 12.345.678/0001-95`,
    e esse CNPJ é o da contratada por construção — está entre o rótulo que a nomeia
    e o próximo campo. Capturá-lo aqui evita gravar como documento da contratada o
    primeiro CNPJ do texto, que pode ser o do próprio órgão.
    """
    rotulados = rotulos_na_prosa(alvo)
    bruto = rotulados.get(CAMPO_CONTRATADA)
    resultado = parte.porta_de_nome(
        bruto,
        lado_publico_conhecido=lado_publico or rotulados.get(CAMPO_CONTRATANTE),
    )
    if resultado.nome and bruto and alvo:
        documento = _documento_apos_o_nome(bruto, alvo)
        if documento:
            return parte.ResultadoNome(resultado.nome, resultado.ressalvas,
                                       documento=documento)
    return resultado


def _documento_apos_o_nome(nome_bruto: str, alvo: str) -> str | None:
    """
    O documento fiscal que a fonte escreve logo depois do nome rotulado.

    Procura, na janela que começa onde o nome termina, o primeiro CNPJ ou CPF válido. É a
    posição relativa que importa: um documento imediatamente após o nome da contratada é o
    documento dela, enquanto o primeiro documento do texto pode ser o do órgão citado antes.
    """
    pos = alvo.find(nome_bruto)
    if pos < 0:
        return None
    janela = alvo[pos + len(nome_bruto): pos + len(nome_bruto) + _JANELA_DA_ANCORA]
    for achado in _RE_DOCUMENTO_NO_TEXTO.finditer(janela):
        if parte.classificar_documento(achado.group(0)).valido:
            return achado.group(0)
    return None


# CAMADA 3 — O DOCUMENTO FISCAL COMO ÂNCORA
# Existe porque o estrato `sem_rotulo` não tem rótulo para a camada 2 procurar. Mas
# um ato do resíduo pode trazer CNPJ ou CPF com dígito verificador válido no
# texto, e o nome da parte vem antes dele.

_RE_DOCUMENTO_NO_TEXTO = re.compile(
    r"(?<![\d.])(?:"
    r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}"      # CNPJ (14 dígitos)
    r"|\d{3}\.?\d{3}\.?\d{3}-?\d{2}"            # CPF (11 dígitos) — contrato de pessoa física
    r")(?![\d.]*\d)"
)

# A frase que a fonte escreve ENTRE o nome e o documento. Sem removê-la, é ela que a busca
# captura: "Cadastro Nacional de Pessoas Jurídicas sob nº" sairia como se fosse a razão
# social.
_RE_QUALIFICACAO_DO_DOCUMENTO = re.compile(
    r"\s*[,;]?\s*(?:pessoa\s+jur[íi]dica[^,]{0,60})?\s*[,;]?\s*"
    r"(?:inscrit[ao]|registrad[ao]|portador(?:a)?)?\s*(?:no|na|sob|com|de|do)?\s*"
    r"(?:Cadastro\s+Nacional\s+de\s+Pessoas?\s+Jur[íi]dicas?|CNPJ(?:/MF)?|CPF(?:/MF)?|MF)"
    r"\s*(?:/MF)?\s*(?:sob)?\s*(?:o|a)?\s*n?[ºo°.]*\s*:?\s*$",
    re.I,
)

# A fronteira esquerda: onde o nome começa. Ancora em marcadores inequívocos de
# preâmbulo (dois-pontos, ponto-e-vírgula, parêntese de fecho, e as preposições de
# regência que a fonte usa para apresentar a contratada: 'à empresa X', 'com a
# firma X'), que não ocorrem dentro de uma razão social — e não em ' e '/' com '
# isolados, que ligam palavras de razão social ('COMERCIO E SERVICOS'). O quantificador é
# guloso para pegar o marcador mais próximo do documento quando há mais de um.
#
# 'do proponente' entra pela mesma razão: no layout de termo de fomento/parceria, a
# fonte apresenta a entidade parceira como '[nome do projeto], do proponente NOME,
# CNPJ ...' — sem o marcador, a fronteira esquerda não veria onde o nome do projeto
# termina e o nome da entidade começa, e o nome capturado incluiria o projeto inteiro.
_RE_FRONTEIRA_ESQUERDA = re.compile(
    r"^.*(?:"
    r":|;|\)"
    r"|\b[àa]s?\s+(?:empresa|firma|sociedade|institui[çc][ãa]o|entidade|contratada|"
    r"detentora|donat[áa]ri[oa]|senhor[a]?)\b"
    r"|\bcom\s+a\s+(?:empresa|firma|sociedade)\b"
    r"|\bdo\s+proponente\b"
    r")\s+",
    re.S | re.I,
)

# A fronteira esquerda em ` e `/`entre … e `, aplicada só quando o lado esquerdo é o
# polo público. Cortar em ` e ` de forma geral decaparia razão social — por isso
# `_RE_FRONTEIRA_ESQUERDA` nunca o faz. Mas o publicador escreve `PMSP/SMC e NOME`,
# `entre a PREFEITURA … e a ASSOCIAÇÃO X`: ali o ` e ` separa o polo público do
# privado, e o que vem antes dele é descartável. O corte só dispara quando o
# preâmbulo é o polo público — sigla com barra (`PMSP/SMC`) ou uma das palavras de
# Administração logo no início, após no máximo duas palavras de abertura (`TERMO`,
# `PARTES`, `entre`) — para não cortar uma razão social que apenas contenha uma
# palavra de Administração no meio (`SECRETARIA DIGITAL E TECNOLOGIA LTDA`).
_MARCADORES_POLO_PUBLICO = (
    r"[A-ZÀ-Ú]{2,}/[A-ZÀ-Ú]{1,}"                       # sigla com barra: PMSP/SMC, SP/SMC
    r"|prefeitura(?:\s+\w+){0,4}"
    r"|munic[íi]pio(?:\s+de\s+s[ãa]o\s+paulo)?"
    r"|secretaria(?:\s+\w+){0,4}"
    r"|subprefeitura(?:\s+\w+){0,3}"
    r"|governo\s+do\s+estado"
    r"|fundo(?:\s+\w+){0,4}"
    r"|autarquia(?:\s+\w+){0,3}"
)
_ABERTURA_DE_PREAMBULO = (
    r"(?:termo|partes|contrato|ajuste|conv[êe]nio|aditamento|processo|"
    r"entre|celebrad[oa]|firmad[oa])"
)
_RE_FRONTEIRA_POLO_PUBLICO = re.compile(
    r"^(?:" + _ABERTURA_DE_PREAMBULO + r"\b[\s:]*){0,2}"
    r"(?:entre\s+a?\s*)?(?:" + _MARCADORES_POLO_PUBLICO + r")\b.*?\s+e\s+(?:a\s+|o\s+)?",
    re.S | re.I,
)

# Quantos caracteres antes do documento podem conter o nome. Acima de 200 a busca começa a
# atravessar a fronteira da frase anterior e capturar o órgão.
_JANELA_DA_ANCORA = 200


def contratada_por_ancora(alvo: str | None,
                          lado_publico: str | None = None) -> parte.ResultadoNome:
    """
    A parte contratada localizada pelo documento fiscal, quando não há rótulo na prosa.

    Explora o fato de que o publicador costuma escrever o documento fiscal
    junto da parte, mesmo quando não usa rótulo. O documento é uma âncora forte: o
    dígito verificador do módulo 11 raramente fecha por acidente, então um
    número que passa tem boa chance de ser um documento de verdade, e não um número de
    processo ou de empenho. Achado o documento, o problema deixa de ser "onde está
    o nome no texto" e passa a ser "onde ele começa e termina".

    Vem depois da camada 2 porque a precisão da âncora é menor que a do rótulo
    explícito.

    Não se remove a raiz do documento colada ao início do nome: a fonte às vezes
    inclui a raiz no próprio nome empresarial (`'12.345.678 FULANA DE TAL
    SOUZA - ME'` é a resposta correta, com o número).

    A validação final é `parte.porta_de_nome` — a mesma porta das camadas 1 e 2.
    """
    if not alvo:
        return parte.porta_de_nome(None)

    for achado in _RE_DOCUMENTO_NO_TEXTO.finditer(alvo):
        if not parte.classificar_documento(achado.group(0)).valido:
            continue
        antes = alvo[max(0, achado.start() - _JANELA_DA_ANCORA):achado.start()]
        antes = _RE_QUALIFICACAO_DO_DOCUMENTO.sub("", antes)
        antes = re.sub(r"\s*[,;]\s*$", "", antes)

        # A vírgula entra na classe de captura porque a busca é ancorada em `$` e razão
        # social tem vírgula: sem ela, 'GAMA ENGENHARIA, CONSTRUÇÕES E COMÉRCIO LTDA'
        # seria capturado como 'CONSTRUÇÕES E COMÉRCIO LTDA'. Com a vírgula na classe, o
        # nome inteiro é capturado e a fronteira esquerda, que ancora só em marcadores
        # de preâmbulo, decide o corte real.
        candidato = re.search(r"([A-ZÀ-Ú0-9][\w\sÀ-Ú&\.\-/',]{5,90})$", antes)
        if not candidato:
            continue
        bruto = candidato.group(1).strip()
        # `or bruto`: se o corte consumir tudo, o candidato inteiro é melhor que nada — a
        # porta ainda vai julgá-lo.
        bruto = _RE_FRONTEIRA_ESQUERDA.sub("", bruto) or bruto
        # Corte adicional em ` e ` só quando o lado esquerdo é o polo público
        # ('PMSP/SMC e NOME', 'entre a PREFEITURA … e a ASSOCIAÇÃO X'). Nunca em ` E `
        # interno de razão social, porque o marcador de polo público é condição do corte.
        bruto = _RE_FRONTEIRA_POLO_PUBLICO.sub("", bruto) or bruto

        resultado = parte.porta_de_nome(bruto, lado_publico_conhecido=lado_publico)
        if resultado.nome:
            # A ressalva é sempre acrescentada, mesmo quando a porta não estranhou nada:
            # a camada inteira depende de duas heurísticas de fronteira, e uma linha que
            # veio dela não tem a mesma garantia de uma que veio de rótulo.
            #
            # `documento` carrega o número que ancorou ESTE nome, para que não seja
            # gravado como documento da contratada o primeiro documento válido do
            # texto — que pode ser o CNPJ do próprio órgão.
            return parte.ResultadoNome(
                resultado.nome,
                (*resultado.ressalvas, "nome_por_ancora_de_documento"),
                documento=achado.group(0),
            )
    return parte.porta_de_nome(None)


# O ESTRATO

COM_ROTULO = "com_rotulo"
SEM_ROTULO = "sem_rotulo"


def estrato(alvo: str | None) -> str:
    """
    `com_rotulo` se a prosa nomeia a parte contratada; `sem_rotulo` se não.

    Sem esta distinção, a métrica de fornecedor seria uma média ponderada de duas
    tarefas de dificuldade muito diferente, sugerindo um desempenho uniforme que
    não existe.

    A classificação olha o alvo, não o texto completo, pela mesma razão que
    `alvo_da_analise` existe: um rótulo que aparece no formulário não é rótulo na
    prosa.

    `estrato` responde sobre a presença do rótulo, não sobre o sucesso da
    extração: uma publicação pode ter `CONTRATADA:` e a porta de limpeza recusar o
    valor.
    """
    if not alvo:
        return SEM_ROTULO
    for campo, padrao in _COMPILADOS:
        if campo == CAMPO_CONTRATADA and padrao.search(alvo):
            return COM_ROTULO
    return COM_ROTULO if _RE_DETENTORA_EM_PROSA.search(alvo) else SEM_ROTULO


# A GUARDA
# Um rótulo declarado nas duas listas seria ambíguo: mapeado e ignorado ao mesmo
# tempo, com o resultado dependendo da ordem de leitura em `ler`.

_ambiguos = set(ROTULOS) & NAO_MAPEADOS
if _ambiguos:
    raise AssertionError(f"rótulo em ROTULOS e em NAO_MAPEADOS: {sorted(_ambiguos)}")
del _ambiguos
