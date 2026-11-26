"""
gabarito.py — o gabarito automático por redundância intradocumento

Medir F1 de extração exige pares (texto, resposta). Este módulo os constrói sem
anotação manual, por redundância intradocumento.

## A observação que resolve

A fonte publica o mesmo dado duas vezes: as publicações do Painel de Negócios trazem
a parte contratada no campo estruturado do formulário E na prosa do despacho,
escritos independentemente pelo publicador. Escondendo o formulário e pedindo ao
extrator que leia só a prosa, o campo do formulário passa a ser a resposta certa —
obtêm-se pares `(prosa, nome_correto)` sem anotação manual.

## As três condições que fazem o par ser gabarito

Um par só entra no gabarito se as três valem:

  1. o formulário declara a contratada — há resposta a comparar;
  2. o formulário tem campo de prosa — há onde o extrator procurar;
  3. a resposta OCORRE no texto da prosa — verificada por `modelo.rotulo.localizar`,
     o mesmo alinhamento que a rotulagem do modelo usa.

Sem a condição 3, o par mediria a redação do publicador (que escreveu o nome no
formulário mas não na prosa) em vez da extração. A condição 3 define também o viés
declarado do gabarito — ele cobre os casos em que a fonte se repetiu, que não são
amostra aleatória de todos os atos. A avaliação é estratificada por
`formulario.estrato` como resposta a esse viés.

## Não é circular

O gabarito vem do campo estruturado; a entrada do extrator é a prosa — dois textos
diferentes, escritos independentemente pelo publicador. É legítimo que a prosa
contenha, ela própria, um rótulo (`CONTRATADA:`), e é isso que a estratificação
expõe: em `com_rotulo` a prosa tem rótulo e a tarefa é quase determinística; em
`sem_rotulo` não tem e a tarefa é difícil. Publicar os dois números separados impede
o gabarito de parecer circular.

## As três verificações independentes, todas de custo humano zero

  (a) numeral × extenso na mesma frase — a fonte escreve `R$ 1.220,00 (mil duzentos
      e vinte reais)`, e dois parsers independentes (`dinheiro.para_float` e
      `dinheiro.numero_por_extenso`) leem a mesma frase por caminhos de código
      distintos. Um erro de leitura do numeral quase nunca coincide com o mesmo erro
      no extenso;
  (b) dígito verificador módulo 11 — CNPJ/CPF carregam DV aritmético
      (`parte.classificar_documento`). Um número lido da posição errada quase nunca
      fecha o DV;
  (c) concordância entre camadas — formulário (camada 1) e prosa (camada 2) são
      lidos por caminhos de código distintos (`formulario.contratada_do_formulario`
      e `formulario.contratada_da_prosa`). Onde as duas respondem e concordam, o
      dado está confirmado por duas leituras que não compartilham código.

## Dependências

`domsp.dominio`, `domsp.modelo.rotulo`, `domsp.texto`, `domsp.fonte` e biblioteca padrão.
Este módulo não importa pyspark, torch, boto3 nem pandas: o gabarito é função pura sobre
HTML/texto. O corpus é lido por `fonte`, que carrega as edições.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .. import texto as T
from ..dominio import dinheiro, parte, triagem
from ..dominio import formulario as F
from ..fonte import edicao as fonte
from ..modelo import rotulo as R

# O PAR DE GABARITO


@dataclass(frozen=True)
class Par:
    """
    Um par de gabarito: a prosa que o extrator lê, e o nome que o formulário declarou.

    `estrato` é `com_rotulo` ou `sem_rotulo` (de `formulario.estrato`): a
    estratificação é a resposta ao viés do gabarito e ao risco de redundância
    interna inflar o desempenho aparente.

    `documento` é o identificador da publicação (para auditar qualquer par até a
    fonte). `origem_span` diz como a condição 3 foi satisfeita — `exato` (o nome
    ocorre literalmente na prosa) ou `prefixo` (ocorre por prefixo estendido) —,
    porque a confiabilidade das duas é diferente.
    """

    prosa: str
    nome_correto: str
    estrato: str
    documento: str
    origem_span: str


# A CONSTRUÇÃO DO GABARITO


@dataclass
class Construcao:
    """
    O resultado de construir o gabarito, com a contabilidade de por que cada publicação
    entrou ou não. Os contadores tornam auditável a afirmação "o gabarito tem N pares".
    """

    pares: list[Par] = field(default_factory=list)
    publicacoes: int = 0
    apos_triagem: int = 0
    sem_formulario: int = 0
    sem_prosa: int = 0
    sem_contratada_no_form: int = 0    # falha a condição 1
    resposta_nao_ocorre_na_prosa: int = 0   # falha a condição 3
    por_estrato: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.pares)


def construir(corpus: str | Path, n_edicoes: int | None = None) -> Construcao:
    """
    Constrói o gabarito automático sobre as edições de `corpus`.

    Para cada publicação, aplica a triagem e as três condições, nesta ordem:

      0. triagem — se a publicação é descartada (licitação, fora de escopo), não é
         ato de contratação e não entra. Usa o mesmo `triagem.motivo_recusa` do
         pipeline, para que o gabarito seja construído sobre exatamente os atos que
         o pipeline processa;
      -  o formulário existe (`form.existe`); sem ele, conta em `sem_formulario`;
      2. o formulário tem campo de prosa (`form.tem_prosa`) — há onde procurar; sem
         ele, conta em `sem_prosa`;
      1. o formulário declara a contratada (`contratada_do_formulario`), pela porta
         única — é a resposta; sem ela, conta em `sem_contratada_no_form`;
      3. a resposta OCORRE na prosa (`rotulo.localizar`) — o par é justo; senão, conta
         em `resposta_nao_ocorre_na_prosa`.

    A ordem de teste no código é: existência do formulário, condição 2, condição 1,
    condição 3. Cada condição só é testada se a anterior passou, e cada falha é
    contada na sua categoria.

    `n_edicoes=None` usa todas as edições do corpus; um inteiro limita.
    """
    c = Construcao()
    arquivos = fonte.arquivos(corpus)
    if n_edicoes is not None:
        arquivos = arquivos[:n_edicoes]

    for arquivo in arquivos:
        ed = fonte.carregar(arquivo)
        for pub in ed.publicacoes:
            c.publicacoes += 1
            html = pub.get("conteudo") or ""
            form = F.ler(html)
            completo = T.html_para_texto(html)
            alvo = F.alvo_da_analise(form, completo)

            # condição 0: só atos que o pipeline processaria
            if triagem.motivo_recusa(pub.get("serie"), pub.get("veiculo"), alvo):
                continue
            c.apos_triagem += 1

            # condição 1 (parte 1): existe formulário?
            if not form.existe:
                c.sem_formulario += 1
                continue
            # condição 2: o formulário tem prosa?
            if not form.tem_prosa:
                c.sem_prosa += 1
                continue
            # condição 1 (parte 2): o formulário declara a contratada?
            resultado = F.contratada_do_formulario(form, alvo)
            if not resultado.nome:
                c.sem_contratada_no_form += 1
                continue
            # condição 3: a resposta ocorre na prosa?
            span = R.localizar(resultado.nome, alvo, R.CONTRATADA)
            if span is None:
                c.resposta_nao_ocorre_na_prosa += 1
                continue

            estr = F.estrato(alvo)
            c.pares.append(Par(
                prosa=alvo, nome_correto=resultado.nome, estrato=estr,
                documento=str(pub.get("documento") or ""), origem_span=span.origem,
            ))
            c.por_estrato[estr] = c.por_estrato.get(estr, 0) + 1

    return c


def pares(corpus: str | Path, n_edicoes: int | None = None) -> list[Par]:
    """Só os pares, para quem não precisa da contabilidade. Conveniência sobre `construir`."""
    return construir(corpus, n_edicoes).pares


# VERIFICAÇÃO (a) — NUMERAL × EXTENSO


@dataclass(frozen=True)
class ConcordanciaValor:
    """O resultado de conferir numeral contra extenso sobre um conjunto de textos."""

    conferidos: int      # ocorrências com numeral E extenso na mesma frase
    concordantes: int    # em que os dois parsers deram o mesmo número
    total_ocorrencias: int

    @property
    def cobertura(self) -> float:
        """Fração das ocorrências monetárias que têm extenso a conferir."""
        return self.conferidos / self.total_ocorrencias if self.total_ocorrencias else 0.0

    @property
    def taxa_concordancia(self) -> float:
        return self.concordantes / self.conferidos if self.conferidos else 0.0


def verificar_valor(textos: Iterable[str]) -> ConcordanciaValor:
    """
    Verificação (a): numeral × extenso, dois parsers independentes na mesma frase.

    Para cada ocorrência monetária, `dinheiro.ocorrencias_monetarias` já traz o
    `extenso` lido por `dinheiro.numero_por_extenso` (caminho de código distinto do
    `dinheiro.para_float` que leu o numeral) e o booleano `confere`. Esta função
    apenas agrega.

    Não custa anotação humana: se o extrator leu `R$ 2.718,89` como `R$ 27.188,90`
    (ponto no lugar errado), o extenso "dois mil setecentos e dezoito reais e
    oitenta e nove centavos" não bateria.
    """
    conferidos = concordantes = total = 0
    for texto in textos:
        for oc in dinheiro.ocorrencias_monetarias(texto):
            total += 1
            if oc.extenso is not None:
                conferidos += 1
                if oc.confere:
                    concordantes += 1
    return ConcordanciaValor(conferidos, concordantes, total)


# VERIFICAÇÃO (b) — DÍGITO VERIFICADOR MÓDULO 11


@dataclass(frozen=True)
class ConcordanciaDocumento:
    """
    O resultado de conferir o DV dos documentos fiscais de um conjunto de textos,
    separado por tipo.

    A separação CNPJ × CPF importa: rodar sobre a prosa crua (e não sobre o campo já
    extraído) faz `documentos_no_texto` capturar sequências de 11 dígitos que não são
    CPF — trechos de dotação orçamentária, números de processo, protocolos. Elas
    quase nunca fecham o módulo 11, e misturá-las no total afundaria a taxa com um
    artefato de medição. Por isso a taxa reportada é a de CNPJ, e a de CPF fica à
    parte, com a ressalva de que inclui esses falsos positivos.
    """

    cnpj_verificaveis: int
    cnpj_validos: int
    cpf_verificaveis: int
    cpf_validos: int
    total: int           # documentos localizados (todos os tipos)

    @property
    def taxa_dv_cnpj(self) -> float:
        return self.cnpj_validos / self.cnpj_verificaveis if self.cnpj_verificaveis else 0.0

    @property
    def taxa_dv_cpf(self) -> float:
        return self.cpf_validos / self.cpf_verificaveis if self.cpf_verificaveis else 0.0

    # Não há taxa agregada que some CNPJ e CPF: sobre a prosa crua, o padrão de 11
    # dígitos do CPF pode casar com trechos de dotação orçamentária, então a taxa de CPF
    # tende a ser mais baixa, e somar as duas produziria um número que não
    # descreve nem uma população nem a outra.


def verificar_documento(textos: Iterable[str]) -> ConcordanciaDocumento:
    """
    Verificação (b): dígito verificador módulo 11, pura aritmética, separada por tipo.

    Localiza os documentos fiscais no texto (`parte.documentos_no_texto`) e classifica
    cada um (`parte.classificar_documento`, que aplica o DV). Um documento mascarado
    pela fonte não tem dígitos suficientes e não conta como verificável.

    Custo humano zero: um CNPJ lido deslocado por um dígito quase nunca fecha o
    módulo 11 (`taxa_dv_cnpj`). A `taxa_dv_cpf` é reportada à parte porque, sobre a
    prosa crua, o localizador pode capturar sequências de 11 dígitos de dotações
    orçamentárias como se fossem CPF (ver o docstring da dataclass) — artefato de medição, não erro de
    leitura.
    """
    cnpj_v = cnpj_ok = cpf_v = cpf_ok = total = 0
    for texto in textos:
        for bruto in parte.documentos_no_texto(texto):
            total += 1
            doc = parte.classificar_documento(bruto)
            digitos = sum(ch.isdigit() for ch in bruto)
            if doc.tipo == "cnpj" and digitos == 14:
                cnpj_v += 1
                cnpj_ok += doc.valido
            elif doc.tipo == "cpf" and digitos == 11:
                cpf_v += 1
                cpf_ok += doc.valido
    return ConcordanciaDocumento(cnpj_v, cnpj_ok, cpf_v, cpf_ok, total)


# VERIFICAÇÃO (c) — CONCORDÂNCIA ENTRE CAMADAS


@dataclass(frozen=True)
class ConcordanciaCamadas:
    """O resultado de comparar a camada 1 (formulário) com a camada 2 (prosa)."""

    ambas_responderam: int
    concordantes: int
    total_atos: int

    @property
    def taxa_concordancia(self) -> float:
        return self.concordantes / self.ambas_responderam if self.ambas_responderam else 0.0


def textos_do_corpus(corpus: str | Path, n_edicoes: int | None = None) -> Iterator[str]:
    """
    Os textos (alvo de análise) dos atos que sobrevivem à triagem, um por vez.

    Gerador, e não lista: materializar todos os textos juntos desperdiça memória
    quando as verificações de valor e de documento consomem um por vez. É o alimento
    de `verificar_valor` e `verificar_documento`.
    """
    arquivos = fonte.arquivos(corpus)
    if n_edicoes is not None:
        arquivos = arquivos[:n_edicoes]
    for arquivo in arquivos:
        ed = fonte.carregar(arquivo)
        for pub in ed.publicacoes:
            html = pub.get("conteudo") or ""
            form = F.ler(html)
            completo = T.html_para_texto(html)
            alvo = F.alvo_da_analise(form, completo)
            if triagem.motivo_recusa(pub.get("serie"), pub.get("veiculo"), alvo):
                continue
            if alvo:
                yield alvo


def verificar_concordancia_camadas(corpus: str | Path,
                                   n_edicoes: int | None = None,
                                   limiar_similaridade: float = 0.90) -> ConcordanciaCamadas:
    """
    Verificação (c): concordância entre a leitura do formulário e a da prosa.

    Onde as duas camadas respondem — `contratada_do_formulario` (campo estruturado) e
    `contratada_da_prosa` (rótulo na prosa) —, compara os dois nomes. Os dois são
    lidos por caminhos de código distintos e não compartilham a regra de extração (só
    a porta de limpeza), então a concordância não depende de gabarito externo nenhum.

    A comparação é por similaridade com limiar (não igualdade exata) porque o
    formulário e a prosa grafam o mesmo nome com pequenas diferenças de acento e
    sufixo — a mesma razão de `rotulo.localizar` comparar na forma comparável.
    `limiar_similaridade` tem padrão 0.90 e é comparado com a razão do `SequenceMatcher`
    sobre os nomes sem acento, sem espaço duplicado e em minúsculas.
    """
    from difflib import SequenceMatcher

    ambas = concordantes = total = 0
    arquivos = fonte.arquivos(corpus)
    if n_edicoes is not None:
        arquivos = arquivos[:n_edicoes]

    for arquivo in arquivos:
        ed = fonte.carregar(arquivo)
        for pub in ed.publicacoes:
            html = pub.get("conteudo") or ""
            form = F.ler(html)
            completo = T.html_para_texto(html)
            alvo = F.alvo_da_analise(form, completo)
            if triagem.motivo_recusa(pub.get("serie"), pub.get("veiculo"), alvo):
                continue
            total += 1
            r1 = F.contratada_do_formulario(form, alvo)
            r2 = F.contratada_da_prosa(alvo, form.campos.get(F.CAMPO_CONTRATANTE))
            if r1.nome and r2.nome:
                ambas += 1
                a = T.remover_acentos(T.colapsar_espacos(r1.nome)).lower()
                b = T.remover_acentos(T.colapsar_espacos(r2.nome)).lower()
                if SequenceMatcher(None, a, b).ratio() >= limiar_similaridade:
                    concordantes += 1
    return ConcordanciaCamadas(ambas, concordantes, total)

