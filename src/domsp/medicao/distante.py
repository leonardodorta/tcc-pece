"""
distante — supervisão distante: pares (prosa, resposta) obtidos removendo o rótulo da prosa.

## A limitação do gabarito limpo

`gabarito.construir` produz pares limpos por redundância intradocumento: a resposta
vem do campo estruturado do formulário, a entrada é a prosa. A exigência das três
condições é restritiva: só entram atos com formulário que declara a contratada e cuja
resposta ocorre na prosa. O rendimento em pares limpos fica limitado a esses atos, e este
módulo produz pares adicionais, com ruído, a partir dos atos que não passam nelas.

## O que este módulo faz

Usa a camada 2 como professora: nos atos em que a prosa rotula a contratada
(`CONTRATADA:`, `DETENTORA:`, `PERMISSIONÁRIA:`, `DONATÁRIA:`), o rótulo dá a
resposta. Para que a entrada não contenha a pista, o rótulo é removido do texto e o
valor fica:

    original    "... CONTRATADA: EMPRESA EXEMPLO LTDA, CNPJ 00.000.000/0001-00 ..."
    sem rótulo  "... EMPRESA EXEMPLO LTDA, CNPJ 00.000.000/0001-00 ..."
    resposta                    "EMPRESA EXEMPLO LTDA"

O texto resultante é o que um ato de resíduo parece: um nome de empresa em prosa
corrida, sem rótulo que o anuncie.

## Por que não é circular, e onde está o limite

A pista é removida, então a regra que produziu o rótulo não pode ser reaprendida a
partir de um texto de onde o rótulo saiu. O modelo tem de usar contexto — sufixo
societário, proximidade de CNPJ, posição na frase.

O limite é outro, e é declarado: os pares deste módulo herdam os erros da camada 2.
Onde o rótulo aponta para o trecho errado, o par ensina o erro. Por isso a divisão
de uso é estrita:

    treino       pares deste módulo (com ruído)
    validação    pares deste módulo (para escolher época e limiar)
    teste        SÓ pares de `gabarito.construir` (limpos)

Treinar no ruidoso e avaliar no limpo é o que impede o número final de ser uma
medida do professor em vez do aluno.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .. import texto as T
from ..dominio import formulario as F
from ..dominio import triagem
from ..fonte import edicao as fonte
from ..modelo import rotulo as R
from .gabarito import Par

# OS RÓTULOS A REMOVER
#
# Reaproveitados de `formulario`, e não reescritos: se fossem reescritos aqui, um rótulo novo
# reconhecido pela camada 2 deixaria de ser removido da entrada de treino, e o modelo passaria
# a ver a pista sem que nada indicasse.

_PADROES_DE_ROTULO: tuple[re.Pattern[str], ...] = tuple(
    padrao for campo, padrao in F._COMPILADOS if campo == F.CAMPO_CONTRATADA
)

ESTRATO = "rotulo_removido"

# Motivos de um ato não render par. Nomeados porque a distribuição deles é o que diz se o
# rendimento caiu por mudança da fonte ou por defeito do módulo.
MOTIVO_JA_LIMPO = "ja_e_par_limpo"
MOTIVO_SEM_ROTULO = "sem_rotulo_na_prosa"
MOTIVO_PORTA_RECUSOU = "porta_recusou_o_valor"
MOTIVO_NOME_CURTO = "nome_curto_para_treino"
MOTIVO_PISTA_SOBROU = "pista_sobrou_antes_da_resposta"
MOTIVO_SEM_SPAN = "span_nao_localiza_apos_remover"
MOTIVO_CURTO = "texto_curto_demais"

# Abaixo disto o texto não dá contexto para o modelo decidir nada, e um exemplo desses ensina a
# adivinhar. Os 60 caracteres são um piso fixo, escolhido por este módulo.
MINIMO_TEXTO = 60

# O piso de tamanho do nome para servir de exemplo de treino. Mais rígido que o da porta, que é
# permissivo para não recusar sigla legítima numa coluna de dados. Aqui o critério é outro: um
# span de poucos caracteres não ensina a delimitar razão social.
MINIMO_NOME = 8


@dataclass(frozen=True)
class Rendimento:
    """
    A contabilidade da construção.

    `por_motivo` é a razão de cada ato recusado, e é o que permite ver deriva: se
    `span_nao_localiza_apos_remover` subir, é sinal de que a remoção do rótulo está
    removendo parte do valor.
    """

    pares: list[Par]
    apos_triagem: int
    por_motivo: Counter[str]

    @property
    def n(self) -> int:
        return len(self.pares)

    @property
    def rendimento(self) -> float:
        """Fração dos atos triados que rendeu par."""
        return self.n / self.apos_triagem if self.apos_triagem else 0.0


def remover_rotulo(alvo: str) -> tuple[str, int]:
    """
    Tira todos os rótulos de contratada do texto, deixando os valores. Devolve
    (texto, quantos_removidos).

    O recorte usa os grupos da própria expressão da camada 2: `span()` dá o casamento inteiro
    (rótulo mais valor) e `span("valor")` dá só o valor. Remover é manter o que está antes do
    casamento, o valor, e o que está depois do casamento.

    ## Duas decisões de recorte

    Remove todos, e não o primeiro. Numa publicação com `CONTRATANTE:` antes de
    `CONTRATADA /`, remover só o primeiro casamento tiraria o rótulo errado e a pista
    ficaria visível no texto de treino — o modelo aprenderia a procurar a palavra
    `CONTRATADA`, que é exatamente o que este módulo existe para evitar. A verificação `_tem_pista`,
    chamada em `de_publicacao`, confere a ausência da pista antes da resposta e guarda contra isso.

    Loop até estabilizar, com teto. Um `for` sobre os padrões removeria um casamento de
    cada padrão, e não todos os casamentos de cada um. O teto existe porque a remoção encurta a
    string mas não garante progresso se algum padrão casar vazio — e um `while` sem teto num
    módulo que roda sobre o corpus inteiro é um travamento à espera de dado
    estranho.

    Uma implementação ingênua — `texto[:inicio] + texto[inicio_do_valor:]` — deixaria pendurado
    o trecho entre o fim do valor e o fim do casamento.
    """
    MAX_PASSADAS = 12
    removidos = 0
    for _ in range(MAX_PASSADAS):
        for padrao in _PADROES_DE_ROTULO:
            achado = padrao.search(alvo)
            if achado is None:
                continue
            inicio, fim = achado.span()
            vi, vf = achado.span("valor")
            # antes do rótulo + o valor + depois do casamento inteiro
            alvo = alvo[:inicio] + alvo[vi:vf] + alvo[fim:]
            removidos += 1
            break
        else:
            break
    return alvo, removidos


def de_publicacao(bruto: str, conteudo: str) -> tuple[Par | None, str]:
    """
    Um par de treino a partir de uma publicação já triada. Devolve (par|None, motivo).

    Recusa o ato que já é par limpo: ele pertence ao conjunto de teste, e deixá-lo entrar no
    treino vazaria o teste. É a primeira verificação de propósito — a mais barata e a que tem
    consequência mais grave se falhar.

    A resposta vem de `formulario.contratada_da_prosa`, a mesma função que a camada 2 do
    pipeline usa. Não de uma extração própria: se este módulo lesse o rótulo à sua maneira, o
    treino aprenderia uma tarefa que a camada 2 não executa, e a comparação entre cobertura do
    modelo e cobertura da camada deixaria de fazer sentido.
    """
    form = F.ler(bruto)

    if F.contratada_do_formulario(form).nome:
        return None, MOTIVO_JA_LIMPO

    alvo = F.alvo_da_analise(form, conteudo)
    if not alvo or len(alvo) < MINIMO_TEXTO:
        return None, MOTIVO_CURTO

    # A resposta, pela função autoritativa da camada 2 — já passada pela porta única.
    aceito = F.contratada_da_prosa(alvo)
    if not aceito.nome:
        return None, MOTIVO_SEM_ROTULO

    # Um nome curto sobrevive à porta (o mínimo dela é permissivo de propósito, para não recusar
    # sigla legítima) mas não serve como exemplo de treino: um span de três caracteres ensina a
    # adivinhar. O piso aqui é mais rígido que o da porta, e é escolha deste módulo.
    if len(aceito.nome) < MINIMO_NOME:
        return None, MOTIVO_NOME_CURTO

    entrada, removidos = remover_rotulo(alvo)
    if removidos == 0:
        return None, MOTIVO_SEM_ROTULO

    # O span tem de existir no texto SEM o rótulo. Se não existir, a remoção comeu o valor ou a
    # porta reescreveu o nome além do que o texto contém — nos dois casos o par mentiria sobre a
    # posição, e um span errado é pior que um par ausente.
    span = R.localizar(aceito.nome, entrada)
    if span is None:
        return None, MOTIVO_SEM_SPAN

    # A pista não pode ter sobrado antes da resposta. A verificação vive no código e não só no
    # teste porque um par com pista é pior que par nenhum: ele treina o modelo a ler o rótulo e
    # a métrica não acusa.
    if _tem_pista(entrada, span.inicio):
        return None, MOTIVO_PISTA_SOBROU

    return Par(
        prosa=entrada,
        nome_correto=aceito.nome,
        estrato=ESTRATO,
        documento=None,
        origem_span=span.origem,
    ), ESTRATO


# Quantos caracteres antes do span são inspecionados em busca de pista. 60 cobre o rótulo mais o
# separador e alguma pontuação; ir muito além acusaria uma menção legítima a `CONTRATADA` numa
# cláusula distante, que não é pista.
JANELA_PISTA = 60

_RE_PISTA = re.compile(r"(?i)\b(CONTRATAD[AO]|DETENTORA|PERMISSION[AÁ]RI[AO]|DONAT[AÁ]RI[AO])\b")


def _tem_pista(texto: str, inicio_do_span: int) -> bool:
    """Alguma palavra de rótulo sobrou nos caracteres imediatamente antes da resposta."""
    janela = texto[max(0, inicio_do_span - JANELA_PISTA) : inicio_do_span]
    return _RE_PISTA.search(janela) is not None


def construir(corpus: str | Path, n_edicoes: int | None = None) -> Rendimento:
    """
    Os pares de treino do corpus, com a contabilidade.

    `n_edicoes` limita às primeiras N edições, para desenvolvimento. Por padrão fica
    None (todas as edições): o corte por período é feito depois, na divisão temporal.
    """
    pares: list[Par] = []
    motivos: Counter[str] = Counter()
    triados = 0

    for pub, conteudo in _publicacoes_triadas(corpus, n_edicoes):
        triados += 1
        par, motivo = de_publicacao(pub.get("conteudo") or "", conteudo)
        if par is None:
            motivos[motivo] += 1
        else:
            pares.append(par)
            motivos[ESTRATO] += 1

    return Rendimento(pares=pares, apos_triagem=triados, por_motivo=motivos)


def _publicacoes_triadas(
    corpus: str | Path, n_edicoes: int | None
) -> Iterator[tuple[dict, str]]:
    """
    As publicações que a triagem aceita, com o texto já convertido.

    Repete a lógica de dois estágios de `pipeline.silver` de propósito: importar `pipeline`
    aqui inverteria a direção da dependência — `pipeline` é execução e `medicao` é medida.
    """
    for arquivo in fonte.arquivos(corpus)[:n_edicoes]:
        ed = fonte.carregar(arquivo)
        for pub in ed.publicacoes:
            serie = pub.get("serie") or ""
            veiculo = pub.get("veiculo") or ""
            bruto = pub.get("conteudo") or ""
            if not bruto.strip():
                continue
            if triagem.eh_metadado_suficiente(serie, veiculo):
                if triagem.triar(serie, veiculo)[0] is not None:
                    continue
                conteudo = T.html_para_texto(bruto)
            else:
                conteudo = T.html_para_texto(bruto)
                if triagem.triar(serie, veiculo, conteudo)[0] is not None:
                    continue
            yield pub, conteudo


def data_da_publicacao(corpus: str | Path, n_edicoes: int | None = None) -> Iterator[str]:
    """
    A data de edição de cada publicação triada que rende par, na mesma ordem de `construir`.

    Existe porque a divisão temporal precisa da data e o `Par` não a carrega, e
    acrescentar a data ao `Par` mudaria o contrato que `gabarito` já usa.
    """
    for arquivo in fonte.arquivos(corpus)[:n_edicoes]:
        ed = fonte.carregar(arquivo)
        for pub, conteudo in _publicacoes_de(ed):
            par, _ = de_publicacao(pub.get("conteudo") or "", conteudo)
            if par is not None:
                yield ed.data


def _publicacoes_de(ed) -> Iterator[tuple[dict, str]]:
    """As publicações triadas de UMA edição já carregada."""
    for pub in ed.publicacoes:
        serie = pub.get("serie") or ""
        veiculo = pub.get("veiculo") or ""
        bruto = pub.get("conteudo") or ""
        if not bruto.strip():
            continue
        if triagem.eh_metadado_suficiente(serie, veiculo):
            if triagem.triar(serie, veiculo)[0] is not None:
                continue
            conteudo = T.html_para_texto(bruto)
        else:
            conteudo = T.html_para_texto(bruto)
            if triagem.triar(serie, veiculo, conteudo)[0] is not None:
                continue
        yield pub, conteudo


def com_data(corpus: str | Path, n_edicoes: int | None = None) -> list[tuple[str, Par]]:
    """
    Os pares com a data da edição ao lado, numa travessia só.

    `construir` e `data_da_publicacao` continuam existindo porque medem coisas
    diferentes — rendimento e datas.
    """
    resultado: list[tuple[str, Par]] = []
    for arquivo in fonte.arquivos(corpus)[:n_edicoes]:
        ed = fonte.carregar(arquivo)
        for pub, conteudo in _publicacoes_de(ed):
            par, _ = de_publicacao(pub.get("conteudo") or "", conteudo)
            if par is not None:
                resultado.append((ed.data, par))
    return resultado
