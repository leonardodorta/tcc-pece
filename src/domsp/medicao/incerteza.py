"""
incerteza.py — o intervalo de confiança de cada número publicado

Nenhuma métrica deste pacote é publicada como ponto: cada função aqui devolve a
mesma quíntupla — `(valor, limite_inferior, limite_superior, n, metodo)` —, que
corresponde às colunas `proportion`, `ci_lower`, `ci_upper`, `denominator` e
`ci_method` de `gold_quality` (ver `contrato.esquema.QUALIDADE`), sem camada de
tradução por método. `metodo` é um valor de `vocabulario.METODOS_INTERVALO`,
verificado na guarda ao fim do módulo.

Três métodos, cada um resolvendo um problema que o óbvio (Wald/normal) não resolve:

- Wilson para proporção: a aproximação normal colapsa para `[1; 1]` quando
  acertos == total, porque `p(1-p) = 0` — certeza a partir de uma amostra finita.
  Wilson não colapsa porque não centra o intervalo em `p`.
- Bootstrap para F1: F1 é média harmônica de duas proporções sobre a mesma amostra,
  e não há forma fechada para seu intervalo sem supor a correlação entre precisão e
  revocação. O bootstrap reconstrói a distribuição amostral reamostrando com
  reposição; a semente fixa (`SEMENTE`) garante reprodutibilidade.
- Katz para razão de contagens: uma razão vive em `(0, ∞)` com erro multiplicativo,
  então o intervalo simétrico não existe na escala linear. Katz constrói na escala
  log (onde a razão volta a ser simétrica) e exponencia de volta.

Não há teste de hipótese aqui (nem McNemar), e por isso `mcnemar` não aparece em
`vocabulario.METODOS_INTERVALO`.

Só biblioteca padrão (`math`, `random`) e `contrato.vocabulario`; nenhum `scipy`. O `z` de 95% é
uma constante nomeada, para que a fórmula de Wilson e de Katz seja conferível à mão.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..contrato import vocabulario as V

# CONSTANTES DECLARADAS

# O quantil normal de 95% bilateral (z_{0,975}). Constante nomeada e não
# `scipy.stats.norm.ppf(0.975)`, para que a conta de Wilson e de Katz seja conferível
# à mão.
Z_95 = 1.959963984540054

# A confiança que todos os intervalos deste módulo reportam.
CONFIANCA = 0.95

# A semente do bootstrap. Fixa e declarada porque o bootstrap é aleatório, e um
# intervalo publicado que mude a cada execução é irreprodutível.
SEMENTE = 20260504  # valor arbitrário e fixo

# Quantas reamostragens o bootstrap faz. Os limites publicados são arredondados a quatro
# casas decimais, como os demais intervalos deste módulo.
REAMOSTRAS = 2000


# O RESULTADO — a quíntupla que alimenta gold_quality


@dataclass(frozen=True)
class Intervalo:
    """
    Um número publicável: o valor, seus dois extremos, o tamanho da amostra e o método.

    A ordem dos campos e a tupla que `__iter__` devolve casam com as colunas
    `proportion`, `ci_lower`, `ci_upper`, `denominator`, `ci_method` de `gold_quality`.
    Quem grava a linha faz `valor, li, ls, n, metodo = intervalo`.

    `metodo` é sempre um valor de `vocabulario.METODOS_INTERVALO` — a guarda no fim do
    módulo verifica isso.
    """

    valor: float
    limite_inferior: float
    limite_superior: float
    n: int
    metodo: str

    def __iter__(self):
        yield self.valor
        yield self.limite_inferior
        yield self.limite_superior
        yield self.n
        yield self.metodo

    @property
    def largura(self) -> float:
        """A largura do intervalo. Um número só, útil para ordenar por incerteza."""
        return self.limite_superior - self.limite_inferior


# WILSON — proporção


def wilson(acertos: int, total: int, z: float = Z_95) -> Intervalo:
    """
    O intervalo de Wilson para a proporção `acertos/total`.

    A fórmula, para `p = acertos/total` e `z` o quantil normal:

        centro = (p + z²/2n) / (1 + z²/n)
        raio   = (z/(1 + z²/n)) · √( p(1-p)/n + z²/4n² )

    e o intervalo é `centro ± raio`. O valor publicado é `p` (a proporção
    observada), não o `centro` de Wilson — o centro é um artefato do método,
    deslocado em direção a 0,5. Só os limites vêm de Wilson.

    Por que Wilson e não Wald: para `acertos == total`, `p(1-p) = 0` e Wald
    devolveria raio zero, isto é `[1; 1]` — certeza a partir de uma amostra finita.
    Wilson mantém `z²/n` no numerador do centro e sob a raiz, então o limite
    inferior fica abaixo de 1 mesmo com zero erros observados.

    `total == 0` devolve `[0; 1]` com valor 0: sem observação nenhuma, a proporção
    verdadeira pode ser qualquer coisa.
    """
    if total <= 0:
        return Intervalo(0.0, 0.0, 1.0, 0, "wilson")
    if acertos < 0 or acertos > total:
        raise ValueError(f"acertos={acertos} fora de [0, {total}]")

    p = acertos / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centro = (p + z2 / (2 * total)) / denom
    raio = (z / denom) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    li = max(0.0, centro - raio)
    ls = min(1.0, centro + raio)
    return Intervalo(round(p, 4), round(li, 4), round(ls, 4), total, "wilson")


# BOOTSTRAP — F1 e qualquer estatística que não seja proporção simples


def _f1(vp: int, fp: int, fn: int) -> float:
    """
    F1 a partir das contagens de verdadeiro-positivo, falso-positivo, falso-negativo.

    Definido aqui porque o bootstrap precisa recalcular F1 a cada reamostragem: a
    estatística e o seu intervalo têm de compartilhar a mesma definição.

    A definição de revocação segue a desta tarefa, não a do CoNLL genérico. Cada caso
    tem exatamente uma resposta certa, e o extrator ou acerta (vp), ou responde
    errado (fp), ou se abstém (fn) — mutuamente exclusivos, `vp + fp + fn = n`. Logo:

      - precisão = vp / (vp+fp)  — dos que respondeu, quantos acertou;
      - revocação = vp / (vp+fp+fn) = vp / n  — do que havia, quanto recuperou certo.

    Um caso `fp` (respondeu errado) também é uma resposta certa que se perdeu, e por
    isso entra no denominador da revocação. Usar o `vp/(vp+fn)` do CoNLL contaria o
    `fp` só contra a precisão e daria uma revocação inflada que não bate com
    `wilson(vp, n)`, que mede a mesma proporção `vp/n` — duas definições de revocação
    no mesmo módulo. Aqui as duas coincidem por construção.
    """
    if vp == 0:
        return 0.0
    n = vp + fp + fn
    precisao = vp / (vp + fp) if (vp + fp) else 0.0
    revocacao = vp / n if n else 0.0
    if precisao + revocacao == 0:
        return 0.0
    return 2 * precisao * revocacao / (precisao + revocacao)


def bootstrap_f1(casos: Sequence[tuple[bool, bool]],
                 reamostras: int = REAMOSTRAS,
                 semente: int = SEMENTE, z: float = Z_95) -> Intervalo:
    """
    O intervalo de F1 por bootstrap percentílico sobre casos `(respondeu, acertou)`.

    Cada caso é um par de booleanos:
      - `respondeu`: o extrator devolveu um nome para este caso (não abstenção);
      - `acertou`: o nome devolvido bate com o gabarito.

    Daí, por caso: verdadeiro-positivo = `respondeu and acertou`; falso-positivo =
    `respondeu and not acertou`; falso-negativo = `not respondeu` (o gabarito tem
    resposta e o extrator não a deu). Nesta tarefa todo caso do gabarito tem resposta
    certa — o gabarito automático só contém pares em que a fonte declarou a
    contratada —, então não há verdadeiro-negativo, e é por isso que F1 (e não
    acurácia) é a métrica.

    O procedimento: reamostra `n` casos com reposição `reamostras` vezes, recalcula F1
    em cada reamostra, e devolve os percentis 2,5% e 97,5% dessa distribuição. O
    valor publicado é o F1 da amostra original, não a média das reamostras.

    `semente` fixa a reprodutibilidade. `random.Random(semente)` cria um gerador local
    em vez de mexer no estado global de `random`.

    Menos de dois casos: bootstrap não tem o que reamostrar de forma útil, então
    devolve o F1 pontual com intervalo `[valor; valor]` e deixa `n` dizer que a base é
    insuficiente.
    """
    n = len(casos)
    vp0 = sum(1 for r, a in casos if r and a)
    fp0 = sum(1 for r, a in casos if r and not a)
    fn0 = sum(1 for r, a in casos if not r)
    valor = _f1(vp0, fp0, fn0)

    if n < 2:
        return Intervalo(round(valor, 4), round(valor, 4), round(valor, 4), n,
                         "bootstrap")

    rng = random.Random(semente)
    f1s: list[float] = []
    indices = range(n)
    for _ in range(reamostras):
        amostra = [casos[rng.choice(indices)] for _ in indices]
        vp = sum(1 for r, a in amostra if r and a)
        fp = sum(1 for r, a in amostra if r and not a)
        fn = sum(1 for r, a in amostra if not r)
        f1s.append(_f1(vp, fp, fn))
    f1s.sort()

    alfa = (1 - _confianca_de(z)) / 2
    li = _percentil(f1s, alfa)
    ls = _percentil(f1s, 1 - alfa)
    return Intervalo(round(valor, 4), round(li, 4), round(ls, 4), n, "bootstrap")


def _percentil(ordenados: Sequence[float], q: float) -> float:
    """
    O percentil `q` (0..1) de uma sequência já ordenada, por interpolação linear.

    Interpolação e não índice truncado porque, com 2000 reamostras, o percentil de
    2,5% cai entre os índices 49 e 50, e arredondar para um deles enviesaria o limite.
    """
    if not ordenados:
        return 0.0
    if q <= 0:
        return ordenados[0]
    if q >= 1:
        return ordenados[-1]
    pos = q * (len(ordenados) - 1)
    baixo = int(math.floor(pos))
    alto = int(math.ceil(pos))
    if baixo == alto:
        return ordenados[baixo]
    fracao = pos - baixo
    return ordenados[baixo] * (1 - fracao) + ordenados[alto] * fracao


# KATZ — razão de duas contagens


def katz(contagem_a: int, contagem_b: int, z: float = Z_95) -> Intervalo:
    """
    O intervalo de Katz para a razão `contagem_a / contagem_b` (log-normal).

    Uma razão não tem intervalo simétrico na escala linear: ela vive em `(0, ∞)` e o
    erro é multiplicativo. Katz constrói o intervalo na escala log, onde ele volta a
    ser simétrico, e exponencia:

        log(a/b) ± z · √(1/a + 1/b)

    Tratando cada contagem como Poisson independente, o erro-padrão de `log(a/b)` é
    `√(1/a + 1/b)`, que é o que o código usa.

    O valor publicado é `a/b`, a razão observada. Os limites vêm do log.

    Por que não tratar como proporção: um intervalo simétrico em torno de `a/b` na
    escala linear pode descer abaixo de zero, sem sentido para uma razão, e trata `2`
    (dobro) e `0,5` (metade) como distâncias diferentes da unidade quando elas são a
    mesma distância multiplicativa. A leitura de "contém 1?" — que decide se a razão
    difere da unidade — só é válida na escala em que a unidade é o ponto neutro,
    e essa é a log.

    Contagem zero em qualquer lado: `log(0)` e `1/0` não existem. Devolve a razão
    pontual (0 ou infinito, conforme o lado) com intervalo `[0; inf]` e deixa `n`
    (aqui `a + b`) sinalizar a base.
    """
    n = contagem_a + contagem_b
    if contagem_a < 0 or contagem_b < 0:
        raise ValueError("contagens não podem ser negativas")
    if contagem_a == 0 or contagem_b == 0:
        razao = (contagem_a / contagem_b) if contagem_b else math.inf
        return Intervalo(round(razao, 4) if math.isfinite(razao) else math.inf,
                         0.0, math.inf, n, "katz")

    razao = contagem_a / contagem_b
    log_razao = math.log(razao)
    erro = math.sqrt(1.0 / contagem_a + 1.0 / contagem_b)
    li = math.exp(log_razao - z * erro)
    ls = math.exp(log_razao + z * erro)
    return Intervalo(round(razao, 4), round(li, 4), round(ls, 4), n, "katz")


# AUXILIAR — o z de volta em confiança


def _confianca_de(z: float) -> float:
    """
    A confiança bilateral correspondente a um `z`, pela função erro.

    `bootstrap_f1` recebe `z` para casar a assinatura das outras funções, mas precisa
    da confiança (para achar `alfa`). Deriva um do outro por `erf`, que é a CDF normal.
    Para `z = Z_95` isto devolve 0,95 de volta.
    """
    return math.erf(z / math.sqrt(2.0))


# A GUARDA
# Todo método que este módulo publica tem de estar declarado em
# `vocabulario.METODOS_INTERVALO`. Um método aqui e não lá é a mesma classe de defeito
# que `test_vocabulario.py` persegue.
#
# McNemar não aparece nesta lista: não é intervalo e não produz a quíntupla.

_METODOS_PRODUZIDOS = {"wilson", "bootstrap", "katz"}
_nao_declarados = _METODOS_PRODUZIDOS - set(V.METODOS_INTERVALO)
if _nao_declarados:
    raise AssertionError(
        f"método(s) produzido(s) por incerteza.py e não declarado(s) em "
        f"vocabulario.METODOS_INTERVALO: {sorted(_nao_declarados)}"
    )
del _nao_declarados
