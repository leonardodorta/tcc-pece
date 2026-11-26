"""
medicao — o gabarito automático e as métricas que ele permite afirmar.

    gabarito    pares (prosa, resposta) por redundância intradocumento — LIMPOS
    distante    pares por rótulo removido da prosa — MUITOS, com ruído
    incerteza   intervalo de Wilson, bootstrap e Katz

## Como o gabarito é construído

A fonte publica o mesmo dado duas vezes: as publicações do Painel de Negócios trazem
a parte contratada e o objeto no campo estruturado do formulário e na prosa do
despacho, escritos independentemente pelo publicador. Escondendo o formulário e
pedindo ao extrator que leia só a prosa, o campo escondido é a resposta certa — pares
(texto, resposta) sem nenhuma anotação manual.

Não é circular: o gabarito vem do campo estruturado, a entrada é a prosa. É legítimo
que a prosa contenha ela própria um rótulo, e é justamente isso que a estratificação
expõe em vez de esconder.

Três verificações independentes se somam, todas de custo humano zero:

- valor — a fonte escreve o número e o extenso na mesma frase; dois parsers
  independentes sobre a mesma evidência;
- documento fiscal — CNPJ e CPF carregam dígito verificador; um número lido da
  posição errada praticamente nunca satisfaz o módulo 11;
- concordância entre camadas — formulário e prosa são lidos por caminhos de código
  distintos, e a convergência não depende de gabarito externo algum.

## O teto de rotulagem

`modelo.rotulo.medir_ida_e_volta` mede quantos pares sobrevivem ao ciclo
texto → span → BIO → span → texto. O que não sobrevive não é erro do modelo: é limite
da rotulagem. Nenhuma métrica do modelo pode ser lida como maior que esse teto,
porque a própria referência usada para treinar tem esse teto.
"""
from __future__ import annotations

from . import distante, gabarito, incerteza

__all__ = ["distante", "gabarito", "incerteza"]
