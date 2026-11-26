#!/usr/bin/env python3
"""
Consolida as execuções de treino e aplica a regra de decisão declarada.

A regra: três execuções de treino com configuração idêntica; o veredicto vem da média, a
amplitude entre as execuções é registrada como medida da variação do treino, e o checkpoint
selecionado é a execução mais próxima da média no teste-resíduo (soma das distâncias em F1 e
precisão normalizados).

Os limiares do veredicto são frações do `teto_da_rotulagem` registrado em `avaliacao`: F1 mínimo
de 0,80 do teto e precisão mínima de 0,90 do teto (parâmetros `limiar_f1_fracao` e
`limiar_p_fracao` de `consolidar`). O teto vem do arquivo de avaliação, produzido fora deste
repositório.

## Uso

    python infraestrutura/scripts/consolidar_execucoes.py \\
        --execucao modelos/historico/run1/metricas_modelo.json \\
        --execucao modelos/historico/run2/metricas_modelo.json \\
        --execucao modelos/historico/run3/metricas_modelo.json

Cada `--execucao` é um `metricas_modelo.json` completo, com a chave `avaliacao`. A ordem é
a de execução, e é preservada na saída. A saída padrão é
`resultados/metricas_modelo_tres_execucoes.json` (`--saida`).

As pastas `modelos/historico/` e `resultados/` do exemplo não fazem parte do repositório: são
locais, geradas por quem executa o treino. A pasta de saída deve existir antes da execução.

## O que ele recusa

- menos de três execuções, porque a regra pede três;
- arquivo sem a chave `avaliacao`;
- a mesma execução repetida (mesmo nome de arquivo ou mesmo job de treino);
- execução sem `teto_da_rotulagem`, porque os limiares dependem dele;
- execuções com valores diferentes em: tamanho do conjunto de teste, tamanho do teste-resíduo,
  teto da rotulagem, número de exemplos de treino e hash do gabarito, porque a média de
  métricas medidas sobre insumos distintos não é comparável;
- execuções com hiperparâmetros diferentes (a semente está em `hiperparametros` e entra na
  comparação).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

#: As métricas sobre as quais a média e a amplitude são calculadas.
CAMPOS = (
    "val_f1",
    "teste_f1_exato",
    "teste_f1_norm",
    "residuo_f1_exato",
    "residuo_p_exato",
    "residuo_f1_norm",
    "residuo_p_norm",
    "residuo_r_norm",
)

#: Quantas execuções a regra exige.
EXIGIDAS = 3


def _extrair(caminho: Path, ordem: int) -> dict:
    """Uma execução, reduzida aos campos que a regra usa."""
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    avaliacao = dados.get("avaliacao") or {}
    if not avaliacao:
        raise ValueError(
            f"{caminho} não tem a chave `avaliacao`: é um arquivo de treino sem a "
            f"avaliação aplicada. Aplique a avaliação a este checkpoint antes"
        )
    teste, residuo = avaliacao["teste"], avaliacao["residuo"]
    return {
        "execucao": ordem,
        "arquivo": caminho.name,
        "job": dados.get("job_name", ""),
        "melhor_epoca": dados.get("melhor_epoca"),
        "exemplos_treino": (dados.get("corpus") or {}).get("exemplos_treino"),
        # Proveniência que amarra a execução ao insumo: semente, hiperparâmetros e hash do
        # gabarito. `_conferir_comparabilidade` recusa execuções que não compartilham estes.
        "semente": (dados.get("hiperparametros") or {}).get("semente"),
        "hiperparametros": dados.get("hiperparametros") or {},
        "hash_gabarito": (dados.get("gabarito") or {}).get("hash"),
        "teto_da_rotulagem": avaliacao.get("teto_da_rotulagem"),
        "n_teste": teste.get("n"),
        "n_residuo": residuo.get("n"),
        "val_f1": round(dados.get("melhor_entity_f1", 0.0), 4),
        "teste_f1_exato": round(teste["exato"]["f1"], 4),
        "teste_f1_norm": round(teste["normalizado"]["f1"], 4),
        "residuo_f1_exato": round(residuo["exato"]["f1"], 4),
        "residuo_p_exato": round(residuo["exato"]["precisao"], 4),
        "residuo_f1_norm": round(residuo["normalizado"]["f1"], 4),
        "residuo_p_norm": round(residuo["normalizado"]["precisao"], 4),
        "residuo_r_norm": round(residuo["normalizado"]["revocacao"], 4),
        "ic_precisao": residuo["normalizado"].get("ic_precisao"),
    }


def _conferir_comparabilidade(execucoes: list[dict]) -> None:
    """Recusa médias entre execuções que não medem a mesma coisa."""
    # Nenhuma execução repetida: mesmo arquivo ou mesmo job são a mesma execução contada
    # duas vezes, e a média de uma execução consigo mesma mascara a variância real.
    for campo, rotulo in (("arquivo", "arquivo"), ("job", "job de treino")):
        vistos = [e[campo] for e in execucoes if e.get(campo)]
        repetidos = {v for v in vistos if vistos.count(v) > 1}
        if repetidos:
            raise ValueError(
                f"execução repetida pelo {rotulo}: {sorted(repetidos)}. Três execuções "
                f"independentes exigem três {rotulo}s distintos — consolidar a mesma duas "
                f"vezes publica uma variância falsa"
            )
    # Teto ausente é erro, não zero: sem ele os limiares seriam zero.
    for e in execucoes:
        if not e.get("teto_da_rotulagem"):
            raise ValueError(
                f"execução {e['execucao']} ({e['arquivo']}) não tem teto_da_rotulagem. "
                f"Sem teto os limiares seriam zero e o veredicto perderia o sentido — "
                f"reavalie o checkpoint com um gabarito que meça o teto"
            )
    # Comparabilidade dos insumos: tamanhos, teto, exemplos de treino e hash do gabarito
    # devem ser os mesmos nas três execuções.
    for campo, rotulo in (("n_teste", "tamanho do conjunto de teste"),
                          ("n_residuo", "tamanho do teste-resíduo"),
                          ("teto_da_rotulagem", "teto da rotulagem"),
                          ("exemplos_treino", "número de exemplos de treino"),
                          ("hash_gabarito", "hash do gabarito")):
        valores = {e[campo] for e in execucoes if e.get(campo) is not None}
        if len(valores) > 1:
            raise ValueError(
                f"as execuções divergem no {rotulo}: {sorted(valores)}. A média de "
                f"métricas medidas sobre insumos diferentes não tem significado — "
                f"reavalie todas sobre o mesmo gabarito e configuração antes de consolidar"
            )
    # Hiperparâmetros (incluem a semente): dicts não são hasheáveis para o set acima;
    # compara em JSON canônico.
    chaves_hp = {json.dumps(e["hiperparametros"], sort_keys=True)
                 for e in execucoes if e.get("hiperparametros")}
    if len(chaves_hp) > 1:
        raise ValueError(
            "as execuções divergem nos hiperparâmetros. Três repetições do mesmo "
            "experimento têm de usar a mesma configuração, inclusive a semente, que consta "
            "em `hiperparametros` e é comparada aqui"
        )


def consolidar(caminhos: list[Path], limiar_f1_fracao: float = 0.80,
               limiar_p_fracao: float = 0.90) -> dict:
    """
    O registro consolidado, com a média, a amplitude e o checkpoint que a regra seleciona.

    Os limiares são `limiar_f1_fracao` e `limiar_p_fracao` multiplicados pelo teto da rotulagem.
    """
    if len(caminhos) < EXIGIDAS:
        raise ValueError(
            f"a regra declarada pede {EXIGIDAS} execuções e foram passadas "
            f"{len(caminhos)}. Publicar a média de menos que três seria publicar outra "
            f"regra que a declarada"
        )
    execucoes = [_extrair(c, i) for i, c in enumerate(caminhos, start=1)]
    _conferir_comparabilidade(execucoes)

    media = {c: round(statistics.fmean(e[c] for e in execucoes), 4) for c in CAMPOS}
    amplitude = {
        c: round(max(e[c] for e in execucoes) - min(e[c] for e in execucoes), 4)
        for c in CAMPOS
    }

    # `_conferir_comparabilidade` já garantiu que o teto existe e é o mesmo em todas.
    teto = execucoes[0]["teto_da_rotulagem"]
    limiar_f1 = round(limiar_f1_fracao * teto, 4)
    limiar_p = round(limiar_p_fracao * teto, 4)

    # Distância medida nas duas métricas que o critério cobra, somadas.
    distancia = {
        e["execucao"]: round(
            abs(e["residuo_f1_norm"] - media["residuo_f1_norm"])
            + abs(e["residuo_p_norm"] - media["residuo_p_norm"]), 4)
        for e in execucoes
    }
    # Desempate: havendo empate na distância à média, vence a menor ordem de execução —
    # determinístico e explícito.
    menor_distancia = min(distancia.values())
    empatadas = sorted(k for k, v in distancia.items() if v == menor_distancia)
    selecionada = empatadas[0]
    houve_empate = len(empatadas) > 1

    for e in execucoes:
        e["veredicto_isolado"] = (
            "aprova" if (e["residuo_f1_norm"] >= limiar_f1
                         and e["residuo_p_norm"] >= limiar_p) else "REPROVA"
        )

    aprovado = bool(media["residuo_f1_norm"] >= limiar_f1
                    and media["residuo_p_norm"] >= limiar_p)
    margem_p = round(media["residuo_p_norm"] - limiar_p, 4)

    return {
        "regra_de_decisao": {
            "declarada_em": "infraestrutura/scripts/consolidar_execucoes.py",
            "gerado_por": "infraestrutura/scripts/consolidar_execucoes.py",
            "execucoes_exigidas": EXIGIDAS,
            "veredicto_por": "media",
            "metrica_que_seleciona_o_checkpoint":
                "soma das distancias a media em F1 e precisao no teste-residuo",
            "desempate": "menor numero de execucao, quando ha empate na distancia",
            "houve_empate_no_desempate": houve_empate,
            "distancia_a_media": {f"execucao_{k}": v for k, v in distancia.items()},
            "checkpoint_publicado": f"execucao {selecionada}",
            "job_do_checkpoint_publicado":
                next(e["job"] for e in execucoes if e["execucao"] == selecionada),
            "teto_da_rotulagem": teto,
            "hash_gabarito": execucoes[0].get("hash_gabarito"),
            "semente_de_dados": execucoes[0].get("semente"),
            "limiares": {"f1_minimo": limiar_f1, "precisao_minima": limiar_p},
        },
        "execucoes": execucoes,
        "media": media,
        "amplitude": amplitude,
        "veredicto": {
            "aprovado": aprovado,
            "por": f"media de {len(execucoes)} execucoes",
            "execucoes_que_aprovam_isoladamente":
                sum(e["veredicto_isolado"] == "aprova" for e in execucoes),
            "margem_da_media_ao_limiar_de_precisao": margem_p,
            "amplitude_da_precisao": amplitude["residuo_p_norm"],
            "margem_menor_que_amplitude":
                abs(margem_p) < amplitude["residuo_p_norm"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execucao", action="append", type=Path, required=True,
                    help="um metricas_modelo.json por execução; repetir o argumento")
    ap.add_argument("--saida", type=Path,
                    default=Path("resultados/metricas_modelo_tres_execucoes.json"))
    args = ap.parse_args(argv)

    registro = consolidar(args.execucao)
    args.saida.write_text(
        json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    regra, vere = registro["regra_de_decisao"], registro["veredicto"]
    print(f"{len(registro['execucoes'])} execuções consolidadas")
    print(f"  {'run':>4} {'época':>6} {'val':>8} {'res F1':>8} {'res P':>8}  veredicto")
    for e in registro["execucoes"]:
        print(f"  {e['execucao']:>4} {e['melhor_epoca']:>6} {e['val_f1']:>8.4f} "
              f"{e['residuo_f1_norm']:>8.4f} {e['residuo_p_norm']:>8.4f}  "
              f"{e['veredicto_isolado']}")
    m, a = registro["media"], registro["amplitude"]
    print(f"  {'MÉDIA':>4} {'':>6} {m['val_f1']:>8.4f} {m['residuo_f1_norm']:>8.4f} "
          f"{m['residuo_p_norm']:>8.4f}  {'APROVA' if vere['aprovado'] else 'REPROVA'}")
    print(f"  {'ampl':>4} {'':>6} {a['val_f1']:>8.4f} {a['residuo_f1_norm']:>8.4f} "
          f"{a['residuo_p_norm']:>8.4f}")
    print(f"\n  limiares: F1 >= {regra['limiares']['f1_minimo']} "
          f"e precisão >= {regra['limiares']['precisao_minima']} "
          f"(teto {regra['teto_da_rotulagem']})")
    print(f"  distância à média: {regra['distancia_a_media']}")
    print(f"  checkpoint publicado: {regra['checkpoint_publicado']}")
    print(f"  margem da média ao limiar de precisão: "
          f"{vere['margem_da_media_ao_limiar_de_precisao']:+.4f}  "
          f"(amplitude {vere['amplitude_da_precisao']:.4f})")
    if vere["margem_menor_que_amplitude"]:
        print("  ATENÇÃO: a margem é menor que a amplitude entre execuções — "
              "o veredicto não é reproduzível execução a execução")
    print(f"\n  escrito {args.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
