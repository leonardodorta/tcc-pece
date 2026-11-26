"""
gerar_dicionario.py — escreve o dicionário de dados do lago a partir da fonte única de esquema.

O esquema de todas as tabelas é declarado em `contrato/esquema.py`. Deste único lugar derivam,
por geração, o esquema Python, o do Spark, as colunas do catálogo do Glue, o DDL do Athena e
este dicionário.

Uso:
    PYTHONPATH=src python infraestrutura/scripts/gerar_dicionario.py
    PYTHONPATH=src python infraestrutura/scripts/gerar_dicionario.py --conferir

`--conferir` não escreve: compara o arquivo existente com o que o esquema produz agora e
devolve 1 se divergirem, para servir de porta em verificação automatizada.

O destino é `resultados/dicionario-de-dados.md`. A pasta `resultados/` não faz parte do
repositório: é local e é criada na primeira geração.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

from domsp.contrato import esquema, lago

RAIZ = Path(__file__).resolve().parents[2]
DESTINO = RAIZ / "resultados" / "dicionario-de-dados.md"
# Nome do banco Glue/Athena escrito no DDL do dicionário: `domsp_<Environment>`, aqui com
# `Environment=prod`.
BANCO = "domsp_prod"


def main(argv: list[str] | None = None) -> int:
    argumentos = argv if argv is not None else sys.argv[1:]
    gerado = esquema.dicionario_completo(BANCO)

    if "--conferir" in argumentos:
        if not DESTINO.exists():
            print(f"FALHA: {DESTINO} não existe; rode sem --conferir para gerar")
            return 1
        atual = DESTINO.read_text(encoding="utf-8")
        if atual == gerado:
            print(f"ok: {DESTINO.name} bate com contrato/esquema.py "
                  f"({len(lago.CONJUNTOS)} tabelas)")
            return 0
        print(f"FALHA: {DESTINO.name} divergiu de contrato/esquema.py. "
              f"Rode `gerar_dicionario.py` para regerar.")
        for linha in difflib.unified_diff(
            atual.splitlines(), gerado.splitlines(),
            fromfile=f"{DESTINO.name} (atual)", tofile="gerado do esquema", lineterm="",
        ):
            print(f"  {linha}")
        return 1

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(gerado, encoding="utf-8")
    # `campos` devolve as colunas de dados; a coluna de partição é contada à parte nas
    # tabelas particionadas. O dicionário lista as duas.
    dados = sum(len(esquema.campos(c)) for c in lago.CONJUNTOS)
    particoes = sum(1 for c in lago.CONJUNTOS if c.particionado)
    print(f"escrito {DESTINO}")
    print(f"  {len(lago.CONJUNTOS)} tabelas, {dados} colunas de dados "
          f"+ {particoes} de partição = {dados + particoes}, "
          f"{len(gerado.splitlines())} linhas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
