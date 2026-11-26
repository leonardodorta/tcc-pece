"""
exportar_ocds.py — exporta uma amostra da base no padrão OCDS e valida contra o esquema oficial.

A validação cobre a base inteira; o arquivo publicado é apenas uma amostra (os primeiros
releases pela ordem determinística por identificador do ato). O `publishedDate` do pacote
registra a hora da geração, então o arquivo não é byte-idêntico entre execuções, mesmo que o
conjunto de releases e o conteúdo de cada um sejam.

O esquema está em `infraestrutura/esquemas/ocds-1.1.5/`, copiado sem modificação da Open
Contracting Partnership, com hash registrado.

Uso:
    PYTHONPATH=src python infraestrutura/scripts/exportar_ocds.py
    PYTHONPATH=src python infraestrutura/scripts/exportar_ocds.py --amostra 500

A entrada é um diretório de Parquet da prata consolidada (`contract-final`), lido com
`pyarrow` (dependência não listada em `pyproject.toml`; instale-a à parte). Sem `--entrada`, o
padrão é `dados/silver/contract-final`, pasta local que não faz parte do repositório (por
exemplo, uma cópia de `silver/contract-final/` do lago). A saída vai para `resultados/`
(`ocds-amostra.json` e `ocds-validacao.json`), pasta local também fora do repositório; o
script cria `resultados/` se ela não existir.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from domsp.contrato import ocds

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA_PADRAO = RAIZ / "dados" / "silver" / "contract-final"
DESTINO = RAIZ / "resultados" / "ocds-amostra.json"
RELATORIO = RAIZ / "resultados" / "ocds-validacao.json"
ESQUEMAS = RAIZ / "infraestrutura" / "esquemas" / "ocds-1.1.5"

# Quem publica. O OCDS exige o campo; sem ele o pacote não valida. O valor é gravado no
# JSON publicado.
PUBLICADOR = "Extração de contratos do Diário Oficial da Cidade de São Paulo (DOM-SP)"
# URI do pacote no OCDS. Placeholder (domínio `.invalid`): substitua pelo endereço real se o
# arquivo for publicado.
URI_PUBLICACAO = "https://exemplo.invalid/ocds/domsp-amostra.json"

# Quantos releases vao para o arquivo publicado. A validacao cobre a base inteira.
TAMANHO_DA_AMOSTRA = 200


def _ler_parquet(diretorio: Path) -> list[dict[str, Any]]:
    """Lê a prata consolidada de um diretório de Parquet."""
    import pyarrow.dataset as ds

    if not diretorio.exists():
        raise SystemExit(
            f"não encontrei {diretorio}. Gere a prata local antes, "
            f"ou aponte --entrada para outro diretório."
        )
    tabela = ds.dataset(str(diretorio), format="parquet", partitioning="hive").to_table()
    return tabela.to_pylist()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="exporta a base em OCDS 1.1 e valida o resultado")
    p.add_argument("--entrada", type=Path, default=ENTRADA_PADRAO,
                   help="diretório de Parquet da prata consolidada")
    p.add_argument("--amostra", type=int, default=TAMANHO_DA_AMOSTRA,
                   help="quantos releases vão para o arquivo publicado (0 = todos)")
    args = p.parse_args(argv)

    linhas = _ler_parquet(args.entrada)
    # Ordem determinística: o mesmo corpus produz a mesma amostra.
    linhas.sort(key=lambda r: str(r.get("contract_id") or ""))

    # Validação sobre a base inteira, não apenas sobre a amostra publicada.
    completo, omitidos = ocds.de_atos(
        linhas,
        uri=URI_PUBLICACAO,
        publicador=PUBLICADOR,
        publicado_em=dt.datetime.now(dt.timezone.utc),
    )
    erros = ocds.validar(completo, ESQUEMAS)

    # Arquivo publicado: amostra pela ordem determinística.
    publicado = dict(completo)
    if args.amostra > 0:
        publicado["releases"] = completo["releases"][: args.amostra]

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(publicado, ensure_ascii=False, indent=2), encoding="utf-8")

    relatorio = {
        "gerado_em": dt.datetime.now(dt.timezone.utc).isoformat(),
        "versao_ocds": ocds.VERSAO_OCDS,
        "esquema": "infraestrutura/esquemas/ocds-1.1.5 (cópia oficial, hash em PROCEDENCIA.txt)",
        "validacao": {
            "escopo": "base inteira",
            "atos_lidos": len(linhas),
            "releases_validados": len(completo["releases"]),
            "atos_omitidos": omitidos,
            "erros": len(erros),
            "detalhe_dos_erros": erros[:50],
        },
        "arquivo_publicado": {
            "caminho": str(DESTINO.relative_to(RAIZ)),
            "releases": len(publicado["releases"]),
            "observacao": "amostra da ordem determinística; a validação cobre a base inteira",
        },
        "mapeamento": {
            "campos_mapeados": len(ocds.MAPA_DE_CAMPOS),
            "campos_nao_mapeados": len(ocds.CAMPOS_NAO_MAPEADOS),
            "prefixo_ocid": ocds.PREFIXO_OCID,
            "prefixo_registrado": False,
        },
    }
    RELATORIO.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"atos lidos ............. {len(linhas)}")
    print(f"releases validados ..... {len(completo['releases'])}")
    print(f"omitidos ............... {omitidos}  (anulação, ou sem identificador/data)")
    print(f"balanço ................ {len(completo['releases'])} + {omitidos} = "
          f"{len(completo['releases']) + omitidos} de {len(linhas)}")
    print(f"erros de validação ..... {len(erros)}  (escopo: base inteira)")
    for e in erros[:10]:
        print(f"    {e}")
    print(f"\npublicado {DESTINO.relative_to(RAIZ)} com {len(publicado['releases'])} releases")
    print(f"relatório {RELATORIO.relative_to(RAIZ)}")

    # Código de saída não-zero quando a validação falha: o script serve de porta.
    return 1 if erros else 0


if __name__ == "__main__":
    raise SystemExit(main())
