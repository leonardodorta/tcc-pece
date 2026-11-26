#!/usr/bin/env bash
#
# validate.sh — valida a infraestrutura sem tocar na AWS.
#
# Roda, em ordem de custo crescente:
#   1. cfn-lint em todos os templates
#   2. compileall nos jobs Glue
#   3. o conferidor de argumentos (DefaultArguments == getResolvedOptions)
#   4. re-geracao do catalogo e diff contra o template versionado
#
# NAO implanta nada. Sai com codigo != 0 no primeiro passo que falhar.
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA="$(cd "$AQUI/.." && pwd)"
RAIZ="$(cd "$INFRA/.." && pwd)"

# PYTHON tem precedencia; depois o venv do projeto; depois o PATH.
PY="${PYTHON:-$RAIZ/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
LINT="${CFN_LINT:-$RAIZ/.venv/bin/cfn-lint}"
[ -x "$LINT" ] || LINT="cfn-lint"

echo "== 1/4 cfn-lint =="
"$LINT" "$INFRA"/cloudformation/*.yaml
echo "  ok"

echo "== 2/4 compileall dos jobs Glue =="
"$PY" -m compileall -q "$INFRA/jobs"
echo "  ok"

echo "== 3/4 contrato de argumentos =="
"$PY" "$AQUI/conferir_argumentos.py"

echo "== 4/4 catalogo em sincronia com a fonte unica de esquema =="
# Copia o 03-catalog.yaml atual para um diretorio temporario, regera o 03-catalog.yaml NO
# PROPRIO LUGAR (gerar_catalogo.py sobrescreve o arquivo) e compara a copia com o regerado.
# Em caso de divergencia o arquivo em cloudformation/ ja esta reescrito quando o script sai.
TEMP="$(mktemp -d)"
trap 'rm -rf "$TEMP"' EXIT
cp "$INFRA/cloudformation/03-catalog.yaml" "$TEMP/atual.yaml"
PYTHONPATH="$RAIZ/src" "$PY" "$AQUI/gerar_catalogo.py" >/dev/null
if ! diff -q "$TEMP/atual.yaml" "$INFRA/cloudformation/03-catalog.yaml" >/dev/null; then
  echo "  DIVERGENCIA: 03-catalog.yaml nao casa com esquema.py."
  echo "  Rode: PYTHONPATH=src python infraestrutura/scripts/gerar_catalogo.py"
  diff "$TEMP/atual.yaml" "$INFRA/cloudformation/03-catalog.yaml" || true
  exit 1
fi
echo "  ok"

echo ""
echo "validacao completa — nada implantado."
