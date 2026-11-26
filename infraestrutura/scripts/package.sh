#!/usr/bin/env bash
#
# package.sh — empacota o que o deploy sobe para o bucket de artefatos.
#
# NAO implanta CloudFormation. Faz duas coisas:
#   1. constroi o wheel do domsp (o mesmo pacote que os jobs Glue importam)
#   2. sincroniza cloudformation/, jobs/ e o wheel para s3://<artifact-bucket>/
#
# O upload para o S3 so acontece se ARTIFACT_BUCKET estiver definido. Sem ele, o script so
# constroi o wheel e imprime o que subiria.
#
# Uso:
#   ARTIFACT_BUCKET=meu-bucket bash infraestrutura/scripts/package.sh
#   bash infraestrutura/scripts/package.sh            # so constroi, nao sobe
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA="$(cd "$AQUI/.." && pwd)"
RAIZ="$(cd "$INFRA/.." && pwd)"

PY="${PYTHON:-$RAIZ/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

echo "== construindo o wheel do domsp =="
cd "$RAIZ"
"$PY" -m build --wheel --outdir "$INFRA/build/wheel"
WHEEL="$(ls -t "$INFRA/build/wheel"/domsp-*.whl | head -1)"
echo "  wheel: $WHEEL"

if [ -z "${ARTIFACT_BUCKET:-}" ]; then
  echo ""
  echo "ARTIFACT_BUCKET nao definido — nada foi enviado."
  echo "Subiria:"
  echo "  cloudformation/*.yaml  -> s3://<bucket>/cloudformation/"
  echo "  jobs/*.py              -> s3://<bucket>/jobs/"
  echo "  $(basename "$WHEEL")   -> s3://<bucket>/wheel/"
  exit 0
fi

echo "== sincronizando artefatos para s3://$ARTIFACT_BUCKET =="
# Cada sync exclui tudo e inclui um unico padrao: sobem apenas os templates `*.yaml` e os
# jobs `*.py`.
aws s3 sync "$INFRA/cloudformation" "s3://$ARTIFACT_BUCKET/cloudformation" \
  --exclude "*" --include "*.yaml"
aws s3 sync "$INFRA/jobs" "s3://$ARTIFACT_BUCKET/jobs" \
  --exclude "*" --include "*.py"
aws s3 cp "$WHEEL" "s3://$ARTIFACT_BUCKET/wheel/$(basename "$WHEEL")"
echo "  ok"
