#!/usr/bin/env bash
#
# deploy.sh — implanta a pilha raiz. EXIGE confirmacao explicita.
#
# Faz, em ordem:
#   1. valida tudo primeiro (chama validate.sh)
#   2. empacota (chama package.sh, que sobe wheel/templates/jobs)
#   3. mostra o comando de deploy e PARA, a menos que CONFIRMO=sim
#
# Variaveis:
#   ENV   (padrao dev)  so compoe o nome da pilha: STACK=domsp-$ENV. NAO define o parametro
#                       `Environment` do CloudFormation, que vem de parametros/producao.json
#                       (hoje `prod`) e determina o sufixo dos nomes dos recursos.
#   STACK, REGION (padrao us-east-1), ARTIFACT_BUCKET (obrigatoria), CONFIRMO.
#
# Uso:
#   ARTIFACT_BUCKET=meu-bucket bash infraestrutura/scripts/deploy.sh            # so mostra
#   ARTIFACT_BUCKET=meu-bucket CONFIRMO=sim bash infraestrutura/scripts/deploy.sh  # implanta
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA="$(cd "$AQUI/.." && pwd)"

ENV="${ENV:-dev}"
STACK="${STACK:-domsp-$ENV}"
REGION="${REGION:-us-east-1}"
PARAMS="$INFRA/parametros/producao.json"

if [ -z "${ARTIFACT_BUCKET:-}" ]; then
  echo "ERRO: defina ARTIFACT_BUCKET (bucket existente que guarda templates, jobs e wheel)."
  exit 2
fi
[ -f "$PARAMS" ] || { echo "ERRO: nao achei $PARAMS"; exit 2; }

echo "== validando =="
bash "$AQUI/validate.sh"

echo "== empacotando e enviando artefatos =="
ARTIFACT_BUCKET="$ARTIFACT_BUCKET" bash "$AQUI/package.sh"

# Le os parametros do JSON e monta o --parameter-overrides, injetando o ArtifactBucket.
OVERRIDES="$(python3 - "$PARAMS" "$ARTIFACT_BUCKET" <<'PY'
import json, sys
params = json.load(open(sys.argv[1]))
params.setdefault("ArtifactBucket", sys.argv[2])
params["ArtifactBucket"] = sys.argv[2]
print(" ".join(f"{k}={v}" for k, v in params.items()))
PY
)"

CMD=(aws cloudformation deploy
  --template-file "$INFRA/cloudformation/main.yaml"
  --stack-name "$STACK"
  --region "$REGION"
  --capabilities CAPABILITY_NAMED_IAM
  --parameter-overrides $OVERRIDES)

echo ""
echo "== comando de deploy =="
printf '  %q ' "${CMD[@]}"; echo

if [ "${CONFIRMO:-nao}" != "sim" ]; then
  echo ""
  echo "PARADO. Nada foi implantado. Para implantar de fato:"
  echo "  ARTIFACT_BUCKET=$ARTIFACT_BUCKET CONFIRMO=sim ENV=$ENV bash infraestrutura/scripts/deploy.sh"
  exit 0
fi

echo ""
echo "== implantando (CONFIRMO=sim) =="
"${CMD[@]}"
