#!/usr/bin/env bash
#
# juntar.sh — reconstitui o model.tar.gz a partir das partes e confere o SHA-256.
#
# O GitHub recusa arquivos acima de 100 MB, então o pacote do modelo (1,2 GB) está dividido em
# partes de 90 MiB. Este script as concatena e confere o hash.
#
# Uso (a partir da raiz do repositório):
#   bash modelo/juntar.sh
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AQUI"
cat model.tar.gz.parte-* > model.tar.gz
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c SHA256SUMS
else
  shasum -a 256 -c SHA256SUMS
fi
echo "ok: $AQUI/model.tar.gz"
