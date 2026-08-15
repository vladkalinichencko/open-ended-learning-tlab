#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

[ -d ext/jaxued ] || git clone --depth 1 https://github.com/DramaCow/jaxued.git ext/jaxued
[ -d .venv ] || python3 -m venv .venv

.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e ext/jaxued

# GPU: .venv/bin/pip install --upgrade "jax[cuda12]==0.4.30"
echo "ok: source .venv/bin/activate"
