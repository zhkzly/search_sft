#!/bin/bash

set -euo pipefail

# Canonical entrypoint for the current formal SFT experiment.
# This script intentionally wraps the existing compact-k2 trainer so users
# only need to remember one command.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

exec bash sft/scripts/train_qwen_0.5b_compact_k2.sh "$@"
