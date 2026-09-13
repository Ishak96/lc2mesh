#!/usr/bin/env bash
# Launch an INR reconstruction run.
#
# Usage: ./run.sh <GPU_ID> <ASTEROID_ID> [extra run.py arguments...]
# Example: ./run.sh 0 3
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <ASTEROID_ID> [extra run.py arguments...]" >&2
  echo "Example: $0 0 3" >&2
  exit 1
fi

GPU_ID="$1"
ASTEROID_ID="$2"
shift 2

case "$GPU_ID" in
  ''|*[!0-9]*) echo "Error: GPU_ID must be a non-negative integer, got '$GPU_ID'" >&2; exit 1;;
esac
case "$ASTEROID_ID" in
  ''|*[!0-9]*) echo "Error: ASTEROID_ID must be an integer in 1-10, got '$ASTEROID_ID'" >&2; exit 1;;
esac
if [ "$ASTEROID_ID" -lt 1 ] || [ "$ASTEROID_ID" -gt 10 ]; then
  echo "Error: ASTEROID_ID must be in 1-10, got '$ASTEROID_ID'" >&2
  exit 1
fi

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd -P)"
PYTHON="${PYTHON:-python}"

# The GPU index is passed through to torch (cuda:<GPU_ID>); CUDA_VISIBLE_DEVICES
# is deliberately left untouched so the index means the same thing as nvidia-smi.
exec "$PYTHON" "$SCRIPT_DIR/run.py" --gpu "$GPU_ID" --asteroid "$ASTEROID_ID" "$@"
