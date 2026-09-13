#!/usr/bin/env bash
# Regenerate the convex-inversion prior for one asteroid.
#
#   ./convinv/convexinitial.sh <ASTEROID_ID> [ORIGIN] [extra convexinitial.py args]
#   ./convinv/convexinitial.sh 3 simulated
#
# Results are written to <repo>/convex_inversions/. Existing files are kept
# unless --force is passed. Works from any working directory.
#
# The LSF headers below are only used when the script is submitted with
#   bsub < convinv/convexinitial.sh
# in which case the job-array index supplies the asteroid id. LSF copies the
# submitted script to a spool directory, so export CONVINV_REPO_ROOT=/path/to/lc2mesh
# before submitting (see the README) — the script cannot locate itself there.
#
#BSUB -q c47511
#BSUB -J "convexinitial[4-10]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R 'span[hosts=1]'
#BSUB -R "rusage[mem=5GB]"
#BSUB -B
#BSUB -N
#BSUB -o convexinitial_%I_%J.out
#BSUB -e convexinitial_%I_%J.err

set -euo pipefail

# --- Repository root: explicit override, else the location of this script ---
if [ -n "${CONVINV_REPO_ROOT:-}" ]; then
  REPO_ROOT="$(CDPATH= cd "$CONVINV_REPO_ROOT" && pwd -P)"
else
  SCRIPT_DIR="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
  REPO_ROOT="$(CDPATH= cd "$SCRIPT_DIR/.." && pwd -P)"
fi

if [ ! -f "$REPO_ROOT/convinv/convexinitial.py" ]; then
  echo "Error: could not locate the lc2mesh repository (looked in '$REPO_ROOT')." >&2
  echo "Set CONVINV_REPO_ROOT to the repository root and retry." >&2
  exit 1
fi

# --- Asteroid id: first argument, or the LSF job-array index ---
if [ "$#" -ge 1 ]; then
  ASTEROID_ID="$1"
  shift
elif [ -n "${LSB_JOBINDEX:-}" ]; then
  ASTEROID_ID="$LSB_JOBINDEX"
else
  echo "Usage: $0 <ASTEROID_ID> [ORIGIN] [extra convexinitial.py args]" >&2
  echo "Example: $0 3 simulated" >&2
  exit 1
fi

case "$ASTEROID_ID" in
  ''|*[!0-9]*) echo "Error: ASTEROID_ID must be an integer in 1-10, got '$ASTEROID_ID'" >&2; exit 1;;
esac
if [ "$ASTEROID_ID" -lt 1 ] || [ "$ASTEROID_ID" -gt 10 ]; then
  echo "Error: ASTEROID_ID must be in 1-10, got '$ASTEROID_ID'" >&2
  exit 1
fi

# --- Origin: second argument, default "simulated" (what the INR pipeline uses) ---
ORIGIN="simulated"
case "${1:-}" in
  simulated|real) ORIGIN="$1"; shift;;
esac

# --- Interpreter: $PYTHON, else the uv-managed .venv, else uv run, else python3 ---
cd "$REPO_ROOT"
if [ -n "${PYTHON:-}" ]; then
  RUNNER=("$PYTHON")
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  RUNNER=("$REPO_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run --project "$REPO_ROOT" python)
else
  RUNNER=(python3)
fi

# `convinv` is imported from the repository root; `src` is a fallback for when
# the lc2mesh package has not been installed with `pip install -e .` / `uv sync`.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Repository:  $REPO_ROOT"
echo "Interpreter: ${RUNNER[*]}"
echo "Asteroid:    $ASTEROID_ID (origin: $ORIGIN)"
echo "Output:      $REPO_ROOT/convex_inversions"

exec "${RUNNER[@]}" -m convinv.convexinitial "$ASTEROID_ID" "$ORIGIN" "$@"
