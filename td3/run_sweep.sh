#!/usr/bin/env bash
# ── Launch 8 parallel wandb sweep agents ──
#
# Usage:
#   ./run_sweep.sh <sweep_id>            # 8 workers (default)
#   ./run_sweep.sh <sweep_id> <n>        # n workers
#
# Example:
#   wandb sweep sweep.yaml               # prints e.g. "acme/td3-bayes-sweep/abc123"
#   ./run_sweep.sh acme/td3-bayes-sweep/abc123

set -euo pipefail

SWEEP_ID="${1:?Usage: $0 <sweep_id> [num_workers]}"
NUM_WORKERS="${2:-8}"

echo "=== Sweep:   ${SWEEP_ID}"
echo "=== Workers: ${NUM_WORKERS}"
echo ""

# Cleanup on Ctrl+C
cleanup() {
    echo ""
    echo "=== Stopping all agents..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "=== Done."
}
trap cleanup EXIT INT TERM

# Launch agents in background
for ((i = 1; i <= NUM_WORKERS; i++)); do
    echo "[$(date +%T)] Starting agent ${i}/${NUM_WORKERS}..."
    wandb agent "${SWEEP_ID}" &
done

echo ""
echo "All ${NUM_WORKERS} agents running. Press Ctrl+C to stop."
wait
