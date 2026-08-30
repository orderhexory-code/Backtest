#!/usr/bin/env bash
set -euo pipefail

echo "==> Running US100 M1 Scalper Backtest Engine"

CSV_FILE="${1:-}"

if [ -z "$CSV_FILE" ]; then
    if [ -d "data" ]; then
        FOUND_CSV=$(find data -maxdepth 3 -type f -name "*.csv" -size +1k | head -n 1 || true)
        if [ -n "$FOUND_CSV" ]; then
            CSV_FILE="$FOUND_CSV"
        fi
    fi
fi

if [ -z "$CSV_FILE" ] || [ ! -f "$CSV_FILE" ]; then
    echo "[ERROR] No valid CSV dataset found: $CSV_FILE"
    exit 1
fi

echo "[INFO] Running backtest with CSV: $CSV_FILE"

python3 scripts/backtest.py \
    --csv "$CSV_FILE" \
    --initial-balance 10000.0 \
    --risk-per-trade 100.0 \
    --risk-reward 2.0 \
    --spread 1.0 \
    --slippage 0.5 \
    --commission 0.0 \
    --enable-c2c \
    --c2c-trigger-r 1.0 \
    --output-dir results \
    --html-output backtest.html

# Copy to index.html for instant GitHub Pages compatibility
cp backtest.html index.html

test -s results/trades.csv || { echo "[ERROR] trades.csv missing"; exit 1; }
test -s results/statistics.json || { echo "[ERROR] statistics.json missing"; exit 1; }
test -s results/audit.csv || { echo "[ERROR] audit.csv missing"; exit 1; }
test -s backtest.html || { echo "[ERROR] backtest.html missing"; exit 1; }

echo "==> Verification completed successfully. Artifacts ready."
