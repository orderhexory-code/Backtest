#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " Starting US100 M1 Scalping Backtest Runner"
echo "============================================================"

CSV_FILE="${1:-}"

# Auto-discovery if CSV_FILE is not explicitly passed
if [ -z "$CSV_FILE" ]; then
    echo "[INFO] No CSV path argument supplied. Searching standard data directories..."
    if [ -d "data" ]; then
        # Find real CSV file (exclude empty or placeholder files)
        FOUND_CSV=$(find data -maxdepth 3 -type f -name "*.csv" -size +1k | head -n 1 || true)
        if [ -n "$FOUND_CSV" ]; then
            CSV_FILE="$FOUND_CSV"
        fi
    fi
fi

if [ -z "$CSV_FILE" ]; then
    echo "[ERROR] No valid CSV file located. Please specify path: ./scripts/run.sh <path_to_csv>"
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "[ERROR] Specified CSV file does not exist: $CSV_FILE"
    exit 1
fi

echo "[INFO] Target dataset: $CSV_FILE"
echo "[INFO] File size: $(ls -lh "$CSV_FILE" | awk '{print $5}')"

# Run python backtest engine
python3 scripts/backtest.py \
    --csv "$CSV_FILE" \
    --initial-balance 10000.0 \
    --risk-per-trade 100.0 \
    --risk-reward 2.0 \
    --spread 1.0 \
    --slippage 0.5 \
    --commission 0.0 \
    --output-dir results \
    --html-output backtest.html

# Verify output artifacts exist and are non-empty
echo "[INFO] Verifying generated artifacts..."

if [ ! -f "results/trades.csv" ]; then
    echo "[ERROR] results/trades.csv was not generated!"
    exit 1
fi

if [ ! -f "results/statistics.json" ]; then
    echo "[ERROR] results/statistics.json was not generated!"
    exit 1
fi

if [ ! -f "results/audit.csv" ]; then
    echo "[ERROR] results/audit.csv was not generated!"
    exit 1
fi

if [ ! -f "backtest.html" ]; then
    echo "[ERROR] backtest.html was not generated!"
    exit 1
fi

echo "============================================================"
echo " Backtest completed successfully. All artifacts validated."
echo "============================================================"
