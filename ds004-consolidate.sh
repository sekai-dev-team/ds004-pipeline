#!/usr/bin/env bash
# DS-004 Consolidation cron script (v2.1)
# Runs the consolidation pipeline with 512MB memory limit,
# captures the JSON report, and pushes a summary to Discord.
#
# Usage:
#   DEEPSEEK_API_KEY=sk-... DISCORD_WEBHOOK=https://... ./ds004-consolidate.sh
#
# Environment:
#   DEEPSEEK_API_KEY   Required — DeepSeek API key
#   DISCORD_WEBHOOK    Optional — Discord webhook URL for notifications
#
# Dependencies: docker, curl, python3
set -u

IMAGE="kona01z/ds004-pipeline:latest"
DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-}"

echo "============================================"
echo " DS-004 Consolidation Round"
echo " $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo " Memory limit: 512MB"
echo "============================================"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set"
    exit 1
fi

# Run pipeline with 512MB memory limit.
# stdout = JSON report (single line), stderr = logs.
REPORT_JSON=$(docker run --rm \
    --memory=512m --memory-swap=512m \
    --network agent-net \
    -v vault_data:/vault \
    -e DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}" \
    -e DS004_VAULT_PATH=/vault \
    "${IMAGE}" \
    --mode consolidate 2>/dev/null)

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Docker run failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

if [ -z "$REPORT_JSON" ]; then
    echo "WARNING: No JSON report output from pipeline"
    exit 0
fi

echo ""
echo "--- JSON Report ---"
echo "$REPORT_JSON" | python3 -m json.tool --no-ensure-ascii 2>/dev/null || echo "$REPORT_JSON"
echo "---"

# Push summary to Discord via webhook
if [ -n "$DISCORD_WEBHOOK" ]; then
    echo ""
    echo "==> Pushing summary to Discord..."

    SUMMARY=$(python3 -c "
import json, sys

r = json.loads(sys.stdin.read())
s = r.get('summary', {})
m = r.get('memory', {})
t = r.get('token_usage', {})
cost = t.get('estimated_cost_usd', 0)
mem = m.get('peak_rss_mb', 0)

new_n = s.get('new_notes', 0)
ok_n = s.get('consolidated', 0)
skip_n = s.get('skipped', 0)
err_n = s.get('errors', 0)

summary_text = (
    f'**DS-004 Consolidation Report**\n'
    f'New: {new_n} | Consolidated: {ok_n} | Skipped: {skip_n} | Errors: {err_n}\n'
    f'Peak RSS: {mem}MB | Est. cost: \${cost:.4f}\n'
    f'Timestamp: {r.get(\"timestamp\", \"?\")}'
)

payload = json.dumps({'content': summary_text})
sys.stdout.write(payload)
" <<< "$REPORT_JSON")

    curl -s -H "Content-Type: application/json" \
        -d "$SUMMARY" \
        "$DISCORD_WEBHOOK" || echo "Warning: Discord push failed (exit $?)"

    echo "Discord notification sent."
fi

echo ""
echo "============================================"
echo " Consolidation round complete"
echo "============================================"
