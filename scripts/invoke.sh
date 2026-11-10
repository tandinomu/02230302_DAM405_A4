#!/usr/bin/env bash
# Invoke the deployed function three ways and save the evidence to docs/evidence/.
#
#   ./scripts/invoke.sh localstack            # LocalStack (direct invoke + function URL)
#   ./scripts/invoke.sh aws <PredictUrl>      # real AWS (direct invoke + API Gateway URL from `sam deploy` outputs)
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-localstack}"
FUNCTION="${LAMBDA_FUNCTION_NAME:-wine-inference-dev}"
OUT=docs/evidence
mkdir -p "$OUT"

if [[ "$TARGET" == "localstack" ]]; then
  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}" AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
  export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  EP=(--endpoint-url "${AWS_ENDPOINT_URL:-http://localhost:4566}")
  URL=$(aws "${EP[@]}" lambda get-function-url-config --function-name "$FUNCTION" --query FunctionUrl --output text)
  PREDICT_URL="${URL%/}/predict"
else
  EP=()
  PREDICT_URL="${2:?usage: invoke.sh aws <PredictUrl>}"
fi

echo "== 1. Direct invocation (aws lambda invoke) =="
aws "${EP[@]}" lambda invoke --function-name "$FUNCTION" \
  --cli-binary-format raw-in-base64-out --payload file://events/direct_invoke.json \
  --log-type Tail "$OUT/direct_invoke_response.json" \
  --query LogResult --output text | base64 -d > "$OUT/direct_invoke_logs.txt" || true
cat "$OUT/direct_invoke_response.json"; echo

echo "== 2. HTTP POST $PREDICT_URL =="
curl -sS -X POST "$PREDICT_URL" -H 'Content-Type: application/json' \
  -d @events/direct_invoke.json | tee "$OUT/http_predict_response.json"; echo

echo "== 3. Bad request (expect HTTP 400) =="
curl -sS -o "$OUT/http_bad_request_response.json" -w "HTTP %{http_code}\n" -X POST "$PREDICT_URL" \
  -H 'Content-Type: application/json' -d @events/bad_request.json
cat "$OUT/http_bad_request_response.json"; echo

echo "Evidence written to $OUT/"
