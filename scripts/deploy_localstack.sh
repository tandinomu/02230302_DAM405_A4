#!/usr/bin/env bash
# Deploy the inference Lambda to LocalStack with the AWS CLI (Infrastructure as Code, scripted).
# Idempotent: re-running updates the existing function instead of failing.
#
#   docker compose up -d
#   ./scripts/deploy_localstack.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"        # LocalStack accepts any dummy value
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ENDPOINT="${AWS_ENDPOINT_URL:-http://localhost:4566}"
FUNCTION="${LAMBDA_FUNCTION_NAME:-wine-inference-dev}"
BUCKET="${MODEL_BUCKET:-dam405-wine-models-dev}"
ROLE_ARN="arn:aws:iam::000000000000:role/lambda-wine-inference"
aws() { command aws --endpoint-url "$ENDPOINT" "$@"; }

echo ">> S3 bucket for model artefacts: $BUCKET"
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3 mb "s3://$BUCKET"
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled

echo ">> IAM role"
aws iam get-role --role-name lambda-wine-inference >/dev/null 2>&1 || \
  aws iam create-role --role-name lambda-wine-inference \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null

echo ">> Packaging src/inference"
mkdir -p build
rm -f build/inference.zip
(cd src/inference && zip -qr ../../build/inference.zip app.py model/)
ls -lh build/inference.zip

if aws lambda get-function --function-name "$FUNCTION" >/dev/null 2>&1; then
  echo ">> Updating function code: $FUNCTION"
  aws lambda update-function-code --function-name "$FUNCTION" --zip-file fileb://build/inference.zip >/dev/null
else
  echo ">> Creating function: $FUNCTION"
  aws lambda create-function --function-name "$FUNCTION" \
    --runtime python3.12 --handler app.lambda_handler \
    --role "$ROLE_ARN" --memory-size 256 --timeout 10 \
    --zip-file fileb://build/inference.zip \
    --environment "Variables={LOG_LEVEL=INFO,AWS_ENDPOINT_URL=http://localhost.localstack.cloud:4566}" >/dev/null
fi
aws lambda wait function-active-v2 --function-name "$FUNCTION"

echo ">> Function URL"
aws lambda get-function-url-config --function-name "$FUNCTION" >/dev/null 2>&1 || \
  aws lambda create-function-url-config --function-name "$FUNCTION" --auth-type NONE >/dev/null
URL=$(aws lambda get-function-url-config --function-name "$FUNCTION" --query FunctionUrl --output text)

echo
echo "Deployed. Function URL: $URL"
echo "For the Airflow deploy task:"
echo "  export DEPLOY_MODE=aws AWS_ENDPOINT_URL=$ENDPOINT MODEL_BUCKET=$BUCKET LAMBDA_FUNCTION_NAME=$FUNCTION"
