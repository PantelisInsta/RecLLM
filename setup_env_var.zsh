#!/usr/bin/env zsh

# Set environment variables
export SNOWFLAKE_USER=PANTELISVAFIDIS
export SNOWFLAKE_PRIVATE_KEY=$(cat ~/.ssh/rsa_key.pub | grep -v "BEGIN PUBLIC KEY" | grep -v "END PUBLIC KEY" | sed 's/^[ \t]*//' | tr -d '\n')
export RPC_INSTACART_ML_FEATURE_STORE_V2_ADDR=https://rpc-dev-proxy-fs-online-ml.icprivate.com
export API_TYPE="open_ai"
export OPENAI_API_KEY=$(isc conf -e production api.query-understanding-service.search get OPENAPI_KEY)

# Print confirmation
echo "Environment activated and variables set:" >&2
echo "SNOWFLAKE_USER: $SNOWFLAKE_USER" >&2
echo "SNOWFLAKE_PRIVATE_KEY: ${SNOWFLAKE_PRIVATE_KEY:0:20}..." >&2
echo "RPC_INSTACART_ML_FEATURE_STORE_V2_ADDR: $RPC_INSTACART_ML_FEATURE_STORE_V2_ADDR" >&2