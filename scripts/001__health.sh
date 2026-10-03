#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

curl --fail-with-body --silent --show-error "${BASE_URL}/health"
echo
