#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"

# Observação do conjunto breast_cancer conhecida como benigna.
curl --fail-with-body --silent --show-error \
  --request POST "${BASE_URL}/predict" \
  --header "Content-Type: application/json" \
  --data @- <<'JSON'
{
  "mean texture": 14.36,
  "mean perimeter": 87.46,
  "mean area": 566.3,
  "mean smoothness": 0.09779,
  "mean compactness": 0.08129,
  "mean concavity": 0.06664,
  "mean concave points": 0.04781,
  "mean symmetry": 0.1885,
  "mean fractal dimension": 0.05766,
  "radius error": 0.2699,
  "texture error": 0.7886,
  "perimeter error": 2.058,
  "area error": 23.56,
  "smoothness error": 0.008462,
  "compactness error": 0.0146,
  "concavity error": 0.02387,
  "concave points error": 0.01315,
  "symmetry error": 0.0198,
  "fractal dimension error": 0.0023,
  "worst radius": 15.11,
  "worst texture": 19.26,
  "worst perimeter": 99.7,
  "worst area": 711.2,
  "worst smoothness": 0.144,
  "worst compactness": 0.1773,
  "worst concavity": 0.239,
  "worst concave points": 0.1288,
  "worst symmetry": 0.2977,
  "worst fractal dimension": 0.07259,
  "faixa_raio": "medio"
}
JSON
echo
