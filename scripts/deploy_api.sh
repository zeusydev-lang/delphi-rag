#!/usr/bin/env bash
# Deploy the API to Google Cloud Run. Secrets come from Secret Manager.
#
# One-time setup (see docs/OPERATIONS.md, Website):
#   gcloud config set project <project-id>
#   gcloud services enable run.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
#   scripts/deploy_api.sh --create-secrets     # from .env.production; values never printed
#
# Deploy:
#   scripts/deploy_api.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${REGION:-us-west1}"     # near Neon's us-west-2
SERVICE="delphi-api"
SECRETS=(ANTHROPIC_API_KEY VOYAGE_API_KEY QUERY_DATABASE_URL API_SHARED_SECRET)

env_value() {  # read one KEY from .env.production without printing it
    uv run --no-sync python -c "
import sys
from dotenv import dotenv_values
v = dotenv_values('.env.production').get(sys.argv[1])
sys.stdout.write(v or '')" "$1"
}

if [[ "${1:-}" == "--create-secrets" ]]; then
    for name in "${SECRETS[@]}"; do
        value="$(env_value "$name")"
        if [[ -z "$value" ]]; then echo "$name is missing from .env.production" >&2; exit 1; fi
        if gcloud secrets describe "$name" >/dev/null 2>&1; then
            printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- >/dev/null
            echo "  $name: new version"
        else
            printf '%s' "$value" | gcloud secrets create "$name" --data-file=- --replication-policy=automatic >/dev/null
            echo "  $name: created"
        fi
    done
    exit 0
fi

secret_flags=""
for name in "${SECRETS[@]}"; do secret_flags+="${name}=${name}:latest,"; done

gcloud run deploy "$SERVICE" \
    --source . \
    --region "$REGION" \
    --allow-unauthenticated \
    --min-instances 0 \
    --max-instances 1 \
    --concurrency 4 \
    --memory 512Mi \
    --timeout 60 \
    --set-env-vars "SITE_DAILY_CAP_USD=0.10,LLM_MODEL=anthropic/claude-haiku-5-5,EMBEDDING_MODEL=voyage/voyage-3.5" \
    --set-secrets "${secret_flags%,}"
