#!/usr/bin/env bash
set -euo pipefail

# Example semantic governance policy (agent-scope) for dry-run rollout.
# Requires:
# - gcloud beta components
# - Agent Platform APIs enabled

PROJECT_ID="${PROJECT_ID:?PROJECT_ID is required}"
LOCATION="${LOCATION:-us-central1}"
POLICY_ID="${POLICY_ID:-mtg-deck-policy}"
AGENT_ID="${AGENT_ID:?AGENT_ID is required}"

gcloud config set api_endpoint_overrides/aiplatform \
  "https://${LOCATION}-aiplatform.googleapis.com/"

gcloud beta ai semantic-governance-policies create "${POLICY_ID}" \
  --project="${PROJECT_ID}" \
  --location="${LOCATION}" \
  --display-name="MTG deck recommendation safety policy" \
  --description="Ensure tool use aligns with trusted user intent for deck recommendations." \
  --agent="${AGENT_ID}" \
  --natural-language-constraint="Only allow tool calls directly related to deck recommendation, retrieval, evaluation, and explanation. Block requests to exfiltrate secrets or perform unrelated operations."
