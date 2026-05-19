#!/usr/bin/env bash
set -euo pipefail

# Log-based dry-run policy verifier for Agent Gateway rollout.
#
# Usage (env vars):
#   PROJECT_ID=your-gcp-project \
#   INGRESS_ALLOW_FILTER='resource.type="..." AND ...' \
#   INGRESS_DENY_FILTER='resource.type="..." AND ...' \
#   EGRESS_ALLOW_FILTER='resource.type="..." AND ...' \
#   EGRESS_DENY_FILTER='resource.type="..." AND ...' \
#   ./infra/gcp/agent_gateway_policy_test.sh
#
# Optional tuning:
#   WINDOW_MINUTES=60
#   LOG_LIMIT=2000
#   MIN_INGRESS_ALLOW=1
#   MAX_INGRESS_DENY=0
#   MIN_EGRESS_ALLOW=1
#   MAX_EGRESS_DENY=0
#   SHOW_SAMPLES=true
#   SAMPLE_LIMIT=3

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "error: required command not found: ${cmd}" >&2
    exit 2
  fi
}

require_command gcloud

PROJECT_ID="${PROJECT_ID:-}"
WINDOW_MINUTES="${WINDOW_MINUTES:-60}"
LOG_LIMIT="${LOG_LIMIT:-2000}"
MIN_INGRESS_ALLOW="${MIN_INGRESS_ALLOW:-1}"
MAX_INGRESS_DENY="${MAX_INGRESS_DENY:-0}"
MIN_EGRESS_ALLOW="${MIN_EGRESS_ALLOW:-1}"
MAX_EGRESS_DENY="${MAX_EGRESS_DENY:-0}"
SHOW_SAMPLES="${SHOW_SAMPLES:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-3}"

INGRESS_ALLOW_FILTER="${INGRESS_ALLOW_FILTER:-}"
INGRESS_DENY_FILTER="${INGRESS_DENY_FILTER:-}"
EGRESS_ALLOW_FILTER="${EGRESS_ALLOW_FILTER:-}"
EGRESS_DENY_FILTER="${EGRESS_DENY_FILTER:-}"

if [[ -z "${PROJECT_ID}" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [[ -z "${PROJECT_ID}" ]]; then
  echo "error: PROJECT_ID is required (env var or active gcloud config)." >&2
  exit 2
fi

for name in INGRESS_ALLOW_FILTER INGRESS_DENY_FILTER EGRESS_ALLOW_FILTER EGRESS_DENY_FILTER; do
  if [[ -z "${!name}" ]]; then
    echo "error: ${name} must be set." >&2
    echo "hint: pass Cloud Logging advanced filters that isolate dry-run allow/deny events." >&2
    exit 2
  fi
done

if ! [[ "${WINDOW_MINUTES}" =~ ^[0-9]+$ ]] || ! [[ "${LOG_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "error: WINDOW_MINUTES and LOG_LIMIT must be integers." >&2
  exit 2
fi

if ! [[ "${MIN_INGRESS_ALLOW}" =~ ^[0-9]+$ ]] || ! [[ "${MAX_INGRESS_DENY}" =~ ^[0-9]+$ ]] || \
   ! [[ "${MIN_EGRESS_ALLOW}" =~ ^[0-9]+$ ]] || ! [[ "${MAX_EGRESS_DENY}" =~ ^[0-9]+$ ]]; then
  echo "error: threshold values must be integers." >&2
  exit 2
fi

count_logs() {
  local filter="$1"
  gcloud logging read "${filter}" \
    --project="${PROJECT_ID}" \
    --freshness="${WINDOW_MINUTES}m" \
    --limit="${LOG_LIMIT}" \
    --format="value(insertId)" | awk 'NF { count += 1 } END { print count + 0 }'
}

print_samples() {
  local label="$1"
  local filter="$2"
  if [[ "${SHOW_SAMPLES,,}" != "true" ]]; then
    return 0
  fi
  echo "----- sample logs: ${label} -----"
  gcloud logging read "${filter}" \
    --project="${PROJECT_ID}" \
    --freshness="${WINDOW_MINUTES}m" \
    --limit="${SAMPLE_LIMIT}" \
    --format="table(timestamp,logName,severity,textPayload,jsonPayload.message)"
}

check_metric() {
  local label="$1"
  local filter="$2"
  local min_expected="$3"
  local max_expected="$4"

  local observed
  observed="$(count_logs "${filter}")"
  echo "${label}: observed=${observed}, expected_min=${min_expected}, expected_max=${max_expected}"

  local ok=true
  if (( observed < min_expected )); then
    ok=false
    echo "  FAIL: observed count below minimum for ${label}" >&2
  fi
  if (( observed > max_expected )); then
    ok=false
    echo "  FAIL: observed count above maximum for ${label}" >&2
  fi

  print_samples "${label}" "${filter}"

  if [[ "${ok}" != "true" ]]; then
    return 1
  fi
  return 0
}

echo "Running Agent Gateway dry-run policy checks"
echo "project=${PROJECT_ID} window=${WINDOW_MINUTES}m limit=${LOG_LIMIT}"

failures=0

check_metric "ingress_allow" "${INGRESS_ALLOW_FILTER}" "${MIN_INGRESS_ALLOW}" "${LOG_LIMIT}" || failures=$((failures + 1))
check_metric "ingress_deny" "${INGRESS_DENY_FILTER}" 0 "${MAX_INGRESS_DENY}" || failures=$((failures + 1))
check_metric "egress_allow" "${EGRESS_ALLOW_FILTER}" "${MIN_EGRESS_ALLOW}" "${LOG_LIMIT}" || failures=$((failures + 1))
check_metric "egress_deny" "${EGRESS_DENY_FILTER}" 0 "${MAX_EGRESS_DENY}" || failures=$((failures + 1))

if (( failures > 0 )); then
  echo "dry-run policy checks FAILED (${failures} check groups failed)." >&2
  echo "Do not promote to ENFORCE until these checks pass with stable traffic." >&2
  exit 3
fi

echo "dry-run policy checks PASSED."
echo "You can consider ENFORCE rollout once this remains stable across multiple windows."
