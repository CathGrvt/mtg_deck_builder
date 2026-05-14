# GCP Infrastructure Notes (US / us-central1)

This folder contains operational templates for enterprise rollout.

Terraform bootstrap is available in `infra/terraform/` for setting up
deployment prerequisites (WIF, service accounts, Artifact Registry, staging bucket).

## Suggested IAM Principle

Use dedicated service accounts:

- `mtg-backend-sa` for Cloud Run backend adapter
- `mtg-agent-runtime-sa` for Agent Engine runtime identity
- `mtg-corpus-sync-sa` for scheduled corpus sync jobs

Grant least privilege only for required resources (Secret Manager, Cloud Storage, Agent Engine invocation, logging/trace write).

## Rollout Sequence

1. Deploy with policy controls in dry-run mode.
2. Observe violations/false positives.
3. Tune semantic governance constraints.
4. Switch enforcement to `ENFORCE`.

## CI/CD Deployment

Use the manual GitHub workflow:

- `.github/workflows/deploy-gcp.yml`

Required repository variables:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_ARTIFACT_REPO`
- `GCP_STAGING_BUCKET`
- `GCP_BACKEND_SERVICE`
- `GCP_UI_SERVICE`
- `GCP_BACKEND_RUNTIME_SERVICE_ACCOUNT` (recommended from Terraform output)
- `GCP_UI_RUNTIME_SERVICE_ACCOUNT` (recommended from Terraform output)
- `MTG_OPENAI_API_KEY_SECRET_RESOURCE` (Secret Manager resource path)

Required repository secrets:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

This workflow is `workflow_dispatch` only by default (manual trigger), so production deploys stay explicit and auditable.

The workflow also supports a secrets-only setup path (vars optional) after Terraform bootstrap.

## Policy Templates

- `sgp_policy_example.sh` for semantic governance policy creation
- `agent_gateway_policy_test.sh` for dry-run ingress/egress checks
