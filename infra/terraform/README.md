# Terraform Bootstrap for GCP Deploy

This Terraform stack bootstraps the minimum GCP resources needed by `.github/workflows/deploy-gcp.yml`:

- required APIs
- Artifact Registry Docker repository
- staging bucket for Agent Engine packaging
- GitHub OIDC Workload Identity Pool + Provider
- deployer/runtime service accounts and IAM bindings

## Prerequisites

- Terraform >= 1.5
- `gcloud` authenticated to a project owner/admin account
- GitHub repository where Actions deployment workflow runs

## Apply

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars
terraform init
terraform apply
```

## Wire GitHub Secrets (secrets-only path)

After apply, capture outputs and set these repository secrets:

- `GCP_WORKLOAD_IDENTITY_PROVIDER` = `github_workload_identity_provider` output
- `GCP_SERVICE_ACCOUNT` = `github_deployer_service_account` output
- `GCP_PROJECT_ID` = project id
- `GCP_REGION` = region
- `GCP_ARTIFACT_REPO` = artifact repo id
- `GCP_STAGING_BUCKET` = staging bucket name

Optional (otherwise workflow defaults apply):

- `GCP_BACKEND_SERVICE`
- `GCP_UI_SERVICE`
- `MTG_BACKEND_MODE`
- `MTG_LLM_PROVIDER`
- `MTG_VERTEX_PROXY_RESEARCH`
- `MTG_VERTEX_PROXY_CHAT`
- `MTG_VERTEX_AGENT_ENGINE_RESOURCE`
- `MTG_OPENAI_MODEL`
- `MTG_VERTEX_MODEL`
- `MTG_LLM_TIMEOUT_SEC`
- `MTG_CHAT_MAX_CLARIFICATION_TURNS`
- `OPENAI_API_KEY` (only if testing OpenAI provider in deployed backend/Agent Engine)

The GitHub workflow now accepts values from `vars` or `secrets`; if variables are absent, secrets are used.

## Notes

- This stack intentionally focuses on deployment bootstrap. It does not create Cloud Run services directly.
- You can tighten IAM roles further after initial rollout.
