# Terraform Bootstrap for GCP Deploy

This Terraform stack bootstraps the minimum GCP resources needed by `.github/workflows/deploy-gcp.yml`:

- required APIs
- Artifact Registry Docker repository
- staging bucket for Agent Engine packaging
- Secret Manager secrets for API keys
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
- `GCP_BACKEND_RUNTIME_SERVICE_ACCOUNT` (use `backend_runtime_service_account` output)
- `GCP_UI_RUNTIME_SERVICE_ACCOUNT` (use `ui_runtime_service_account` output)
- `MTG_BACKEND_MODE`
- `MTG_LLM_PROVIDER`
- `MTG_VERTEX_PROXY_RESEARCH`
- `MTG_VERTEX_PROXY_CHAT`
- `MTG_VERTEX_AGENT_ENGINE_RESOURCE`
- `MTG_OPENAI_API_KEY_SECRET_RESOURCE` (use `openai_api_key_secret_resource` output)
- `MTG_OPENAI_MODEL`
- `MTG_VERTEX_MODEL`
- `MTG_LLM_TIMEOUT_SEC`
- `MTG_CHAT_MAX_CLARIFICATION_TURNS`

The GitHub workflow now accepts values from `vars` or `secrets`; if variables are absent, secrets are used.

## Add Secret Versions (no secret values in Terraform state)

Create secret versions after `terraform apply`:

```bash
OPENAI_SECRET_ID="$(terraform output -raw openai_api_key_secret_id)"
printf '%s' "$OPENAI_API_KEY" | gcloud secrets versions add "${OPENAI_SECRET_ID}" --data-file=-
```

Optional LangSmith:

```bash
LANGSMITH_SECRET_RESOURCE="$(terraform output -raw langsmith_api_key_secret_resource)"
LANGSMITH_SECRET_ID="${LANGSMITH_SECRET_RESOURCE##*/}"
printf '%s' "$LANGSMITH_API_KEY" | gcloud secrets versions add "${LANGSMITH_SECRET_ID}" --data-file=-
```

If you deploy Agent Engine and want OpenAI provider there too, set `agent_runtime_service_account_email` in `terraform.tfvars` so Terraform grants `roles/secretmanager.secretAccessor` to that runtime identity.

## Notes

- This stack intentionally focuses on deployment bootstrap. It does not create Cloud Run services directly.
- You can tighten IAM roles further after initial rollout.
