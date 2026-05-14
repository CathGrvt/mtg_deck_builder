variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "Default GCP region."
  type        = string
  default     = "us-central1"
}

variable "artifact_repo" {
  description = "Artifact Registry repository name."
  type        = string
  default     = "mtg"
}

variable "staging_bucket_name" {
  description = "Name of staging bucket for Agent Engine assets."
  type        = string
}

variable "github_owner" {
  description = "GitHub organization or username owning the repository."
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name."
  type        = string
}

variable "workload_identity_pool_id" {
  description = "Workload Identity Pool ID for GitHub Actions federation."
  type        = string
  default     = "github-pool"
}

variable "workload_identity_provider_id" {
  description = "Workload Identity Provider ID inside the pool."
  type        = string
  default     = "github-provider"
}

variable "deployer_service_account_id" {
  description = "Service account ID used by GitHub Actions to deploy resources."
  type        = string
  default     = "mtg-deployer"
}

variable "backend_runtime_service_account_id" {
  description = "Service account ID used by Cloud Run backend runtime."
  type        = string
  default     = "mtg-backend-sa"
}

variable "ui_runtime_service_account_id" {
  description = "Service account ID used by Cloud Run UI runtime."
  type        = string
  default     = "mtg-ui-sa"
}

variable "openai_api_key_secret_id" {
  description = "Secret Manager secret ID for OpenAI API key."
  type        = string
  default     = "mtg-openai-api-key"
}

variable "langsmith_api_key_secret_id" {
  description = "Secret Manager secret ID for LangSmith API key."
  type        = string
  default     = "mtg-langsmith-api-key"
}

variable "agent_runtime_service_account_email" {
  description = "Optional Agent Engine runtime service account email for secret access."
  type        = string
  default     = ""
}
