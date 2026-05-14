output "project_id" {
  value       = var.project_id
  description = "Configured GCP project ID."
}

output "region" {
  value       = var.region
  description = "Configured GCP region."
}

output "artifact_repo" {
  value       = google_artifact_registry_repository.containers.repository_id
  description = "Artifact Registry repository name."
}

output "staging_bucket" {
  value       = google_storage_bucket.staging.name
  description = "Agent Engine staging bucket name."
}

output "github_workload_identity_provider" {
  value       = google_iam_workload_identity_pool_provider.github.name
  description = "Use this as GCP_WORKLOAD_IDENTITY_PROVIDER secret."
}

output "github_deployer_service_account" {
  value       = google_service_account.deployer.email
  description = "Use this as GCP_SERVICE_ACCOUNT secret."
}

output "backend_runtime_service_account" {
  value       = google_service_account.backend_runtime.email
  description = "Recommended service account for Cloud Run backend runtime."
}

output "ui_runtime_service_account" {
  value       = google_service_account.ui_runtime.email
  description = "Recommended service account for Cloud Run UI runtime."
}
