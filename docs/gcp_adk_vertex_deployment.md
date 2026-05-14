# MTG Deck Builder on GCP: ADK + Vertex AI Agent Engine

This document describes the implemented deployment architecture for a hybrid production setup:

- Streamlit UI on Cloud Run (`mtg-ui`)
- Backend adapter on Cloud Run (`mtg-backend`) with request/response API contract
- ADK runtime for multi-agent orchestration on Vertex AI Agent Engine
- Managed RAG Engine-compatible retrieval path, with local hybrid fallback

## Implemented Runtime Components

`gcp_agent_runtime/` provides the deployable orchestration stack:

- `RootCoordinatorAgent`
- `QueryRewriteAgent`
- `RetrieverAgent` (+ `LocalHybridRetrieverClient`, `VertexRagRetrieverClient`)
- `RerankAgent`
- `CriticAgent`
- `DeckPlanAgent`
- `SafetyGateAgent`
- `CloudRunAgentAdapter`

### API Contract

Request (`DeckRecommendationRequest`):

- `session_id`
- `user_query`
- `format`
- `colors`
- `archetype_hint`
- `must_include`
- `must_exclude`
- `mode`

Response (`DeckRecommendationResponse`):

- `summary`
- `recommended_decklist`
- `key_claims`
- `citations`
- `confidence`
- `safety_verdict`
- `trace_id`
- `latency_ms`
- `model_used`

Additional backend-local endpoints:

- `POST /v1/research/run` for planner/retriever/critic/writer/validator runs
- `POST /v1/chat/respond` for retrieval-grounded chat responses

## Local Run (Hybrid Mode)

Create local env file:

```bash
cp .env.example .env
```

Start both services:

```bash
docker compose up --build
```

Services:

- UI: `http://localhost:8501`
- Backend adapter: `http://localhost:8080`

Compose uses backend endpoints inside the UI container:

- `http://mtg-backend:8080/v1/deck/recommend`
- `http://mtg-backend:8080/v1/research/run`
- `http://mtg-backend:8080/v1/chat/respond`

Use localhost variants only when Streamlit is running directly on your host.

In `Generate Deck` tab, enable **Use Deployed Agent Backend**.
`Agentic Research` and `Chatbot` tabs also have per-tab backend toggles and endpoint URL fields.

Backend mode selection is env-driven:

- `MTG_BACKEND_MODE=local` (default): run deck recommendations through local coordinator.
- `MTG_BACKEND_MODE=vertex`: proxy deck recommendations to Vertex Agent Engine.
- `MTG_VERTEX_FALLBACK_TO_LOCAL=true`: fallback to local coordinator if Vertex invocation fails.
- `MTG_VERTEX_PROXY_RESEARCH=true`: also proxy `/v1/research/run` through Agent Engine.
- `MTG_VERTEX_PROXY_CHAT=true`: also proxy `/v1/chat/respond` through Agent Engine.

If the proxy toggles above are `false`, research/chat run backend-local logic.

Chat supports clarification mode:

- `MTG_CHAT_ENABLE_CLARIFICATION=true`
- `MTG_CHAT_MAX_CLARIFICATION_TURNS=1`

To compare Gemini vs OpenAI for research/chat in proxied mode, set provider env vars for deployment (`MTG_LLM_PROVIDER`, `MTG_OPENAI_MODEL`, `MTG_VERTEX_MODEL`, optional `OPENAI_API_KEY`).

## Deploy Backend Adapter (Cloud Run)

Build/deploy backend adapter image:

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/mtg/mtg-backend:latest -f Dockerfile.backend .
gcloud run deploy mtg-backend \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/mtg/mtg-backend:latest \
  --region us-central1 \
  --allow-unauthenticated
```

Deploy UI separately with `Dockerfile.ui`.

## Deploy With GitHub Actions

This repo now includes a manual deployment workflow:

- `.github/workflows/deploy-gcp.yml`

It uses OIDC auth and supports:

- Cloud Run backend deploy
- Cloud Run UI deploy
- Optional Agent Engine deploy

Configure repository settings first.

For IaC bootstrap of these prerequisites, use Terraform in `infra/terraform/`.

The workflow supports secrets-only setup and forwards key runtime env vars to Agent Engine deployment so provider comparisons can be performed without local redeploy.

Repository variables:

- `GCP_PROJECT_ID`
- `GCP_REGION` (for example: `us-central1`)
- `GCP_ARTIFACT_REPO` (for example: `mtg`)
- `GCP_STAGING_BUCKET` (bucket name only, no `gs://`)
- `GCP_BACKEND_SERVICE` (default: `mtg-backend`)
- `GCP_UI_SERVICE` (default: `mtg-ui`)

Repository secrets:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

Then run:

1. GitHub → Actions → **Deploy GCP**
2. Click **Run workflow**
3. Choose toggles (`deploy_backend`, `deploy_ui`, `deploy_agent_engine`)

## Deploy ADK Agent to Agent Engine

Use:

```bash
python deploy_agent_engine.py \
  --project "$PROJECT_ID" \
  --location "us-central1" \
  --staging-bucket "gs://$STAGING_BUCKET" \
  --display-name "mtg-deck-builder-agent"
```

Dry run:

```bash
python deploy_agent_engine.py \
  --project "$PROJECT_ID" \
  --location "us-central1" \
  --staging-bucket "gs://$STAGING_BUCKET" \
  --dry-run
```

## Corpus Sync + Scheduler Entry Point

Export corpus and optionally upload:

```bash
python sync_rag_corpus.py \
  --cards data/commander_cards.csv \
  --decks current_commander_decks \
  --gcs-uri gs://$BUCKET/mtg-rag/
```

Use this command as Cloud Scheduler target through Cloud Run Jobs or Cloud Build triggers.

## Security and Governance

Implemented guardrail layers:

- Deterministic safety gate in runtime (`SafetyGateAgent`)
- Mode allowlist and prompt-injection/jailbreak pattern blocking
- Risk scoring for review path

Cloud controls to apply during deployment:

- Least-privilege service account for Agent Runtime
- Agent Gateway + semantic governance policies where available
- Model Armor templates in dry-run first, then enforce

## Evaluation and Release Gates

Primary (Vertex-style) release gate:

```bash
python eval/vertex_release_gate.py --results eval_runs/<timestamp>/results.jsonl
```

Secondary LangSmith OTEL fan-out env generation:

```bash
python eval/langsmith_fanout.py --api-key "$LANGSMITH_API_KEY" --project mtg-deck-builder
```

## Model Routing Policy

`ModelLifecycleGuard` implements:

- default `gemini-2.5-flash`
- escalation `gemini-2.5-pro` for high complexity or low confidence
- lifecycle fallback to flash when an escalation model is retired
- env-based EOL override support (`MODEL_EOL_<MODEL_ID>`)

## Test Coverage Added

- Query rewrite determinism/diversity
- Retrieval merge/rerank dedup
- Pro escalation + lifecycle fallback
- End-to-end coordinator flow including critic-triggered second pass
- Safety blocking behavior
- Adapter request/response schema conformance
- Offline release-gate threshold behavior
