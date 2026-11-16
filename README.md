# GraphWalk POC API

A multi-agent, knowledge-augmented question answering (QA) system with **personal and general routes**, deployable as a
3-service architecture (Orchestrator / Personal / General) with full **LocalMode** support for development and testing.

> **Demo domain.** The repository ships with an invented domain: **Maslul**, a fictional public-transport operator
> (buses and a light-rail line). Riders ask about charges, refunds, passes, discounts, cards and fines.
> Every operator, rider, price, rule and API in this repository is made up. Questions and the knowledge base are in
> **Hebrew**; code, specs and documentation are in English. Swap the files under `Resources/` to model another domain.

## Features

### Dual Route Handling

- **General (KAG)**  
  Stateless general QA over a curated knowledge base (Weaviate + Gemini). Example: *"What discount does a student get?"*
- **Personal Route**  
  Authenticated, deterministic workflow execution using blocks, tools and actions against the rider's own data.
  Example: *"I was charged twice for the same trip."*
- **Mixed Queries (Decomposition)**  
  A query may be **split into atomic sub-queries**, each routed and executed independently:
    - Personal parts: sequential, interactive
    - General parts: parallel, non-interactive

### Architecture

- 3 FastAPI services: `orchestrator-service` (routing + high-level logic), `personal-service` (sessions + workflow
  execution), `general-service` (general QA)
- **LocalMode**: `python main.py` runs everything in-process, no Docker needed
- **Remote mode**: the orchestrator talks to the other services over HTTP

### Backend Technologies

- Google Vertex AI / Gemini (LLMs + embeddings)
- Non-LLM deterministic execution of workflow blocks
- Spec-driven: states, blocks, workflows, actions, tools and endpoints are JSON files under `Resources/Specs/`

## The demo domain at a glance

| Workflow | Topic | What it does |
|---|---|---|
| `wf 1` | Duplicate charge | rider picks a trip, checks for a double charge, then refund eligibility |
| `wf 2` | Trip refund | rider picks a trip, refund eligibility, existing refund requests |
| `wf 3` | Refund status | lists the rider's refund requests |
| `wf 4` | Discounts | rider category, discount eligibility, discount validity |
| `wf 5` | Pass renewal | rider picks a pass, renewal terms, card balance, balance after renewal |
| `wf 6` | Fine appeal | rider picks a fine, checks the appeal window |
| `wf 7` | Lost / blocked card | card status, replacement terms |
| `wf 8` | Card balance | card status and balance |

The personal API (`Resources/Specs/transit_endpoints.json`) is served locally by a small mock
(`Tools/MockApi/server.py`) with invented riders from `Resources/MockApi/riders.json`.

## Getting Started

### Prerequisites

- Python 3.10+
- A Google Cloud project with the Vertex AI API enabled and the Google Cloud SDK (`gcloud`) - needed for every LLM call
- (Optional) Docker Desktop; Weaviate + Postgres for the General route

### Installation

```bash
git clone <repository-url>
cd GraphWalkPOCAPI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Google Cloud Authentication

```bash
gcloud auth application-default login
```

Then copy `.env.example` to `.env` and set at least `GOOGLE_CLOUD_PROJECT` (and `VERTEXAI_LOCATION` if you do not use
`us-central1`). Application-default credentials are stored in `~/.config/gcloud/application_default_credentials.json`
(macOS / Linux) or `%APPDATA%\gcloud\application_default_credentials.json` (Windows).

## Running the System

### 0. Mock transit API (personal route)

```
python -m Tools.MockApi.server        # http://localhost:9000/mock-api  (bearer token: TRANSIT_API_TOKEN, default "dev-token")
```

`main.py` logs in as the demo rider `100000101`; see `Resources/MockApi/riders.json` for the other riders.

### 1. LocalMode (single process, ideal for development)

```
python main.py
```

This uses:

- In-process **SessionStore**
- `LocalSessionsAdapter`
- `LocalPersonalAdapter`
- `LocalGeneralAdapter`

### 2. Local Web Server (still single process)

```
SERVER_THREADPOOL_WORKERS=32 uvicorn Web.server:app --reload --port 8000 --limit-concurrency 50 --backlog 2048
```

Open:

```
http://127.0.0.1:8000/chat
```

### 3. Full 3‑Service Docker Setup

#### Build & Run

```
docker compose build
docker compose up
```

Services started:

- http://localhost:8000 → **Orchestrator**
- http://localhost:8001 → **Personal**
- http://localhost:8002 → **General**

#### GCP Auth Inside Docker

docker-compose mounts host ADC:

```

Note: `orchestrator-service` and `personal-service` must run with `--workers 1` due to in-memory session/auth state (already enforced in `docker-compose.yml`).

${HOME}/.config/gcloud → /root/.config/gcloud
```

Configures env:

```
GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
```

## Configuration

Main config file:

```
Resources/SystemConfigs/exp_1_create_dbs.json
```

Important sections:

```json
{
  "services": {
    "personal_base_url": "http://personal-service:8000",
    "general_base_url": "http://general-service:8000"
  }
}
```

Environment variable:

```
ORC_MODE=local   # default
ORC_MODE=remote  # Docker deployment
```

## Project Structure

```
GraphWalkPOCAPI/
├── Core/
│   ├── Agents/                  # Routing, critic, ranking, QA agents
│   ├── Clients/                 # HTTP clients + LocalMode adapters
│   ├── WorkflowSelection/       # Workflow selection methods (topic-based, unified), ANN, aggregator
│   ├── WFExecution/             # Deterministic executor
│   └── routing_orchestrator.py  # Core orchestration (decomposition + split execution + composition)
│
├── Web/
│   ├── server.py                # Orchestrator service
│   ├── personal_server.py       # Personal service API
│   └── general_server.py        # General service API
│
├── General/                     # General route: topic classifier, answer agent, Weaviate search
│
├── Resources/
│   ├── Specs/                   # transit_{states,blocks,workflows,actions,tools,endpoints}.json
│   ├── PersonalQuestions/       # curated Hebrew questions per workflow (CSV)
│   ├── KnowledgeBase/           # invented Hebrew knowledge base for the General route
│   ├── MockApi/                 # invented demo riders served by the mock API
│   ├── PromptConfigs/           # router/traversal/critic prompts
│   └── SystemConfigs/           # main config JSON
│
├── Tools/                       # SessionStore, transit API tools, MockApi, MockTools, option selection
├── tests/                       # scenarios + fixtures, offline checker, LLM-backed system tests
├── main.py                      # LocalMode runner
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Key Components

### Orchestrator

- Stateless in remote mode
- Runs Router decomposition, split-aware execution, and final composition
- Delegates personal sub-queries to the personal service and general sub-queries to the general service
- Manages `InputNeeded` resume via session state (`current_sub_id`)

### Personal Service

- Owns:
    - SessionStore
    - PersonalManager
    - Workflow execution
- Endpoints:
    - `/personal/handle`
    - `/sessions/*`
    - `/health`

### General Service

- Stateless general QA
- Endpoint:
    - `/general/ask`
    - `/health`

- **Aggregator**: Merges results by workflow while preserving provenance
- **Critic**: Validates and may request clarification on workflow selection
- **Executor**: Runs workflow blocks deterministically

## Testing

```bash
python tests/utils/check_specs.py     # offline: validates specs and walks every scenario against the fixtures (no LLM)
pytest tests/system                   # end to end in LocalMode; needs Vertex AI credentials (scenarios ending in _api need the mock API)
```

A scenario (`tests/fixtures/scenarios/<group>/<name>.json`) holds the first query, the scripted follow-up answers and the
mock API responses (`tests/fixtures/api_responses/...`). `check_specs.py --update` refreshes the expected facts.

## Development Notes

- Use **LocalMode** for all testing unless you explicitly want to test distributed behavior.
- Ensure **relative paths** only in configs → required for Docker.
- All services expose `/health` for readiness checks.

## Troubleshooting

### Vertex AI errors in Docker

Make sure:

```
gcloud auth application-default login
```

and docker-compose volumes mount your host ADC correctly.

### Personal route not resuming execution

Check:

- `pending_input` stored in SessionStore
- `/personal/handle` returns `pending=True`

---

## License

Add license information here.

All names, prices, rules and data in this repository are fictional and for demonstration only.