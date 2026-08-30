# SchemaRAG

Schema-grounded, secure Natural Language → SQL for relational databases.

SchemaRAG is a schema-grounded RAG architecture that accepts natural-language questions about a relational database, retrieves relevant schema and metadata as grounding context, uses an LLM to generate SQL, performs deterministic parsing and schema grounding of the generated SQL, applies AST-level security validation, executes approved queries through a least-privilege read-only database identity, processes results into typed JSON, and presents answers in a web interface. The college management database included in this repository is a demonstration/simulation environment used to exercise and evaluate the SchemaRAG architecture — it is not the product domain. The architecture is schema-oriented and can be adapted to other relational domains by regenerating schema metadata and RAG artifacts.

Contents
- Project overview
- Motivation
- Core idea (pipeline)
- College demonstration database
- Features (implemented)
- System architecture (diagram + file mappings)
- End-to-end request flow
- RAG architecture (document & retrieval model)
- Knowledge representation & artifact generation
- Text→SQL pipeline (generation, parsing, grounding)
- LLM providers
- SQL grounding & security
- Database execution and least-privilege design
- Backend architecture (file map)
- Frontend architecture (file map)
- API reference (endpoints implemented)
- Backend ↔ Frontend contract (response fields)
- Installation & exact run commands
- Docker
- Testing
- Security & secret management
- Limitations
- Design decisions
- Extensibility & generalization
- Project status & phases
- License

## 1. Project overview

### What SchemaRAG is

- SchemaRAG is a reference implementation that turns natural-language database questions into validated, schema-grounded SQL that is executed safely under a read-only database identity. The system integrates retrieval-augmented generation (RAG) of schema metadata, local embeddings + FAISS vector search, provider-backed LLM text generation, deterministic SQL parsing (sqlglot), schema grounding, an AST-based security validator, and an execution layer that enforces statement timeouts and row limits.

### What problem it solves

- Natural-language interfaces to relational databases are powerful but unsafe when LLMs are relied on without schema knowledge or deterministic validation. SchemaRAG reduces hallucination, enforces schema correctness, rejects unsafe statements, and confines execution to a low-privilege role.

### Why this architecture

- RAG supplies focused schema context so the LLM doesn't invent tables/columns or miss join backbones.
- Deterministic parsing and grounding ensure the SQL refers only to known database objects.
- AST-level security validation blocks DML/DDL and other dangerous constructs.
- A read-only execution identity enforces least privilege at the database level.
- Generation and execution are strictly separated so SQL is never executed without explicit validation.

## 2. Motivation

Common failure modes for direct LLM→SQL approaches (addressed by SchemaRAG):

- Schema hallucination: LLM invents tables/columns not present in the DB.
- Wrong joins or missing join conditions.
- Unsafe statements (INSERT/UPDATE/DELETE/DDL) produced by the LLM.
- Missing schema context: long schemas make prompting impractical.
- Excessive privileges: running generated SQL as an admin is dangerous.
- Non-deterministic behavior and lack of explainability.

SchemaRAG mitigates these with schema extraction, embedding-based retrieval, guaranteed coverage of schema & relationships, deterministic SQL parsing (sqlglot), grounding checks, AST-level security rules, and read-only execution.

## 3. Core idea (pipeline)

Database Schema
  ↓ (extractor)
Schema/Metadata Knowledge (documents)
  ↓ (embeddings)
Embeddings (sentence-transformers)
  ↓ (FAISS index)
Vector Retrieval (type-stratified retriever)
  ↓
Relevant Grounding Context (assembled prompt)
  ↓
LLM provider (Gemini / Ollama)
  ↓
Generated SQL (text)
  ↓
Parsing (sqlglot-backed validation)
  ↓
Schema Grounding (confirm tables/columns/relationships)
  ↓
Security Validation (AST-level checks)
  ↓
Read-only Execution (dedicated DB role)
  ↓
Result Processing (JSON-safe, typed)
  ↓
Frontend (React/TypeScript)

Each stage is implemented in the backend under `backend/app/rag` and `backend/app/services`. Exact module locations are listed throughout this README.

## 4. Important clarification: college demonstration database

- The college management database included in this repository is a demonstration/simulation environment for exercising the SchemaRAG pipeline end-to-end. It provides a realistic relational schema with multiple related tables so multi-table joins, aggregations, filters, and business-rule scenarios can be tested.
- SchemaRAG is schema-driven: the extractor converts a target database's schema and metadata into knowledge documents (tables, relationships, constraints, business rules, and curated query examples). To adopt SchemaRAG for another domain, the same extractor/indexing pipeline is used to regenerate the knowledge artifacts from the new schema.

## 5. Features (implemented)

Documented features (implemented in code):

### Natural Language Querying

- Accept natural-language questions and produce generated SQL (generation-only endpoint).
- Accept natural-language questions and produce executed read-only results (generation + execution endpoint when execution is enabled).

### Schema-Grounded RAG

- Extraction of schema metadata into knowledge documents (per-table schema documents, per-foreign-key relationship documents, per-constraint documents, curated business-rule documents, and curated query-example documents).
  - Implementation: `backend/app/rag/extractor.py` and `backend/app/rag/knowledge.py`
- Deterministic ordering and manifest/freshness checks for reproducible artifact generation.

### Embeddings & Vector Store

- Local sentence-transformers embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`).
  - Implementation: `backend/app/rag/embeddings.py`
- FAISS flat inner-product (`IndexFlatIP`) index for retrieval with on-disk index + manifest.
  - Implementation: `backend/app/rag/vector_store.py`

### Retrieval

- Type-stratified retrieval that ensures coverage of schema and relationship documents (guarantees that schema backbone is not crowded out by similar query examples).
  - Implementation: `backend/app/rag/retriever.py` (`TYPE_FLOORS` and `_select_with_type_cover`)

### Text-to-SQL

- Context assembly with retrieved knowledge and prompt templates.
  - Implementation: `backend/app/rag/context.py`, `backend/app/rag/prompts.py`
- Provider abstraction for LLMs; implementations for Google Gemini and Ollama are present.
  - Implementation: `backend/app/rag/llm/base.py`, `gemini.py`, `ollama.py`
- SQL extraction and generation orchestration (generation-only path).
  - Implementation: `backend/app/rag/text_to_sql.py`

### SQL Grounding

- AST parsing and validation of single SELECT statements with clearly reported issues (parsing, insufficient context, or invalid responses).
  - Implementation: `backend/app/rag/sql_parsing.py`, `grounding.py`, `validation.py`

### SQL Security

- AST-level checks implemented to reject multi-statement SQL, DDL, DML, `SELECT INTO`, `FOR UPDATE`, dangerous system functions, and other disallowed constructs.
  - Implementation: `backend/app/rag/sql_security.py`

### Secure Execution

- Read-only execution support: the system executes queries over a dedicated `exec_db_user` when configured. Server-side statement timeout and maximum row limits are enforced by configuration settings.
  - Implementation: `backend/app/db/session.py`, `backend/app/services/sql_execution.py` (service wiring is in `backend/app/api/routes/query.py`)

### Result Processing

- Typed JSON results, JSON-safe serialization, and response models for generated SQL and executed query results.
  - Implementation: `backend/app/rag/text_to_sql.py` (GeneratedSQL model) and `backend/app/schemas/`

### Frontend

- React + TypeScript UI (Vite), components to enter questions, display results, and handle errors.
  - Implementation: `frontend/` (source tree)

## 6. System architecture (diagram + component mapping)

ASCII diagram (matches code structure)

```
User (browser)
  |
  v
React / TypeScript frontend (frontend/)
  |
  v
FastAPI API (backend/app/main.py)
  |
  v
Query Preprocessing (backend/app/rag/preprocessing.py)
  |
  v
Schema RAG Retrieval (backend/app/rag/retriever.py, vector_store.py, embeddings.py)
  |
  v
Context Assembly (backend/app/rag/context.py, prompts.py)
  |
  v
LLM Provider (backend/app/rag/llm/*)
  |
  v
SQL Extraction & Parsing (backend/app/rag/text_to_sql.py, sql_parsing.py)
  |
  v
Schema Grounding (backend/app/rag/grounding.py)
  |
  v
SQL Security Validation (backend/app/rag/sql_security.py, validation.py)
  |
  v
Read-only Database Execution (backend/app/services/sql_execution.py, backend/app/db/session.py)
  |
  v
PostgreSQL
  |
  v
Result Processing (backend/app/services / route response models)
  |
  v
JSON Response to frontend (backend/app/api/routes/*)
```

Primary file mappings (high level):
- API & routes: `backend/app/api/routes/generate_sql.py`, `backend/app/api/routes/query.py`, `backend/app/api/routes/health.py`
- App entry: `backend/app/main.py`
- Config: `backend/app/core/config.py`
- RAG & retrieval: `backend/app/rag/`
- DB sessions: `backend/app/db/`
- Execution services: `backend/app/services/`
- Frontend: `frontend/`

## 7. End-to-end request flow (numbered)

1. User enters a natural-language question in the browser UI.
2. Frontend sends a JSON POST to the backend API.
   - Generation-only: `POST /api/generate-sql`
   - Generation + execution: `POST /api/query`
3. Backend preprocesses the question (`backend/app/rag/preprocessing.py`).
4. Retriever encodes the processed question using the local sentence-transformer model and performs FAISS search (`backend/app/rag/embeddings.py` + `vector_store.py`).
5. Retriever applies type-stratified selection to guarantee inclusion of schema and relationship documents (`backend/app/rag/retriever.py`).
6. ContextAssembler (`backend/app/rag/context.py`) builds a prompt with the retrieved documents and the processed question.
7. Backend invokes the configured LLM provider via the LLM provider abstraction (`backend/app/rag/llm/*`).
8. The provider returns text; the text is validated and parsed into a single SELECT statement (`backend/app/rag/sql_parsing.py`).
9. The parsed SQL is grounded against the extracted schema snapshot (`backend/app/rag/grounding.py`).
10. If execution is requested, the SQL is passed to the security validator (`backend/app/rag/sql_security.py`).
11. If validation passes and exec credentials are configured, the SQL is executed under the configured read-only execution identity with statement timeout and row limits applied (`backend/app/services/sql_execution.py` and `backend/app/db/session.py`).
12. Results are processed into typed, JSON-safe values and returned to the frontend.
13. Frontend renders the result table and any diagnostic metadata (retrieved docs, grounding status, security notes).

## 8. RAG architecture (how retrieval works)

### Knowledge document types
- `schema`: one document per table with columns and descriptions
- `relationship`: one document per discovered foreign key (many-to-one, etc.)
- `constraint`: CHECK / UNIQUE constraint documents
- `business_rule`: curated domain rules declared in code (`BUSINESS_RULES`)
- `query_example`: curated NL→SQL examples used as worked examples

### Embedding model & vector store
- Embedding model: local sentence-transformers model (default setting: `sentence-transformers/all-MiniLM-L6-v2`)
  - Implementation: `backend/app/rag/embeddings.py`
- Vector engine: FAISS `IndexFlatIP` over L2-normalized vectors
  - Implementation: `backend/app/rag/vector_store.py`
- On-disk artifacts: `index.faiss` and `document_store.json` are persisted; a manifest records the embedding model, index dimension, and the SHA-256 of the `knowledge.jsonl` that vectors were built from. The loader enforces these expectations to avoid silent stale-index usage.

### Retrieval strategy and guaranteed coverage
- The retriever performs an encode + search over the entire corpus, then applies a type-stratified selection algorithm. `TYPE_FLOORS` in `backend/app/rag/retriever.py` ensures the schema and relationship backbone is always present in the assembled prompt. This prevents clusters of similar query examples from crowding out essential schema documents and ensures multi-table questions receive the schema context needed for grounding.

### Retrieval configuration
- Default `top_k`: `settings.rag_top_k`
- Max context size: `settings.max_context_chars`
- Embedding batch size, `rag_index_dir`, and `rag_output_dir` are configurable through settings (`backend/app/core/config.py`).

## 9. Knowledge representation & artifact generation

### Extraction and knowledge artifacts
- The extractor reads the live PostgreSQL catalog (or a saved metadata snapshot) and creates a `SchemaMetadata` JSON snapshot.
  - Extraction command referenced in code: `python -m app.scripts.extract_metadata`
  - The snapshot path: `rag/metadata/schema_metadata.json` (`settings.rag_output_dir / "metadata" / "schema_metadata.json")`
- `KnowledgeGenerator` converts `SchemaMetadata` into deterministic `KnowledgeDocument` objects of types `schema`, `relationship`, `constraint`, `business_rule`, and `query_example`.
  - Implementation: `backend/app/rag/knowledge.py`
- The build process produces: `knowledge.jsonl` (documents), a FAISS index at `rag/index/index.faiss`, and `document_store.json` with a manifest. The loader enforces the SHA-256 of `knowledge.jsonl` to prevent stale indices.

## 10. Text → SQL pipeline (technical)

### Pipeline steps (file-level mapping)
- Preprocessing: `backend/app/rag/preprocessing.py`
- Retrieval: `backend/app/rag/retriever.py`
- Context assembly + prompts: `backend/app/rag/context.py` and `backend/app/rag/prompts.py`
- Provider invocation: `backend/app/rag/llm/*` (`create_provider` / `provider.generate`)
- SQL extraction & parsing: `backend/app/rag/text_to_sql.py` and `backend/app/rag/sql_parsing.py`
- Grounding: `backend/app/rag/grounding.py`
- `GeneratedSQL` model: `backend/app/rag/text_to_sql.py` (fields: `question`, `processed_question`, `sql`, `model`, `grounded`, `retrieved_documents`, `retrieval_scores`, `issues`, `error`)

### Principle
- The LLM is a generator only. Generated SQL is parsed and grounded deterministically before any execution; generation and execution are separated.

## 11. LLM providers (what is implemented)

Provider abstraction and implementations:
- Provider abstraction: `backend/app/rag/llm/base.py`
- Gemini provider: `backend/app/rag/llm/gemini.py` (uses `google-genai`)
- Ollama provider: `backend/app/rag/llm/ollama.py` (local Ollama HTTP-based provider)
- Default provider and model are configurable via environment variables:
  - `llm_provider` (default `"gemini"`)
  - `gemini_api_key`
  - `gemini_model` (default `"gemini-3.6-flash"`)
  - `ollama_base_url`
  - `ollama_model`

### Error handling
- The route handlers map provider unavailability and generation errors to appropriate HTTP statuses (503 for provider unavailable, 502 for other LLM errors). See `backend/app/api/routes/generate_sql.py` and `query.py` for mappings.

## 12. SQL grounding (how generated SQL is validated)

### Grounding process (implemented)
- After parsing, the parsed AST (single SELECT) is checked against the schema snapshot (`SchemaMetadata`) for:
  - existence of referenced tables
  - existence of referenced columns
  - valid join paths (relationship documents)
  - violations are reported as grounding issues and the `grounded` flag is set accordingly
  - The system does not silently repair hallucinated references; it returns errors and issues for inspection.
  - Implementation: `backend/app/rag/grounding.py` and `backend/app/rag/validation.py`

### Example (conceptual)
- A generated SQL referencing a table `students` and column `score` must match an extracted schema document for `students` and include `score` as a known column; otherwise grounding fails and the returned `GeneratedSQL` indicates `grounded=false` and includes issues.

## 13. SQL security (defense-in-depth)

Implemented protections:
- Enforced single-statement SELECT check (multi-statement rejected).
- AST-level rejection of DDL/DML statements (INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, etc.).
- Disallow `SELECT INTO`, data-modifying CTEs, `FOR UPDATE`, or other write-intent constructs.
- Disallow dangerous or privileged PostgreSQL functions and system catalog manipulations.
- Rejection of statements that attempt to access system schemas or perform administrative operations.
- Grounding before execution: SQL must be grounded and pass validator checks before being sent to the database.
- Implementation: `backend/app/rag/sql_security.py` and `backend/app/rag/sql_parsing.py` plus validation pipelines.

## 14. Database security / least privilege

### Execution design (configurable)
- Execution is disabled by default unless `exec_db_user` and `exec_db_password` are set in the environment; generation-only endpoints keep working with execution disabled.
  - Settings keys: `exec_db_user`, `exec_db_password` (`backend/app/core/config.py`)
- Statement timeout and maximum rows are enforced via settings:
  - `sql_statement_timeout_ms`
  - `sql_max_rows`
- Read-only execution user: queries are executed under a dedicated low-privilege read-only account when configured; administrative credentials are not used for executing generated SQL.
- Implementation: `backend/app/db/session.py` and `backend/app/services/sql_execution.py` (execution service wiring and enforcement)

### Why this is important
- Defense-in-depth prevents accidental or malicious data alterations and limits blast radius even if a logic bug allows an unsanitised query through.

## 15. Database schema (demonstration)

- The repository provides an extractor and curated query examples that operate over the demo college database. The extractor writes a canonical metadata snapshot at `rag/metadata/schema_metadata.json`.
- The knowledge generation step produces one schema document per table and relationship documents for foreign keys. These artifacts are the canonical representation of the demo schema used by retrieval and grounding.

> For the canonical, authoritative table/column/constraint listing, consult the generated metadata snapshot `rag/metadata/schema_metadata.json` produced by running the extractor script; that snapshot is the single source of truth for table definitions, primary keys, foreign keys, comments and extracted constraint expressions.

## 16. Backend architecture (file tree excerpt)

```
backend/
├─ app/
│  ├─ main.py                       # FastAPI app entry
│  ├─ api/
│  │  └─ routes/
│  │     ├─ generate_sql.py         # POST /api/generate-sql (generation-only)
│  │     ├─ query.py                # POST /api/query (generation + execution)
│  │     └─ health.py               # health endpoint
│  ├─ core/
│  │  └─ config.py                  # settings + .env handling
│  ├��� db/
│  │  ├─ base.py
│  │  └─ session.py                 # DB session management
│  ├─ rag/
│  │  ├─ embeddings.py
│  │  ├─ extractor.py
│  │  ├─ knowledge.py
│  │  ├─ vector_store.py
│  │  ├─ retriever.py
│  │  ├─ context.py
│  │  ├─ prompts.py
│  │  ├─ text_to_sql.py
│  │  ├─ sql_parsing.py
│  │  ├─ grounding.py
│  │  ├─ validation.py
│  │  ├─ sql_security.py
│  │  └─ llm/
│  │     ├─ base.py
│  │     ├─ gemini.py
│  │     └─ ollama.py
│  ├─ schemas/                      # API Pydantic schemas
│  ├─ services/                     # execution & orchestration services
│  └─ scripts/                      # extractor & index builder scripts
└─ tests/
```

## 17. Frontend architecture

- The frontend is a TypeScript React application (`frontend/`). It communicates with the backend API over HTTP and never talks to PostgreSQL directly. The frontend is served as a Vite-based dev server or can be built for production.
- API client lives in `frontend/src/api` and components live in `frontend/src/components`.
- The UI shows an input for natural-language questions, example questions, the result table, and error messages. The frontend uses the backend API routes to request generation and/or execution.

## 18. API (endpoints implemented)

Implemented HTTP endpoints (FastAPI)

- `POST /api/generate-sql`
  - Purpose: Generation-only endpoint. Converts a natural-language question into a grounded PostgreSQL SELECT SQL string. The endpoint never executes SQL.
  - Request body: `GenerateSQLRequest` (`question: str`)
  - Response: `GenerateSQLResponse` (fields mirror `GeneratedSQL`: `question`, `processed_question`, `sql`, `model`, `grounded`, `retrieved_documents`, `retrieval_scores`, `issues`, `error`)
  - Error codes: 503 (LLM unavailable), 502 (LLM error), 400 (bad input)

- `POST /api/query`
  - Purpose: Full generation → grounding → security validation → read-only execution (when execution is enabled via `exec_db_user`/`exec_db_password`). Returns typed results.
  - Request body: `QueryRequest` (`question: str`)
  - Response: `QueryResponse` (structured execution result)
  - Error codes mapped from execution statuses including:
    - 503 LLM unavailable or DB disabled
    - 502 LLM error
    - 400 no/ungrounded SQL
    - 403 security rejection
    - 504 statement timeout
    - 500 unexpected execution error

- `GET /` (root)
  - Purpose: metadata pointer (service name, docs path, health endpoint)

- `/docs` (automatic FastAPI interactive API docs) and `/redoc` are available at runtime.

## 19. Frontend ↔ Backend contract (response fields)

### Generate-only response (`GenerateSQLResponse` / `GeneratedSQL`)

- `question`: string (original question)
- `processed_question`: string | null (normalized question used for retrieval)
- `sql`: string | null (generated SQL, `null` if not produced)
- `model`: string (provider and model used)
- `grounded`: bool (true when grounding succeeded)
- `retrieved_documents`: list[string] (document IDs returned by retriever)
- `retrieval_scores`: list[float] (scores per retrieved document)
- `issues`: list[string] (grounding/validation issues)
- `error`: string | null

### Query/execution response (`QueryResponse`)

- The executed query result includes typed rows, column metadata, and execution status. Execution status mapping and error reporting are returned in structured fields and follow the status-to-HTTP mapping defined in `backend/app/api/routes/query.py`.

## 20. Project structure (concise)

Top-level:
- `.env.example`
- `.gitignore`
- `docker-compose.yml`
- `backend/` (Python FastAPI backend + RAG)
- `frontend/` (React + TypeScript)
- `docs/` (documentation)
- `rag/` (RAG artifacts directory at runtime)
- `README.md` (this file)

## 21. Technology stack (exact)

| Layer            | Technology / package (from repo)                         |
|------------------|-----------------------------------------------------------|
| Frontend         | TypeScript, React, Vite (frontend/ package.json)         |
| Backend framework| Python, FastAPI (`fastapi >= 0.115.0`)                    |
| ASGI server      | uvicorn (`uvicorn[standard] >= 0.30.0`)                   |
| DB access        | SQLAlchemy (`>= 2.0.30`), psycopg[binary] (`>= 3.2.0`)    |
| Embeddings       | sentence-transformers (local model)                       |
| Vector store     | FAISS (faiss-cpu)                                         |
| Vector math      | numpy                                                      |
| LLM providers    | google-genai (Gemini SDK) and an Ollama adapter           |
| SQL parser       | sqlglot (`>= 25.0.0`)                                     |
| Testing          | pytest, httpx                                             |
| Packaging/infra  | docker-compose.yml                                        |

## 22. Installation (exact commands)

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Node.js (for frontend)
- Docker & docker-compose (if using Compose)
- PostgreSQL (local or container)

### Backend local (non-Docker)
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Configure environment:

- Copy `.env.example` → `.env` and set values for:
  - `db_host`, `db_port`, `db_name`, `db_user`, `db_password`
  - `gemini_api_key` (if using Gemini)
  - `exec_db_user`, `exec_db_password` (if enabling execution)

4. Initialize or provide a PostgreSQL instance and ensure the extractor can access it (the extractor reads live metadata).

5. Extract schema metadata (creates `rag/metadata/schema_metadata.json`):

```bash
python -m app.scripts.extract_metadata
```

6. Build the knowledge index:

```bash
python -m app.scripts.build_vector_index
```

7. Run the backend:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker Compose

To run with Docker Compose:

```bash
docker-compose up --build
```

## 23. Docker (what exists)

- `docker-compose.yml` is present at repository root and orchestrates services required for local development (backend, frontend, and DB). Run `docker-compose up --build` to start the local environment as configured by that file. Environment variables referenced in `docker-compose.yml` should be provided in a `.env` file or via your shell.

## 24. Testing

### Backend tests

- The repository includes a backend tests directory and pytest configuration (`backend/pytest.ini`). To run backend tests:

```bash
cd backend
pytest
```

### Frontend tests

- Refer to `frontend/package.json` test scripts (`npm test` or similar depending on `package.json`).

### Integration notes

- Some tests or live checks call external LLM providers and therefore require valid API keys and available provider endpoints for full integration tests.

## 25. Security & secret management

- Example environment file: `.env.example`. Do not commit real secrets.
- Secrets used by the backend are loaded using Pydantic Settings (`backend/app/core/config.py`). `gemini_api_key` and `exec_db_password` are treated as secret values (`SecretStr`).
- The frontend does not contain server-side secrets; the frontend communicates with the backend, which holds credentials and executes queries under the configured read-only identity.

## 26. Design decisions (engineering rationale)

- RAG + schema grounding: providing schema documents and relationships to the LLM dramatically reduces hallucination and produces more reliable SQL.
- Local embeddings + FAISS: enables fast, reproducible, offline retrieval without external embedding APIs.
- Deterministic grounding + AST-based security: necessary because LLMs are non-deterministic; deterministic checks and parsing are required for safe execution.
- Separate generation and execution: prevents LLMs from being implicitly trusted to execute SQL and allows a validation checkpoint.
- Read-only execution identity: enforces least privilege and reduces blast radius in the event of unexpected SQL.

## 27. Phases (implemented)

Phase 1 — Database
- Objective: interact with PostgreSQL and extract metadata.
- Implementation: `backend/app/db` and `backend/app/scripts`; extractor collects schema metadata.

Phase 2 — Schema RAG / Knowledge Extraction
- Objective: convert schema metadata into knowledge documents.
- Implementation: `backend/app/rag/extractor.py` and `backend/app/rag/knowledge.py`

Phase 3 — Retrieval / FAISS
- Objective: embed documents and build a FAISS index.
- Implementation: `backend/app/rag/embeddings.py` and `backend/app/rag/vector_store.py`

Phase 4 — Text-to-SQL
- Objective: retrieval + context assembly + LLM generation (generation-only).
- Implementation: `backend/app/rag/text_to_sql.py`, `prompts.py`, `context.py`

Phase 4.1 — Gemini Integration
- Objective: provider abstraction and Google Gemini support.
- Implementation: `backend/app/rag/llm/gemini.py` (google-genai usage)

Phase 5 — Security / Execution / API
- Objective: grounding, AST validation, read-only execution, and API endpoints.
- Implementation: `backend/app/rag/sql_parsing.py`, `grounding.py`, `sql_security.py`, `backend/app/services/sql_execution.py`, API routes in `backend/app/api/routes/`

Phase 6 — Frontend
- Objective: Web UI (React + TypeScript) to send questions and render results.
- Implementation: `frontend/` directory (Vite app)

Phase 7 — Query preprocessing & result processing
- Objective: normalize queries, guard prompt sizes, and serialize results.
- Implementation: `backend/app/rag/preprocessing.py` and result handling in services/routes

Phase 8
- Not implemented in this repository.

## 28. Demonstration questions (examples)

The repository includes curated query examples used by the retrieval & prompting pipeline; they demonstrate multi-table joins, aggregation, and filtering. Example question categories include:

- Aggregation and ranking across departments/courses
- Multi-table join queries combining students, courses, marks/enrollments and departments
- Filters based on attendance/marks thresholds

> See the project’s curated query examples artifact for exact question texts used by the assembler.

## 29. Example end-to-end query (conceptual)

Example user question (conceptual): "Get the details of students who scored more than 80% marks in all subjects."

System actions:
1. Preprocess the question.
2. Retriever returns schema documents and relevant relationship documents and example queries.
3. Prompt assembled and sent to the configured LLM provider.
4. LLM returns candidate SQL.
5. SQL is parsed and validated as a single SELECT.
6. Grounding confirms referenced tables and columns exist and that joins are valid.
7. Security validator inspects the AST for forbidden constructs.
8. Read-only execution service executes the SQL under the configured low-privilege role (if execution is enabled).
9. Results are serialized to typed JSON and returned to the frontend.

The generated SQL shown by `GenerateSQLResponse` is for inspection; the UI does not need to expose SQL unless desired.

## 30. Generalization (how to adopt SchemaRAG for another DB)

To adapt to a new relational database:
1. Point extractor at the new database and run `python -m app.scripts.extract_metadata` to produce the metadata snapshot.
2. Run `python -m app.scripts.build_vector_index` to regenerate `knowledge.jsonl`, embeddings, and the FAISS index.
3. Optionally update curated business rules and query examples to reflect domain-specific language.
4. Ensure the read-only execution role for the target DB is created and credentials are set in environment variables if execution is required.
5. Use the same retrieval → generation → grounding → validation → execution logic with the new schema artifacts.

## 31. Limitations (current)

- Live LLM provider dependency: generation endpoints rely on configured LLM providers (Gemini or Ollama). Provider API keys and quotas affect live generation tests.
- Local embedding model & FAISS: model downloads and FAISS index builds require CPU time & disk space.
- Demo schema canonical listing: the canonical demo schema definitions (tables/columns/constraints and row counts) are produced in `rag/metadata/schema_metadata.json` by the extractor; this README references those artifacts as the authoritative source of schema details.
- No LICENSE file is present in the repository root.

## 32. Future improvements (NOT IMPLEMENTED)

- Add CI that runs a full end-to-end test with a local DB and an LLM shim.
- Add a reproducible demo dataset (DDL + seed inserts) checked into the repo for deterministic local runs.
- Add more curated domain-specific example sets and retrieval evaluation tooling.
- Add optional authentication & RBAC for API access.

## 33. Project status

- Implemented phases: 1 through 7.
- Phase 8: not implemented.

## 34. License

- No LICENSE file is present in this repository.

## Appendices

### Run commands summary

- Backend install:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r backend/requirements.txt`
- Build knowledge artifacts:
  - `python -m app.scripts.extract_metadata`
  - `python -m app.scripts.build_vector_index`
- Run backend:
  - `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000`
- Frontend:
  - `cd frontend`
  - `npm install`
  - `npm run dev`
- Docker:
  - `docker-compose up --build`

### Configuration & environment variables (exact names from `backend/app/core/config.py`)

- `db_host`
- `db_port`
- `db_name`
- `db_user`
- `db_password`
- `db_echo`
- `rag_output_dir`
- `embedding_model`
- `rag_index_dir`
- `embedding_batch_size`
- `llm_provider`
- `gemini_api_key`
- `gemini_model`
- `ollama_base_url`
- `ollama_model`
- `llm_timeout_seconds`
- `rag_top_k`
- `max_context_chars`
- `sql_max_rows`
- `sql_statement_timeout_ms`
- `exec_db_user`
- `exec_db_password`

### API endpoints (summary)

- `POST /api/generate-sql` — generation-only (returns generated SQL and grounding diagnostics)
- `POST /api/query` — generation + grounding + security validation + read-only execution (when `exec_db_user` and `exec_db_password` are configured)
- `GET /`, `GET /docs`, `GET /health` (service metadata & health)

### Where to look for canonical artifacts

- Extracted schema snapshot: `rag/metadata/schema_metadata.json` (produced by `python -m app.scripts.extract_metadata`)
- Knowledge documents: `rag/documents/knowledge.jsonl` (produced as part of the knowledge generation step)
- FAISS index: `rag/index/index.faiss` and `rag/index/document_store.json` (produced by `python -m app.scripts.build_vector_index`)

---

### Final confirmation

- README.md updated in the repository with the new authoritative technical documentation.
- Sections added: project overview, motivation, core idea, college demonstration database, features, architecture, end-to-end flow, RAG architecture, knowledge representation, text-to-sql pipeline, LLM providers, grounding, security, database execution, backend & frontend architecture, API details, frontend↔backend contract, installation, docker, testing, security, limitations, design decisions, extensibility, phases/status, license, appendices.
- Only `README.md` was modified by this commit. No other source files, tests, configuration, database files, frontend, or backend files were changed.
