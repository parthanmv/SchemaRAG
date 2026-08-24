# SchemaRAG

A RAG-powered natural-language interface for a PostgreSQL **college database**.

The full product pipeline (built across phases) is:

```
User Question → Query Preprocessing → RAG Retrieval → Relevant Schema + Rules
→ LLM → SQL Generation → SQL Parsing & Validation → PostgreSQL
→ Result Processing → React UI
```

> **Current status: Phase 7 complete** — the full pipeline above is now
> implemented end to end:
>
> - **Phase 1**: PostgreSQL 18 infrastructure, schema, deterministic seed
>   data, SQLAlchemy layer, FastAPI `/health`.
> - **Phase 2**: metadata extraction + deterministic knowledge generation.
> - **Phase 3**: local embeddings + FAISS semantic retrieval (evaluated).
> - **Phase 4 / 4.1**: RAG-backed Text-to-SQL generation via the Gemini API
>   (official `google-genai` SDK, provider abstraction retained).
> - **Phase 5**: SQL security validation + read-only execution through the
>   dedicated low-privilege `schemarag_reader` role (`/api/query`).
> - **Phase 6**: React + Vite + TypeScript UI consuming both endpoints.
> - **Phase 7**: explicit **query preprocessing** (unicode/whitespace
>   normalisation before retrieval and prompting; original question still
>   echoed by the API) and **result processing** (shared JSON-safe coercion +
>   per-column `column_kinds` annotations for presentation).

---

## 1. Project overview & structure

```
SchemaRAG/
├── backend/
│   ├── app/
│   │   ├── api/routes/        # FastAPI routers (/health, /api/generate-sql, /api/query)
│   │   ├── core/              # Pydantic Settings configuration
│   │   ├── db/                # Engine, session factory, FastAPI dependency
│   │   ├── models/            # SQLAlchemy 2.x ORM models (6 tables)
│   │   ├── rag/               # Phase 2: extractor, knowledge generator,
│   │   │                      #   validation; Phase 3: embeddings,
│   │   │                      #   vector store, retriever, evaluation;
│   │   │                      #   Phase 4: context, prompts, llm providers
│   │   │                      #   (Gemini + Ollama), SQL parsing,
│   │   │                      #   grounding, text-to-sql orchestration;
│   │   │                      #   Phase 7: query preprocessing
│   │   ├── schemas/           # Pydantic response models
│   │   ├── scripts/           # seed_database.py, extract_metadata.py,
│   │   │                      #   build_vector_index.py, test_retrieval.py
│   │   ├── services/          # Business logic (health, sql_execution,
│   │   │                      #   result_processing)
│   │   └── main.py            # FastAPI app factory
│   ├── tests/                 # pytest suite (Phase 1-7)
│   ├── requirements.txt
│   └── pytest.ini
├── rag/                        # generated knowledge base (rebuildable)
│   ├── documents/knowledge.jsonl
│   ├── index/                  # Phase 3 FAISS artifacts
│   │   ├── index.faiss
│   │   └── document_store.json
│   └── metadata/schema_metadata.json
├── frontend/                  # Phase 6: React + Vite + TypeScript UI
│   ├── src/api/               # centralized API client + backend types
│   ├── src/components/        # header, health, query input, SQL viewer,
│   │                          #   retrieval panel, results table, errors
│   └── .env.example           # VITE_API_BASE_URL template
├── docs/                      # project documentation
├── docker-compose.yml         # PostgreSQL 18 container (optional)
├── .env.example               # configuration template — copy to .env
├── .gitignore                 # ensures .env is never committed
└── README.md
```

### Phase 1 scope

| Included                                             | Excluded (later phases)      |
| ---------------------------------------------------- | ---------------------------- |
| PostgreSQL schema (6 tables) + constraints + indexes | RAG / embeddings / FAISS     |
| Deterministic realistic seed data                    | LLM integration              |
| SQLAlchemy engine/session/dependency                 | Text-to-SQL generation       |
| Pydantic settings via environment variables          | SQL parsing & validation     |
| `GET /health` API + DB status                        | `/api/query` endpoint        |
| pytest suite against real PostgreSQL                 | React UI                     |

---

## 2. Prerequisites

- **Python 3.11+** (developed on 3.14)
- **PostgreSQL 14+** running locally *or* Docker with Compose
- (Optional) `psql` client for manual inspection

## 3. Environment variables

Copy the template to the repository root and edit:

```bash
cp .env.example .env
```

| Variable            | Description                          | Example value used in dev |
| ------------------- | ------------------------------------ | ------------------------- |
| `DB_HOST`           | PostgreSQL host                      | `localhost`               |
| `DB_PORT`           | PostgreSQL port                      | `5432`                    |
| `DB_NAME`           | Database name                        | `college_db`              |
| `DB_USER`           | Application role (**required**)      | `schemarag`               |
| `DB_PASSWORD`       | Role password (**required**)         | *(set your own)*          |
| `DB_ECHO`           | Echo SQL statements (`true`/`false`) | `false`                   |
| `LLM_PROVIDER`      | Text-to-SQL backend (`gemini`)       | `gemini`                  |
| `GEMINI_MODEL`      | Gemini model name                    | `gemini-3.6-flash`        |
| `GEMINI_API_KEY`    | Google AI Studio key (**secret**)    | *(set your own)*          |
| `OLLAMA_BASE_URL`   | Local fallback server URL            | `http://localhost:11434`  |
| `OLLAMA_MODEL`      | Local fallback model                 | `llama3.2:3b`             |

`.env` is gitignored — never commit real credentials. The equivalent
SQLAlchemy URL is assembled internally:
`postgresql+psycopg://user:password@host:port/college_db`.

### LLM providers (Phase 4.1)

Text-to-SQL generation runs through the pluggable `LLMProvider`
abstraction in `backend/app/rag/llm/`. The active provider is
**Google Gemini** via the official [`google-genai`](https://pypi.org/project/google-genai/)
SDK:

```
RAG retrieval → ContextAssembler → GeminiProvider → SQL parsing → grounding
```

- `GeminiProvider` reads `GEMINI_API_KEY` from the environment / `.env`
  (never hardcoded, never printed) and generates with `GEMINI_MODEL`.
- Missing key, invalid key, network/API failures and empty responses raise
  typed errors (`LLMUnavailableError` / `LLMResponseError`) — no fake SQL is
  ever returned silently.
- `OllamaProvider` remains available as a local, key-free fallback:
  set `LLM_PROVIDER=ollama`, pull a model with `ollama pull llama3.2:3b`.
- Unit tests always use a deterministic fake provider; the one live Gemini
  test runs only when credentials are configured and otherwise skips.

## 4. Start PostgreSQL

**Option A — native installation:** ensure the service is running, then create
the app role and database once (as a superuser):

```sql
CREATE ROLE schemarag LOGIN PASSWORD '<your-password>';
CREATE DATABASE college_db OWNER schemarag;
```

**Option B — Docker Compose:**

```bash
cp .env.example .env      # set DB_PASSWORD first
docker compose up -d      # starts postgres:18-alpine on DB_PORT
```

## 5. Initialize the database (tables)

```bash
cd backend
python -m venv ../.venv
../.venv/Scripts/pip install -r requirements.txt    # Windows Git-Bash path
# source ../.venv/bin/activate                       # Linux/macOS
```

Tables are created automatically by the seeder (step 6). To create them
without data you can also run:

```bash
python -c "import app.models; from app.db.base import Base; from app.db.session import engine; Base.metadata.create_all(engine)"
```

## 6. Seed data

From `backend/` (with the venv active):

```bash
python -m app.scripts.seed_database             # drop + recreate + insert
python -m app.scripts.seed_database --dry-run   # generate only, print digest
python -m app.scripts.seed_database --seed 7    # different reproducible dataset
```

The seeder drops/recreates all tables, inserts ~27k rows in one transaction,
resets identity sequences, and verifies row-count targets. The same seed
always produces byte-identical data (SHA-256 digest is printed).

Seeded volumes (seed=42): **8 departments, 50 courses, 1,000 students,
5,049 enrollments, 15,147 marks, 5,049 attendance records.**

## 7. Start FastAPI

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health: `curl http://127.0.0.1:8000/health`

Healthy:

```json
{"status":"healthy","database":"connected","detail":null}
```

When PostgreSQL is unreachable the endpoint degrades gracefully instead of
crashing (HTTP **503**, may take a few seconds while connection attempts
time out):

```json
{"status":"unhealthy","database":"unavailable",
 "detail":"PostgreSQL is unreachable; check DB_* environment settings."}
```

## 8. Build the RAG knowledge base (Phase 2)

From `backend/`:

```bash
python -m app.scripts.extract_metadata
```

Connects to PostgreSQL, reflects the **actual** schema via SQLAlchemy
inspection, validates it against the Phase 1 contract, and deterministically
writes:

- `rag/metadata/schema_metadata.json` — full metadata snapshot
  (tables, columns, types, nullability, PKs, FKs, CHECK/UNIQUE constraints)
- `rag/documents/knowledge.jsonl` — 38 RAG documents:
  6 schema + 8 relationship + 16 constraint + 3 business rule +
  5 query example, each with `document_id`, `document_type`, `tables`
  and `source` (`postgresql_metadata` | `domain_rules` | `curated_examples`).

The run prints a SHA-256 digest; rerunning with an unchanged schema produces
byte-identical output. Optional: `--output-dir <path>` overrides the output
location.

## 9. Semantic retrieval (Phase 3)

Build the vector index from the knowledge base (local embedding model, no
API keys):

```bash
python -m app.scripts.build_vector_index
```

Embeds all documents with `sentence-transformers/all-MiniLM-L6-v2`
(384-dim, L2-normalised) into a FAISS `IndexFlatIP` (cosine via inner
product). `rag/index/document_store.json` maps every FAISS position to its
document and records the embedding model + `knowledge.jsonl` SHA-256, so a
stale index is detected and rejected rather than silently used.

Try retrieval / run the evaluation:

```bash
python -m app.scripts.test_retrieval --top-k 5 "Which students have attendance below 75%?"
python -m app.scripts.test_retrieval --eval          # Recall@1/3/5 + table-level
```

Programmatic API: `from app.rag.retriever import retrieve; retrieve("...", top_k=5)`
returns ranked results with full metadata; optional `document_type=` filter.
Evaluation on the 10-question set: **Recall@1/3/5 = 100%** (document-level),
table-level recall 80%@1 → 100%@3/5.

## 10. Text-to-SQL generation (Phase 4)

Pipeline: `question → KnowledgeRetriever → ContextAssembler → prompt →
Gemini → SQL parsing → grounding check`. The generated SELECT is returned
for inspection; **it is never executed** (execution + security validation is
Phase 5).

With credentials configured (`LLM_PROVIDER=gemini`, `GEMINI_API_KEY`,
`GEMINI_MODEL`):

```bash
# HTTP API
curl -X POST http://127.0.0.1:8000/api/generate-sql \
     -H "Content-Type: application/json" \
     -d '{"question": "Which department has the highest average marks?"}'

# CLI
python -m app.scripts.generate_sql "Which department has the highest average marks?"
```

The response reports the retrieved documents with scores, the generated
SQL, and the grounding verdict (`grounded: true/false` plus any issues).
Hallucinated tables/columns are flagged `not_grounded` and never repaired.

## 10b. Secure execution: POST /api/query (Phase 5)

Turns a question into an *executed* answer:
`question → RAG generation → grounding check → AST security validation →
read-only PostgreSQL execution → typed JSON result`.

Three independent layers of protection:

1. **Grounding** - SQL may only reference the six real tables/columns from
   Phase 2 metadata (hallucinations rejected before validation).
2. **AST security validator** (`app/rag/sql_security.py`, sqlglot) - allowlist
   of read-only constructs; rejects every write/DDL/admin statement, multiple
   statements, comments, `pg_catalog`/`information_schema`, dangerous
   functions (`pg_sleep`, `pg_read_file`, `dblink`, sequence setters, ...),
   `SELECT INTO`, locking clauses and data-modifying CTEs. Fails closed.
3. **Database role** - execution uses a dedicated login (`schemarag_reader`)
   with SELECT-only grants on the six tables; each connection also sets
   server-side `default_transaction_read_only=on` and `statement_timeout`.

Create the role once (needs admin credentials, e.g. the `postgres` role):

```bash
cd backend
set ADMIN_DB_USER=postgres
set ADMIN_DB_PASSWORD=...
python -m app.scripts.setup_readonly_role   # writes EXEC_DB_* into .env
```

Then ask questions with data-backed answers:

```bash
curl -X POST http://127.0.0.1:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Which department has the highest average marks?"}'
```

Response fields include `execution_status`
(`success | empty_result | row_limit_exceeded | invalid_sql | ungrounded |
security_rejected | statement_timeout | connection_error | permission_denied |
execution_error | execution_disabled`), `columns`, JSON-safe `rows`,
`row_count`, and `execution_time_ms`. Results are capped at `SQL_MAX_ROWS`
(default 500) and every statement is capped at `SQL_STATEMENT_TIMEOUT_MS`
(default 5000). HTTP codes: 400 ungrounded/no SQL, 403 security rejection,
502 LLM error, 503 LLM/DB unavailable or execution disabled, 504 timeout.

## 10c. Web UI (Phase 6)

A React + Vite + TypeScript frontend lives in `frontend/`. It talks **only**
to this API (`GET /health`, `POST /api/generate-sql`, `POST /api/query`) -
the browser never connects to PostgreSQL and never executes SQL.

```bash
cd frontend
npm install
copy .env.example .env        # sets VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev                   # http://localhost:5173
npm run build                 # production bundle in dist/
npm run test                  # vitest suite (no live Gemini/PostgreSQL needed)
```

Features: health indicator, natural-language question input with example
questions, Generate SQL (inspection only - never executed by the browser),
Execute Query via `/api/query`, SQL viewer with copy button, grounding and
security status badges, retrieved-documents panel with scores, results table
(row count / execution time / `executed_as`), friendly error handling and a
responsive layout. The API base URL is configured through `VITE_API_BASE_URL`;
no secrets live in the frontend.

## 10d. Pipeline polish: preprocessing + result processing (Phase 7)

The two remaining stages of the product pipeline diagram are explicit,
tested modules:

- **Query preprocessing** (`backend/app/rag/preprocessing.py`): before
  retrieval and prompting the question is normalised - typographic quotes
  and dashes become ASCII, non-breaking/zero-width spaces disappear,
  whitespace collapses, over-long input truncates at 500 chars. The
  transformation is deterministic and idempotent. The API keeps echoing the
  *original* question; responses additionally expose
  `processed_question`, and the UI shows a "Preprocessed:" line when it
  differs.
- **Result processing** (`backend/app/services/result_processing.py`):
  JSON-safe coercion of driver values (Decimal → float, date/time → ISO
  strings) now has a single shared implementation, plus per-column
  `column_kinds` annotations (`number` / `boolean` / `text` / `null` /
  `unknown`) on `/api/query` results. Values themselves are never altered
  semantically; the UI uses kinds to right-align numeric columns.

Both stages are additive: every Phase 1-6 contract (fields, statuses,
security gates) is unchanged.

## 11. Run tests

```bash
cd backend
python -m pytest -v
```

Tests run against the *real* PostgreSQL database and the real local
embedding model: Phase 1 tests (API, DB, seed data), Phase 2 tests (live
metadata extraction, knowledge generation, artifact validation,
determinism), Phase 3 tests (embeddings, FAISS store, retrieval,
stale-index detection, evaluation), Phase 4 tests (context assembly,
prompting, SQL extraction, grounding, orchestration, mocked Gemini provider,
API), Phase 5 tests (security validator battery, execution gates/limits,
role setup, API contract) and Phase 7 tests (preprocessing idempotency,
column-kind inference, pipeline wiring). LLM unit tests are deterministic;
the single live Gemini test runs only when `LLM_PROVIDER=gemini` and
`GEMINI_API_KEY` are set, otherwise it skips with a clear reason.

Tests assume the database has been seeded (step 6) and `build_vector_index`
has run (step 9).

## 12. Database schema overview

```
departments 1─* students        departments 1─* courses
students *─* courses  (via enrollments: unique per student+course+year+semester)
students 1─* marks *─1 courses  (one row per exam_type: quiz/midterm/final)
students 1─* attendance *─1 courses  (aggregate per course per term)
```

| Table         | Key columns                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `departments` | `department_id` PK, `department_name` UQ, `department_code` UQ                                     |
| `students`    | `student_id` PK, `roll_number` UQ, `name`, `email` UQ, FK→departments, `semester` 1–8, `admission_year` |
| `courses`     | `course_id` PK, `course_code` UQ, `course_name`, `credits` 1–6, FK→departments                     |
| `enrollments` | `enrollment_id` PK, FKs→students/courses, `academic_year`, `semester`; UQ(student,course,year,sem) |
| `marks`       | `mark_id` PK, FKs→students/courses, `exam_type`, `marks` NUMERIC(5,2) CHECK 0–100; UQ incl. exam   |
| `attendance`  | `attendance_id` PK, FKs→students/courses, `classes_held/attended` CHECKs, `attendance_percentage`; UQ |

All FK columns are indexed; uniqueness constraints prevent duplicate
student-course-term records. Seed generation uses per-student latent
"ability"/"diligence" plus department/course offsets so averages differ
realistically across departments, courses, exams and students.

## 13. Example SQL queries

```sql
-- Top 5 students by average marks
SELECT s.name, ROUND(AVG(m.marks),2) AS avg_marks
FROM students s JOIN marks m ON m.student_id = s.student_id
GROUP BY s.student_id, s.name ORDER BY avg_marks DESC LIMIT 5;

-- Department with the highest average marks
SELECT d.department_name, ROUND(AVG(m.marks),2) AS avg_marks
FROM marks m
JOIN students st ON st.student_id = m.student_id
JOIN departments d ON d.department_id = st.department_id
GROUP BY d.department_name ORDER BY avg_marks DESC;

-- Students with attendance below 75%
SELECT DISTINCT s.roll_number, s.name
FROM students s JOIN attendance a ON a.student_id = s.student_id
WHERE a.attendance_percentage < 75;

-- Departments with average marks above 75 (HAVING)
SELECT d.department_code, AVG(m.marks) AS avg_marks
FROM marks m JOIN students s ON s.student_id = m.student_id
JOIN departments d ON d.department_id = s.department_id
GROUP BY d.department_code HAVING AVG(m.marks) > 75;

-- Students scoring above their department average (subquery/CTE)
WITH dept_avg AS (
  SELECT s.department_id, AVG(m.marks) AS avg_marks
  FROM marks m JOIN students s ON s.student_id = m.student_id
  GROUP BY s.department_id
)
SELECT s.name, AVG(m.marks) AS student_avg, da.avg_marks AS dept_avg
FROM students s
JOIN marks m ON m.student_id = s.student_id
JOIN dept_avg da ON da.department_id = s.department_id
GROUP BY s.student_id, s.name, da.avg_marks
HAVING AVG(m.marks) > da.avg_marks;
```

---

## Known issues / notes

- Without Docker on this machine, PostgreSQL runs as a native Windows
  service; `docker-compose.yml` is provided but untested here.
- When the DB is unreachable, `/health` waits for psycopg connection attempts
  (~6–8s worst case) before reporting 503.
- The seeder intentionally **drops and recreates** all tables — do not point
  it at a shared production database.
