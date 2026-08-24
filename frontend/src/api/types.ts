/**
 * Types mirroring the FastAPI backend contracts EXACTLY
 * (app/schemas/generate_sql.py, app/services/sql_execution.py, app/schemas/health.py).
 *
 * The frontend never talks to PostgreSQL and never sees credentials -
 * the backend is the only component allowed to touch the database.
 */

export interface GeneratedSQL {
  question: string;
  /** Phase 7: normalised question used for retrieval/prompting. */
  processed_question?: string | null;
  sql: string | null;
  model: string;
  grounded: boolean;
  retrieved_documents: string[];
  retrieval_scores: number[];
  issues: string[];
  error: string | null;
}

export type ExecutionStatus =
  | "success"
  | "empty_result"
  | "row_limit_exceeded"
  | "invalid_sql"
  | "ungrounded"
  | "security_rejected"
  | "statement_timeout"
  | "connection_error"
  | "permission_denied"
  | "execution_error"
  | "execution_disabled";

export interface QueryResult {
  question: string;
  sql: string | null;
  model: string;
  grounded: boolean;
  security_allowed: boolean;
  security_issues: string[];
  execution_status: ExecutionStatus;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  execution_time_ms: number | null;
  retrieved_documents: string[];
  error: string | null;
  executed_as?: string | null;
  /** Phase 7: per-column display kinds (number/boolean/text/null/unknown). */
  column_kinds?: string[] | null;
}

export interface HealthResponse {
  status: string;
  database: string;
  detail: string | null;
}

/** Structured detail the backend sends on some error responses. */
export interface ErrorDetail {
  error?: string;
  issues?: string[];
}
