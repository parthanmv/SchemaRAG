import type {
  ErrorDetail,
  GeneratedSQL,
  HealthResponse,
  QueryResult,
} from "./types";

const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

const REQUEST_TIMEOUT_MS = 60_000;

/** User-facing error with a friendly message; never contains stack traces. */
export class ApiError extends Error {
  readonly kind:
    | "network"
    | "timeout"
    | "bad_request"
    | "security_rejected"
    | "validation"
    | "llm_error"
    | "unavailable"
    | "server"
    | "malformed";
  /** Backend-provided reasons (e.g. security issues); safe to display. */
  readonly issues: string[];

  constructor(kind: ApiError["kind"], message: string, issues: string[] = []) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.issues = issues;
  }
}

function baseUrl(): string {
  return BASE_URL.replace(/\/+$/, "");
}

async function request<T>(
  path: string,
  init: RequestInit & {
    timeoutMs?: number;
    /** Endpoints like /health deliver usable bodies even with 4xx/5xx. */
    parseNonOk?: boolean;
  } = {},
): Promise<T> {
  const { timeoutMs = REQUEST_TIMEOUT_MS, parseNonOk = false, ...fetchInit } = init;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${baseUrl()}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...fetchInit,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("timeout", "The request timed out. Please try again.");
    }
    throw new ApiError("network", "Backend is unavailable.");
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok && !parseNonOk) {
    throw await toApiError(response, path);
  }

  try {
    return (await response.json()) as T;
  } catch {
    if (!response.ok) throw await toApiError(response, path);
    throw new ApiError("malformed", "The backend returned an unexpected response.");
  }
}

function detailOf(body: unknown): {
  structured: ErrorDetail | null;
  textDetail: string;
} {
  const inner = (body as { detail?: unknown } | null)?.detail;
  const structured =
    typeof inner === "object" && inner !== null ? (inner as ErrorDetail) : null;
  const textDetail =
    typeof inner === "string"
      ? inner
      : Array.isArray(inner)
        ? // FastAPI 422 validation errors
          inner
            .map((e) =>
              typeof e === "object" && e !== null && "msg" in e
                ? String((e as { msg: unknown }).msg)
                : String(e),
            )
            .join("; ")
        : "";
  return { structured, textDetail };
}

async function readBody(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function toApiError(response: Response, path: string): Promise<ApiError> {
  const body = await readBody(response);
  const { structured, textDetail } = detailOf(body);
  const isQuery = path === "/api/query";

  switch (response.status) {
    case 400:
      if (structured?.error === "insufficient_context") {
        return new ApiError(
          "bad_request",
          "Schema information is insufficient to generate a reliable query.",
        );
      }
      return new ApiError(
        "bad_request",
        "Generated SQL references unknown database objects.",
        structured?.issues ?? [],
      );
    case 403:
      return new ApiError(
        "security_rejected",
        "This query was rejected by the SQL security layer.",
        structured?.issues ?? [],
      );
    case 422:
      return new ApiError(
        "validation",
        textDetail || "Please provide a valid question (3-500 characters).",
      );
    case 502:
      return new ApiError(
        "llm_error",
        "SQL generation service is currently unavailable.",
      );
    case 503:
      // On /api/query a 503 can mean the LLM OR the database side is down;
      // the sanitised backend detail tells us which message fits best.
      if (isQuery && !/LLM backend is unavailable/i.test(textDetail)) {
        return new ApiError("server", "Query execution failed.");
      }
      return new ApiError(
        "unavailable",
        "SQL generation service is currently unavailable.",
      );
    case 504:
      return new ApiError(
        "server",
        isQuery
          ? "Query execution failed."
          : "The request exceeded the allowed time and was cancelled.",
      );
    default:
      return new ApiError(
        response.status >= 500 ? "server" : "bad_request",
        response.status >= 500 && isQuery
          ? "Query execution failed."
          : textDetail || `Request failed (HTTP ${response.status}).`,
      );
  }
}

// ---------------------------------------------------------------------------
// Endpoint helpers
// ---------------------------------------------------------------------------
/**
 * GET /health. Both 200 ("healthy") and 503 ("unhealthy") carry a JSON body
 * describing backend/database state, so non-OK answers are parsed, not thrown.
 * Only network failures / timeouts reject here.
 */
export function getHealth(timeoutMs = 8000): Promise<HealthResponse> {
  return request<HealthResponse>("/health", {
    method: "GET",
    timeoutMs,
    parseNonOk: true,
  });
}

export function generateSql(question: string): Promise<GeneratedSQL> {
  return request<GeneratedSQL>("/api/generate-sql", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function executeQuery(question: string): Promise<QueryResult> {
  return request<QueryResult>("/api/query", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}
