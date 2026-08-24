import { useCallback, useState } from "react";
import { ApiError, executeQuery, generateSql } from "./api/client";
import type { GeneratedSQL, QueryResult } from "./api/types";
import Header from "./components/Header";
import HealthStatus from "./components/HealthStatus";
import ExampleQuestions from "./components/ExampleQuestions";
import QueryInput from "./components/QueryInput";
import SQLViewer from "./components/SQLViewer";
import { GroundingStatus, SecurityStatus } from "./components/StatusBadges";
import RetrievalPanel, {
  ExecutionMeta,
  GenerationMeta,
} from "./components/RetrievalPanel";
import ResultsTable from "./components/ResultsTable";
import ErrorMessage from "./components/ErrorMessage";

const SUCCESS_STATUSES = new Set(["success", "empty_result", "row_limit_exceeded"]);

/** Friendly text for error codes returned inside 200 generation payloads. */
const GENERATION_ERROR_MESSAGES: Record<string, string> = {
  insufficient_context:
    "Schema information is insufficient to generate a reliable query.",
  invalid_response: "The AI service returned a response that could not be used.",
  not_grounded: "Generated SQL references unknown database objects.",
};

function generationErrorMessage(error: string | null): string | null {
  if (!error) return null;
  return GENERATION_ERROR_MESSAGES[error] ?? error;
}

function SectionCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
      <h2 className="text-base font-semibold text-slate-800">{title}</h2>
      {children}
    </section>
  );
}

/** Security verdict for a rejected query, with the backend-provided reasons. */
function SecurityRejection({ issues }: { issues: string[] }) {
  return (
    <SectionCard title="SQL security">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <SecurityStatus allowed={false} />
        <span className="text-xs text-slate-500">security_rejected</span>
      </div>
      {issues.length > 0 && (
        <ul
          data-testid="security-issues"
          className="list-inside list-disc space-y-1 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {issues.map((issue) => (
            <li key={issue}>{issue}</li>
          ))}
        </ul>
      )}
    </SectionCard>
  );
}

/**
 * SchemaRAG frontend. Two independent workflows:
 *  - Generate SQL  -> POST /api/generate-sql (generation ONLY, never executed)
 *  - Execute Query -> POST /api/query        (grounding + security + read-only execution)
 * The browser never connects to PostgreSQL and never executes SQL itself.
 */
export default function App() {
  const [question, setQuestion] = useState("");
  const [generating, setGenerating] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [generation, setGeneration] = useState<GeneratedSQL | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [genError, setGenError] = useState<string | null>(null);
  const [execError, setExecError] = useState<ApiError | string | null>(null);

  const reset = useCallback(() => {
    setQuestion("");
    setGeneration(null);
    setResult(null);
    setGenError(null);
    setExecError(null);
  }, []);

  const handleGenerate = useCallback(async () => {
    const q = question.trim();
    if (!q) return;
    setGenerating(true);
    setGenError(null);
    setExecError(null);
    try {
      setGeneration(await generateSql(q));
    } catch (err) {
      setGeneration(null);
      setGenError(
        err instanceof ApiError
          ? err.message
          : "SQL generation service is currently unavailable.",
      );
    } finally {
      setGenerating(false);
    }
  }, [question]);

  const handleExecute = useCallback(async () => {
    const q = question.trim();
    if (!q) return;
    setExecuting(true);
    setExecError(null);
    setGenError(null);
    try {
      const res = await executeQuery(q);
      setResult(res);
      setExecError(null);
      if (!SUCCESS_STATUSES.has(res.execution_status)) {
        // Backend completed the request but refused/could not run the query.
        setExecError(
          res.error ?? `Query could not be executed (${res.execution_status}).`,
        );
      }
    } catch (err) {
      setResult(null);
      setExecError(
        err instanceof ApiError ? err : "Query execution failed.",
      );
    } finally {
      setExecuting(false);
    }
  }, [question]);

  const execApiError = execError instanceof ApiError ? execError : null;
  const execMessage =
    typeof execError === "string"
      ? execError
      : execApiError?.message ?? null;

  return (
    <div className="min-h-screen">
      <Header>
        <HealthStatus />
      </Header>

      <main className="mx-auto max-w-6xl space-y-5 px-4 py-6 sm:px-6">
        <SectionCard title="Ask a question">
          <QueryInput
            value={question}
            onChange={setQuestion}
            onReset={reset}
            onGenerate={handleGenerate}
            onExecute={handleExecute}
            generating={generating}
            executing={executing}
          />
          <ExampleQuestions onSelect={setQuestion} disabled={generating || executing} />
        </SectionCard>

        {generating && (
          <p role="status" data-testid="generating-indicator" className="text-sm text-slate-500">
            Generating SQL…
          </p>
        )}
        {genError && (
          <ErrorMessage title="SQL generation failed" message={genError} />
        )}

        {generation && (
          <SectionCard title="Generated SQL (not executed)">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <GroundingStatus grounded={generation.grounded} />
              {!generation.sql && generation.error && (
                <span className="text-xs text-slate-500">
                  {generationErrorMessage(generation.error)}
                </span>
              )}
            </div>
            {generation.sql ? (
              <>
                <SQLViewer sql={generation.sql} title="Generated SQL" />
                <GenerationMeta gen={generation} />
                <RetrievalPanel
                  documents={generation.retrieved_documents}
                  scores={generation.retrieval_scores}
                />
              </>
            ) : (
              <p className="text-sm italic text-slate-500">
                No SQL was generated. Try rephrasing the question or pick an example.
              </p>
            )}
          </SectionCard>
        )}

        {execApiError?.kind === "security_rejected" && (
          <SecurityRejection issues={execApiError.issues} />
        )}

        {(executing || execMessage || result) && (
          <SectionCard title="Query results">
            {executing && (
              <p role="status" data-testid="executing-indicator" className="text-sm text-slate-500">
                Executing query…
              </p>
            )}
            {execMessage && !executing && (
              <ErrorMessage
                title={
                  result && !SUCCESS_STATUSES.has(result.execution_status)
                    ? "Query not executed"
                    : execApiError?.kind === "bad_request"
                      ? "Grounding check failed"
                      : execApiError?.kind === "security_rejected"
                        ? "Query rejected"
                        : "Query execution failed"
                }
                message={execMessage}
              />
            )}
            {result && SUCCESS_STATUSES.has(result.execution_status) && (
              <>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                  <SecurityStatus allowed={result.security_allowed} />
                  <GroundingStatus grounded={result.grounded} />
                  <ExecutionMeta result={result} />
                </div>
                {result.sql && <SQLViewer sql={result.sql} title="Executed SQL" />}
                <ResultsTable
                  columns={result.columns}
                  rows={result.rows}
                  columnKinds={result.column_kinds}
                />
                {result.execution_status === "row_limit_exceeded" && (
                  <p className="text-xs text-amber-700">
                    Result truncated at the server-side row limit.
                  </p>
                )}
                <RetrievalPanel documents={result.retrieved_documents} />
              </>
            )}
          </SectionCard>
        )}
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-8 pt-2 sm:px-6">
        <p className="text-xs text-slate-400">
          All SQL is generated, validated and executed by the backend through a
          dedicated read-only database role. The frontend never touches PostgreSQL.
        </p>
      </footer>
    </div>
  );
}
