interface QueryActionsProps {
  onReset: () => void;
  onGenerate: () => void;
  onExecute: () => void;
  generating: boolean;
  executing: boolean;
  canSubmit: boolean;
  hasInput: boolean;
}

/**
 * Action row for the question form:
 *  - Generate SQL -> POST /api/generate-sql (generation ONLY)
 *  - Execute Query -> POST /api/query (grounding + security + read-only run)
 */
export default function QueryActions({
  onReset,
  onGenerate,
  onExecute,
  generating,
  executing,
  canSubmit,
  hasInput,
}: QueryActionsProps) {
  const busy = generating || executing;

  return (
    <div className="flex flex-wrap gap-2">
      <button
        type="button"
        onClick={onReset}
        disabled={busy || !hasInput}
        className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Clear
      </button>
      <button
        type="button"
        onClick={onGenerate}
        disabled={!canSubmit || busy}
        data-testid="generate-button"
        className="rounded-md border border-indigo-200 bg-indigo-50 px-4 py-2 text-sm font-medium text-indigo-700 hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {generating ? "Generating…" : "Generate SQL"}
      </button>
      <button
        type="button"
        onClick={onExecute}
        disabled={!canSubmit || busy}
        data-testid="execute-button"
        className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {executing ? "Executing…" : "Execute Query"}
      </button>
    </div>
  );
}
