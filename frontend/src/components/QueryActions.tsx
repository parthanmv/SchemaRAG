interface QueryActionsProps {
  onReset: () => void;
  onExecute: () => void;
  executing: boolean;
  canSubmit: boolean;
  hasInput: boolean;
}

/**
 * Action row for the question form:
 *  - Execute Query -> POST /api/query (grounding + security + read-only run)
 */
export default function QueryActions({
  onReset,
  onExecute,
  executing,
  canSubmit,
  hasInput,
}: QueryActionsProps) {
  const busy = executing;

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
