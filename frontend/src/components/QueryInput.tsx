import QueryActions from "./QueryActions";

export const QUESTION_MIN_LENGTH = 3;
export const QUESTION_MAX_LENGTH = 500;

interface QueryInputProps {
  value: string;
  onChange: (value: string) => void;
  onReset: () => void;
  onGenerate: () => void;
  onExecute: () => void;
  generating: boolean;
  executing: boolean;
}

/**
 * Question textarea with length validation, action buttons
 * (Generate SQL = generation-only, Run Query = full pipeline)
 * and a clear/reset control.
 */
export default function QueryInput({
  value,
  onChange,
  onReset,
  onGenerate,
  onExecute,
  generating,
  executing,
}: QueryInputProps) {
  const trimmed = value.trim();
  const isValid =
    trimmed.length >= QUESTION_MIN_LENGTH && trimmed.length <= QUESTION_MAX_LENGTH;
  const busy = generating || executing;

  return (
    <section aria-label="Question input" className="space-y-3">
      <label htmlFor="question" className="block text-sm font-medium text-slate-700">
        Ask a question about the college database
      </label>
      <textarea
        id="question"
        rows={3}
        maxLength={QUESTION_MAX_LENGTH}
        value={value}
        disabled={busy}
        placeholder="e.g. Which department has the highest average marks?"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && isValid && !busy) {
            e.preventDefault();
            onExecute();
          }
        }}
        className="w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 disabled:bg-slate-100"
      />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-slate-400">
          {value.length}/{QUESTION_MAX_LENGTH} characters · Ctrl+Enter runs the query
        </p>
        <QueryActions
          onReset={onReset}
          onGenerate={onGenerate}
          onExecute={onExecute}
          generating={generating}
          executing={executing}
          canSubmit={isValid}
          hasInput={Boolean(value)}
        />
      </div>
    </section>
  );
}
