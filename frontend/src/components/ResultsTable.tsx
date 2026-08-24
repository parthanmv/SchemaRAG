interface ResultsTableProps {
  columns: string[];
  rows: unknown[][];
  /** Phase 7: per-column display kinds from the backend (number columns
   *  get right-aligned tabular figures). */
  columnKinds?: string[] | null;
}

function Cell({ value, numeric }: { value: unknown; numeric?: boolean }) {
  if (value === null || value === undefined) {
    return (
      <td className="whitespace-nowrap px-3 py-1.5 text-right text-xs italic text-slate-400">
        NULL
      </td>
    );
  }
  const text = typeof value === "number" ? String(value) : String(value);
  return (
    <td
      className={`max-w-xs truncate px-3 py-1.5 text-sm text-slate-700 ${
        numeric ? "text-right tabular-nums" : ""
      }`}
      title={text}
    >
      {text}
    </td>
  );
}

/**
 * Result grid for successful /api/query responses.
 * Renders exactly what the backend returned; respects the server-side
 * row limit; supports empty sets, NULLs, long text and wide tables.
 */
export default function ResultsTable({ columns, rows, columnKinds }: ResultsTableProps) {
  const isNumeric = (j: number) =>
    columnKinds?.[j] === "number" || columnKinds?.[j] === "boolean";
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white" data-testid="results-table">
      <table className="min-w-full divide-y divide-slate-200 text-left">
        <thead className="bg-slate-50">
          <tr>
            {columns.map((col, j) => (
              <th
                key={col}
                scope="col"
                className={`sticky top-0 whitespace-nowrap bg-slate-50 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-500 ${
                  isNumeric(j) ? "text-right" : ""
                }`}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-3 py-6 text-center text-sm italic text-slate-500">
                The query executed successfully but returned no rows.
              </td>
            </tr>
          ) : (
            rows.map((row, i) => (
              <tr key={i} className={i % 2 === 0 ? "" : "bg-slate-50/60"}>
                {row.map((value, j) => (
                  <Cell key={j} value={value} numeric={isNumeric(j)} />
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
