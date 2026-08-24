import type { GeneratedSQL, QueryResult } from "../api/types";

const DOC_GROUPS: { key: string; label: string }[] = [
  { key: "schema", label: "Schema" },
  { key: "relationship", label: "Relationships" },
  { key: "constraint", label: "Constraints" },
  { key: "business_rule", label: "Business rules" },
  { key: "query_example", label: "Query examples" },
];

interface DocEntry {
  id: string;
  score?: number;
}

/** Longest matching id-prefix wins so e.g. business_rule_* is not split. */
function groupOf(docId: string): string {
  const id = docId.toLowerCase();
  const prefixes = DOC_GROUPS.map((g) => g.key)
    .sort((a, b) => b.length - a.length)
    .find((key) => id.startsWith(`${key}_`));
  return prefixes ?? "other";
}

/**
 * Collapsible panel listing the documents the retriever fed to the LLM.
 * Displays only what the backend returns (document ids + scores);
 * no retrieval logic lives in the frontend.
 */
export default function RetrievalPanel({
  documents,
  scores,
}: {
  documents: string[] | undefined;
  scores?: number[];
}) {
  if (!documents || documents.length === 0) return null;

  const entries: DocEntry[] = documents.map((id, i) => ({
    id,
    score: scores?.[i],
  }));

  const grouped = new Map<string, DocEntry[]>();
  for (const entry of entries) {
    const g = groupOf(entry.id);
    if (!grouped.has(g)) grouped.set(g, []);
    grouped.get(g)!.push(entry);
  }
  const orderedGroups = [
    ...DOC_GROUPS.map((g) => ({ ...g, docs: grouped.get(g.key) })),
    { key: "other", label: "Other", docs: grouped.get("other") },
  ].filter((g) => g.docs && g.docs.length > 0);

  return (
    <details className="rounded-lg border border-slate-200 bg-white" data-testid="retrieval-panel">
      <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium text-slate-700">
        Retrieved documents ({documents.length})
      </summary>
      <div className="space-y-3 border-t border-slate-100 px-4 py-3">
        {orderedGroups.map((group) => (
          <div key={group.key}>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
              {group.label}
            </h4>
            <ul className="space-y-1">
              {group.docs!.map((doc) => (
                <li
                  key={doc.id}
                  className="flex flex-wrap items-baseline justify-between gap-2 rounded bg-slate-50 px-2.5 py-1.5"
                >
                  <code className="break-all font-mono text-xs text-slate-700">{doc.id}</code>
                  {typeof doc.score === "number" && (
                    <span className="text-xs tabular-nums text-slate-400">
                      score {doc.score.toFixed(4)}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </details>
  );
}

/** Metadata strip under generated SQL: model, grounded flag, document count. */
export function GenerationMeta({ gen }: { gen: GeneratedSQL }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
      <span>
        Model: <span className="font-medium text-slate-700">{gen.model}</span>
      </span>
      <span>
        Retrieved documents:{" "}
        <span className="font-medium text-slate-700">{gen.retrieved_documents.length}</span>
      </span>
      {gen.issues.length > 0 && (
        <span className="text-amber-700">Issues: {gen.issues.join("; ")}</span>
      )}
      {gen.processed_question && gen.processed_question !== gen.question && (
        <span data-testid="processed-question">
          Preprocessed:{" "}
          <span className="font-medium text-slate-700">{gen.processed_question}</span>
        </span>
      )}
    </div>
  );
}

/** Execution metadata strip: row count, duration, executing role. */
export function ExecutionMeta({ result }: { result: QueryResult }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
      <span>
        Rows returned:{" "}
        <span className="font-medium text-slate-700">{result.row_count}</span>
      </span>
      {result.execution_time_ms !== null && (
        <span>
          Execution time:{" "}
          <span className="font-medium text-slate-700 tabular-nums">
            {result.execution_time_ms.toFixed(1)} ms
          </span>
        </span>
      )}
      {result.executed_as && (
        <span>
          Executed as:{" "}
          <code className="font-mono text-[11px] font-medium text-slate-700">
            {result.executed_as}
          </code>
        </span>
      )}
    </div>
  );
}
