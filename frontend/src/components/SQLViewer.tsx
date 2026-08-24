import { useState } from "react";

const KEYWORDS =
  /\b(SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|ON|AS|AND|OR|NOT|NULL|IS|IN|LIKE|BETWEEN|CASE|WHEN|THEN|ELSE|END|DISTINCT|COUNT|SUM|AVG|MIN|MAX|ROUND|CAST|ASC|DESC)\b/gi;

/** Minimal PostgreSQL keyword highlighter - purely cosmetic. */
function highlight(sql: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  KEYWORDS.lastIndex = 0;
  while ((match = KEYWORDS.exec(sql)) !== null) {
    if (match.index > last) {
      parts.push(sql.slice(last, match.index));
    }
    parts.push(
      <span key={match.index} className="font-semibold text-indigo-700">
        {match[0]}
      </span>,
    );
    last = match.index + match[0].length;
  }
  if (last < sql.length) {
    parts.push(sql.slice(last));
  }
  return parts;
}

interface SQLViewerProps {
  sql: string;
  title?: string;
}

/** Read-only SQL code block with copy button and horizontal scrolling. */
export default function SQLViewer({ sql, title }: SQLViewerProps) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (e.g. insecure context) - ignore silently */
    }
  }

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-900" data-testid="sql-viewer">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-1.5">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
          {title ?? "SQL"}
        </span>
        <button
          type="button"
          onClick={copy}
          className="rounded px-2 py-0.5 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-white"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="max-h-64 overflow-auto p-3 text-left">
        <code className="block whitespace-pre font-mono text-[13px] leading-relaxed text-slate-100">
          {highlight(sql)}
        </code>
      </pre>
    </div>
  );
}
