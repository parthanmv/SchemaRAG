function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      role="status"
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ${
        ok ? "bg-emerald-100 text-emerald-800" : "bg-red-100 text-red-800"
      }`}
    >
      <span aria-hidden="true">{ok ? "\u2713" : "\u2717"}</span>
      {label}
    </span>
  );
}

/** Grounded / Not grounded indicator. */
export function GroundingStatus({ grounded }: { grounded: boolean }) {
  return (
    <Badge ok={grounded} label={grounded ? "Grounded" : "Not grounded"} />
  );
}

/** Security Approved / Rejected indicator. */
export function SecurityStatus({ allowed }: { allowed: boolean }) {
  return (
    <Badge ok={allowed} label={allowed ? "Approved" : "Rejected"} />
  );
}
