import { useEffect, useState } from "react";
import { getHealth } from "../api/client";
import type { HealthResponse } from "../api/types";

type HealthState =
  | { kind: "checking" }
  /** Backend answered /health (200 or 503 - the body carries DB state). */
  | { kind: "up"; data: HealthResponse }
  /** No HTTP response at all: network failure / timeout. */
  | { kind: "down" };

const POLL_INTERVAL_MS = 60_000;

function dotClass(state: HealthState): string {
  switch (state.kind) {
    case "checking":
      return "bg-slate-400";
    case "up":
      return state.data.database === "connected"
        ? "bg-emerald-500"
        : "bg-amber-500";
    case "down":
      return "bg-red-500";
  }
}

function label(state: HealthState): string {
  switch (state.kind) {
    case "checking":
      return "Checking backend…";
    case "up": {
      const db =
        state.data.database === "connected"
          ? "Database connected"
          : "Database unavailable";
      return `Backend connected · ${db}`;
    }
    case "down":
      return "Backend is unavailable.";
  }
}

/**
 * Backend/database status indicator. Polls GET /health gently
 * (once on mount, then every minute) and on window focus.
 * A 503 health answer still means the backend is reachable; only network
 * failures mark the whole backend as down.
 */
export default function HealthStatus() {
  const [state, setState] = useState<HealthState>({ kind: "checking" });

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const data = await getHealth();
        if (!cancelled) setState({ kind: "up", data });
      } catch {
        if (!cancelled) setState({ kind: "down" });
      }
    }

    check();
    const timer = setInterval(check, POLL_INTERVAL_MS);
    window.addEventListener("focus", check);
    return () => {
      cancelled = true;
      clearInterval(timer);
      window.removeEventListener("focus", check);
    };
  }, []);

  const ariaLive = state.kind === "checking" ? "off" : "polite";

  return (
    <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600">
      <span aria-hidden="true" className={`inline-block h-2 w-2 rounded-full ${dotClass(state)}`} />
      <span role="status" aria-live={ariaLive}>
        {label(state)}
      </span>
    </div>
  );
}
