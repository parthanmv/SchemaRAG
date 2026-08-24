import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import HealthStatus from "./HealthStatus";

function stubFetchOnce(impl: () => Response | Promise<Response>) {
  vi.stubGlobal("fetch", vi.fn(() => impl()));
}

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    headers: new Headers(),
  } as unknown as Response;
}

beforeEach(() => {
  stubFetchOnce(() =>
    jsonResponse(200, { status: "healthy", database: "connected", detail: null }),
  );
});

describe("HealthStatus", () => {
  it("shows backend + database connected when healthy", async () => {
    render(<HealthStatus />);
    expect(await screen.findByText(/Backend connected/i)).toHaveTextContent(
      "Backend connected · Database connected",
    );
  });

  it("shows the database-unavailable state when /health answers 503", async () => {
    stubFetchOnce(() =>
      jsonResponse(503, {
        status: "unhealthy",
        database: "unavailable",
        detail: "PostgreSQL is unreachable.",
      }),
    );
    render(<HealthStatus />);
    expect(await screen.findByText(/Database unavailable/i)).toBeInTheDocument();
  });

  it("marks the whole backend down only on network failure", async () => {
    stubFetchOnce(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<HealthStatus />);
    await waitFor(() =>
      expect(screen.getByText("Backend is unavailable.")).toBeInTheDocument(),
    );
  });
});
