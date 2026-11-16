import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import worker, { type Env } from "../src/index";

const ORIGIN = "https://delphi-rag.pages.dev";
const SECRET = "shared-secret-never-in-the-browser";

function limiter(success = true) {
  return { limit: vi.fn(async () => ({ success })) } as unknown as RateLimit & {
    limit: ReturnType<typeof vi.fn>;
  };
}

function env(overrides: Partial<Env> = {}): Env {
  return {
    API_URL: "https://api.example",
    ALLOWED_ORIGINS: `${ORIGIN},http://localhost:8788`,
    API_SHARED_SECRET: SECRET,
    TURNSTILE_SECRET_KEY: "turnstile-secret",
    PER_IP: limiter(),
    SITE: limiter(),
    ...overrides,
  };
}

function ask(body: unknown, init: { origin?: string | null; method?: string } = {}) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "CF-Connecting-IP": "203.0.113.7",
  };
  if (init.origin !== null) headers.Origin = init.origin ?? ORIGIN;
  return new Request("https://worker.example/api/ask", {
    method: init.method ?? "POST",
    headers,
    body: init.method === "GET" ? undefined : JSON.stringify(body),
  });
}

const ANSWER = {
  status: "answered",
  answer: "Tiberius was killed on the Capitol.",
  answerable: true,
  citations: [{ life: "Tiberius Gracchus", section: 24, quote: "clubs and staves" }],
  cache: { served_from_cache: false, outcome: null, similarity: null, matched_question: null },
  cache_only: false,
};

let fetchMock: ReturnType<typeof vi.fn>;

function upstream({ turnstile = true, apiStatus = 200, apiBody = ANSWER as unknown } = {}) {
  fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("turnstile")) return Response.json({ success: turnstile });
    return Response.json(apiBody, { status: apiStatus });
  });
}

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

const good = { question: "How did Tiberius Gracchus die?", turnstileToken: "tok" };

describe("allowed requests", () => {
  it("forwards the question with the secret and returns the answer", async () => {
    upstream();
    const r = await worker.fetch(ask(good), env());
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual(ANSWER);
    expect(r.headers.get("Access-Control-Allow-Origin")).toBe(ORIGIN);

    const [url, init] = fetchMock.mock.calls.at(-1)!;
    expect(url).toBe("https://api.example/ask");
    expect(init.headers["X-Delphi-Key"]).toBe(SECRET);
    expect(JSON.parse(init.body)).toEqual({ question: "How did Tiberius Gracchus die?" });
  });

  it("trims the question before forwarding", async () => {
    upstream();
    await worker.fetch(ask({ ...good, question: "  How did Crassus die?  " }), env());
    expect(JSON.parse(fetchMock.mock.calls.at(-1)![1].body).question).toBe("How did Crassus die?");
  });

  it("answers CORS preflight for allowed origins only", async () => {
    const ok = await worker.fetch(ask(null, { method: "OPTIONS" }), env());
    expect(ok.status).toBe(204);
    expect(ok.headers.get("Access-Control-Allow-Methods")).toBe("POST");
    const bad = await worker.fetch(ask(null, { method: "OPTIONS", origin: "https://evil.example" }), env());
    expect(bad.status).toBe(403);
  });
});

describe("refusals, before anything costs money", () => {
  it.each([
    ["no origin", null],
    ["another site", "https://evil.example"],
  ])("refuses %s", async (_, origin) => {
    const r = await worker.fetch(ask(good, { origin }), env());
    expect(r.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses GET", async () => {
    expect((await worker.fetch(ask(null, { method: "GET" }), env())).status).toBe(405);
  });

  it.each([
    ["empty", ""],
    ["blank", "   "],
    ["too long", "x".repeat(301)],
    ["not a string", 42],
  ])("refuses a question that's %s", async (_, question) => {
    const r = await worker.fetch(ask({ question, turnstileToken: "tok" }), env());
    expect(r.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses bad JSON", async () => {
    const req = new Request("https://worker.example/api/ask", {
      method: "POST",
      headers: { Origin: ORIGIN },
      body: "{not json",
    });
    expect((await worker.fetch(req, env())).status).toBe(400);
  });

  it("limits one visitor, keyed by IP, before checking Turnstile", async () => {
    const perIp = limiter(false);
    const r = await worker.fetch(ask(good), env({ PER_IP: perIp }));
    expect(r.status).toBe(429);
    expect(perIp.limit).toHaveBeenCalledWith({ key: "203.0.113.7" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses a missing or failed Turnstile token", async () => {
    upstream({ turnstile: false });
    expect((await worker.fetch(ask(good), env())).status).toBe(403);
    expect(fetchMock).toHaveBeenCalledTimes(1); // Turnstile only, never the API
    fetchMock.mockClear();
    expect((await worker.fetch(ask({ question: "q?" }), env())).status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("limits the whole site after Turnstile passes", async () => {
    upstream();
    const site = limiter(false);
    const r = await worker.fetch(ask(good), env({ SITE: site }));
    expect(r.status).toBe(429);
    expect(((await r.json()) as { error: string }).error).toContain("3 questions a minute");
    expect(site.limit).toHaveBeenCalledWith({ key: "site" });
    expect(fetchMock).toHaveBeenCalledTimes(1); // Turnstile only
  });
});

describe("upstream trouble stays generic", () => {
  it.each([
    [503, 503, "Busy"],
    [422, 400, "can't be asked"],
    [500, 502, "Couldn't answer"],
    [401, 502, "Couldn't answer"],
  ])("API %i becomes %i", async (apiStatus, status, text) => {
    upstream({ apiStatus, apiBody: { detail: "internal detail api.example secret" } });
    const r = await worker.fetch(ask(good), env());
    expect(r.status).toBe(status);
    const body = await r.text();
    expect(body).toContain(text);
    expect(body).not.toContain("internal detail");
  });

  it("an unreachable API is a 502", async () => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      if (String(input).includes("turnstile")) return Response.json({ success: true });
      throw new TypeError("network down");
    });
    expect((await worker.fetch(ask(good), env())).status).toBe(502);
  });

  it("never sends the secret or the API address to the browser", async () => {
    for (const apiStatus of [200, 500]) {
      upstream({ apiStatus });
      const text = await (await worker.fetch(ask(good), env())).text();
      expect(text).not.toContain(SECRET);
      expect(text).not.toContain("api.example");
    }
  });

  it("other paths are 404", async () => {
    const r = await worker.fetch(new Request("https://worker.example/admin"), env());
    expect(r.status).toBe(404);
  });
});
