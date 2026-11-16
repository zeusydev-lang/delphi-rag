/**
 * The public front door for delphi-rag questions.
 *
 *   POST /api/ask  {"question": "...", "turnstileToken": "..."}
 *
 * Checks, cheapest first, before anything costs money:
 *   1. only POST from an allowed origin (CORS)
 *   2. a question of 1-300 characters
 *   3. at most 5 questions a minute from one visitor
 *   4. a valid Cloudflare Turnstile token (a real browser, not a script)
 *   5. at most 3 questions a minute for the whole site (Voyage's free tier)
 * then forwards the question to the Python API with the shared secret.
 * The browser never sees the secret, the API's address or its errors.
 */

export interface Env {
  API_URL: string;
  ALLOWED_ORIGINS: string;
  API_SHARED_SECRET: string;
  TURNSTILE_SECRET_KEY: string;
  PER_IP: RateLimit;
  SITE: RateLimit;
}

const MAX_QUESTION_CHARS = 300;  // not exported: the runtime treats named exports as handlers
const TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify";

function allowedOrigin(request: Request, env: Env): string | null {
  const origin = request.headers.get("Origin") ?? "";
  const allowed = env.ALLOWED_ORIGINS.split(",").map((o) => o.trim());
  return allowed.includes(origin) ? origin : null;
}

function json(body: unknown, status: number, origin: string | null): Response {
  const headers: Record<string, string> = {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  };
  if (origin) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Vary"] = "Origin";
  }
  return new Response(JSON.stringify(body), { status, headers });
}

const refuse = (message: string, status: number, origin: string | null) =>
  json({ error: message }, status, origin);

async function turnstileOk(token: string, ip: string, env: Env): Promise<boolean> {
  if (!token) return false;
  const form = new FormData();
  form.append("secret", env.TURNSTILE_SECRET_KEY);
  form.append("response", token);
  if (ip) form.append("remoteip", ip);
  try {
    const r = await fetch(TURNSTILE_URL, { method: "POST", body: form });
    const outcome = (await r.json()) as { success?: boolean };
    return outcome.success === true;
  } catch {
    return false;
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname !== "/api/ask") return refuse("Not found.", 404, null);

    const origin = allowedOrigin(request, env);
    if (request.method === "OPTIONS") {
      if (!origin) return new Response(null, { status: 403 });
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": origin,
          "Access-Control-Allow-Methods": "POST",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
          Vary: "Origin",
        },
      });
    }
    if (!origin) return refuse("Not allowed.", 403, null);
    if (request.method !== "POST") return refuse("Use POST.", 405, origin);

    // 2. The question.
    let body: { question?: unknown; turnstileToken?: unknown };
    try {
      body = await request.json();
    } catch {
      return refuse("Send JSON.", 400, origin);
    }
    const question = typeof body.question === "string" ? body.question.trim() : "";
    if (!question) return refuse("Ask a question.", 400, origin);
    if (question.length > MAX_QUESTION_CHARS) {
      return refuse(`Questions can be up to ${MAX_QUESTION_CHARS} characters.`, 400, origin);
    }

    // 3. Per visitor.
    const ip = request.headers.get("CF-Connecting-IP") ?? "";
    if (!(await env.PER_IP.limit({ key: ip || "unknown" })).success) {
      return refuse("That's a lot of questions. Try again in a minute.", 429, origin);
    }

    // 4. A real browser.
    const token = typeof body.turnstileToken === "string" ? body.turnstileToken : "";
    if (!(await turnstileOk(token, ip, env))) {
      return refuse("Couldn't verify you're human. Reload the page and try again.", 403, origin);
    }

    // 5. The whole site.
    if (!(await env.SITE.limit({ key: "site" })).success) {
      return refuse(
        "The site is busy right now (it allows 3 questions a minute). Try again in a minute.",
        429,
        origin,
      );
    }

    // Forward.
    let upstream: Response;
    try {
      upstream = await fetch(`${env.API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Delphi-Key": env.API_SHARED_SECRET },
        body: JSON.stringify({ question }),
      });
    } catch {
      return refuse("Couldn't reach the answer service. Try again shortly.", 502, origin);
    }
    if (upstream.ok) return json(await upstream.json(), 200, origin);
    if (upstream.status === 503) {
      return refuse("Busy right now. Try again in a minute.", 503, origin);
    }
    if (upstream.status === 422) return refuse("That question can't be asked.", 400, origin);
    return refuse("Couldn't answer right now. Try again shortly.", 502, origin);
  },
};
