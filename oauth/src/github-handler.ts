import type { Env, UserProps, AuthRequest } from "./types";

/** TOKEN_MAP values: single token string or {dbName: token} object. */
type TokenMapEntry = string | Record<string, string>;

/** Stashed state for the database picker page. */
interface SelectState {
  oauthReqInfo: AuthRequest;
  user: { login: string; name: string | null };
  tokens: Record<string, string>; // dbName -> token
}

/**
 * Resolve a user's TOKEN_MAP entry into a {dbName: token} record.
 * Single strings are keyed by the database name extracted from the token format.
 */
function resolveTokens(entry: TokenMapEntry): Record<string, string> | null {
  if (typeof entry === "string") {
    const match = entry.match(/^okb_([a-z0-9_-]+)_(ro|rw)_/);
    const dbName = match ? match[1] : "default";
    return { [dbName]: entry };
  }
  if (typeof entry === "object" && entry !== null && Object.keys(entry).length > 0) {
    return entry;
  }
  return null;
}

/** Complete the OAuth flow with a chosen token. */
async function completeWithToken(
  env: Env,
  oauthReqInfo: AuthRequest,
  user: { login: string; name: string | null },
  upstreamToken: string,
): Promise<Response> {
  const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
    request: oauthReqInfo,
    userId: user.login,
    metadata: { label: user.name || user.login },
    scope: oauthReqInfo.scope,
    props: { login: user.login, upstreamToken },
  });
  return Response.redirect(redirectTo, 302);
}

/** Render the database picker HTML page. */
function renderPicker(
  selectKey: string,
  origin: string,
  dbNames: string[],
  userName: string,
): Response {
  const buttons = dbNames
    .map(
      (db) => `
      <form method="POST" action="${origin}/select">
        <input type="hidden" name="key" value="${selectKey}">
        <input type="hidden" name="db" value="${db}">
        <button type="submit">${db}</button>
      </form>`,
    )
    .join("\n");

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Select Knowledge Base</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex; justify-content: center; align-items: center;
      min-height: 100vh; background: #f5f5f5; padding: 1rem;
    }
    .card {
      background: white; border-radius: 12px; padding: 2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 360px; width: 100%;
      text-align: center;
    }
    h1 { font-size: 1.25rem; margin-bottom: 0.5rem; }
    .sub { color: #666; font-size: 0.875rem; margin-bottom: 1.5rem; }
    form { margin-bottom: 0.75rem; }
    button {
      width: 100%; padding: 0.75rem 1rem; font-size: 1rem;
      border: 1px solid #ddd; border-radius: 8px; background: white;
      cursor: pointer; transition: background 0.15s;
    }
    button:hover { background: #f0f0f0; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Select Knowledge Base</h1>
    <p class="sub">Signed in as <strong>${userName}</strong></p>
    ${buttons}
  </div>
</body>
</html>`;

  return new Response(html, {
    status: 200,
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}

/**
 * Handles the default (non-API) routes:
 *   GET  /authorize  -> parse OAuth request, stash in KV, redirect to GitHub
 *   GET  /callback   -> exchange GitHub code, look up token(s), complete or show picker
 *   POST /select     -> complete OAuth with chosen database token
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // ── GET /authorize ──────────────────────────────────────────
    if (url.pathname === "/authorize") {
      const oauthReqInfo = await env.OAUTH_PROVIDER.parseAuthRequest(request);

      if (!oauthReqInfo.clientId) {
        return new Response("Invalid OAuth request", { status: 400 });
      }

      const stateKey = crypto.randomUUID();
      await env.OAUTH_KV.put(`oauth_state:${stateKey}`, JSON.stringify(oauthReqInfo), {
        expirationTtl: 300,
      });

      const githubAuthUrl = new URL("https://github.com/login/oauth/authorize");
      githubAuthUrl.searchParams.set("client_id", env.GITHUB_CLIENT_ID);
      githubAuthUrl.searchParams.set("redirect_uri", `${url.origin}/callback`);
      githubAuthUrl.searchParams.set("scope", "read:user");
      githubAuthUrl.searchParams.set("state", stateKey);

      return Response.redirect(githubAuthUrl.toString(), 302);
    }

    // ── GET /callback ───────────────────────────────────────────
    if (url.pathname === "/callback") {
      const code = url.searchParams.get("code");
      const stateKey = url.searchParams.get("state");

      if (!code || !stateKey) {
        return new Response("Missing code or state", { status: 400 });
      }

      const stored = await env.OAUTH_KV.get(`oauth_state:${stateKey}`);
      if (!stored) {
        return new Response("Invalid or expired state", { status: 400 });
      }
      await env.OAUTH_KV.delete(`oauth_state:${stateKey}`);
      const oauthReqInfo: AuthRequest = JSON.parse(stored);

      // Exchange code for GitHub access token
      const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          client_id: env.GITHUB_CLIENT_ID,
          client_secret: env.GITHUB_CLIENT_SECRET,
          code,
          redirect_uri: `${url.origin}/callback`,
        }),
      });

      const tokenData = (await tokenRes.json()) as { access_token?: string; error?: string };
      if (!tokenData.access_token) {
        return new Response(`GitHub token exchange failed: ${tokenData.error}`, { status: 401 });
      }

      // Fetch GitHub user info
      const userRes = await fetch("https://api.github.com/user", {
        headers: {
          Authorization: `Bearer ${tokenData.access_token}`,
          "User-Agent": "mcp-oauth-shim",
        },
      });
      if (!userRes.ok) {
        return new Response("Failed to fetch GitHub user", { status: 401 });
      }
      const ghUser = (await userRes.json()) as { login?: string; name?: string };
      if (!ghUser.login) {
        return new Response("Failed to fetch GitHub user", { status: 401 });
      }

      // Look up token(s) for this user
      const tokenMap: Record<string, TokenMapEntry> = JSON.parse(env.TOKEN_MAP);
      const entry = tokenMap[ghUser.login];
      if (!entry) {
        return new Response("Access denied", { status: 403 });
      }

      const tokens = resolveTokens(entry);
      if (!tokens) {
        return new Response("Access denied", { status: 403 });
      }

      const user = { login: ghUser.login, name: ghUser.name || null };
      const dbNames = Object.keys(tokens);

      // Single database — complete immediately (unchanged behavior)
      if (dbNames.length === 1) {
        return completeWithToken(env, oauthReqInfo, user, tokens[dbNames[0]]);
      }

      // Multiple databases — show picker
      const selectKey = crypto.randomUUID();
      const selectState: SelectState = { oauthReqInfo, user, tokens };
      await env.OAUTH_KV.put(`oauth_select:${selectKey}`, JSON.stringify(selectState), {
        expirationTtl: 300,
      });

      return renderPicker(selectKey, url.origin, dbNames, user.name || user.login);
    }

    // ── POST /select ────────────────────────────────────────────
    if (url.pathname === "/select" && request.method === "POST") {
      const formData = await request.formData();
      const selectKey = formData.get("key") as string | null;
      const dbName = formData.get("db") as string | null;

      if (!selectKey || !dbName) {
        return new Response("Missing key or db", { status: 400 });
      }

      const stored = await env.OAUTH_KV.get(`oauth_select:${selectKey}`);
      if (!stored) {
        return new Response("Selection expired. Please reconnect.", { status: 400 });
      }
      await env.OAUTH_KV.delete(`oauth_select:${selectKey}`);

      const state: SelectState = JSON.parse(stored);
      const token = state.tokens[dbName];
      if (!token) {
        return new Response("Invalid database selection", { status: 400 });
      }

      return completeWithToken(env, state.oauthReqInfo, state.user, token);
    }

    return new Response("Not found", { status: 404 });
  },
};
