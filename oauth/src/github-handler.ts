import type { Env, UserProps, AuthRequest } from "./types";

/**
 * Handles the default (non-API) routes:
 *   GET  /authorize  -> parse OAuth request, stash in KV, redirect to GitHub
 *   GET  /callback   -> exchange GitHub code, look up token, complete authorization
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // ── GET /authorize ──────────────────────────────────────────
    // Claude.ai sends a standard OAuth authorize request here.
    // We parse it via the library, stash the parsed request in KV,
    // and redirect to GitHub with a state token pointing at the KV entry.
    if (url.pathname === "/authorize") {
      const oauthReqInfo = await env.OAUTH_PROVIDER.parseAuthRequest(request);

      if (!oauthReqInfo.clientId) {
        return new Response("Invalid OAuth request", { status: 400 });
      }

      // Store the parsed OAuth request in KV with a random key (5min TTL)
      const stateKey = crypto.randomUUID();
      await env.OAUTH_KV.put(
        `oauth_state:${stateKey}`,
        JSON.stringify(oauthReqInfo),
        { expirationTtl: 300 }
      );

      // Redirect to GitHub for authentication
      const githubAuthUrl = new URL("https://github.com/login/oauth/authorize");
      githubAuthUrl.searchParams.set("client_id", env.GITHUB_CLIENT_ID);
      githubAuthUrl.searchParams.set("redirect_uri", `${url.origin}/callback`);
      githubAuthUrl.searchParams.set("scope", "read:user");
      githubAuthUrl.searchParams.set("state", stateKey);

      return Response.redirect(githubAuthUrl.toString(), 302);
    }

    // ── GET /callback ───────────────────────────────────────────
    // GitHub redirects here with ?code=...&state=...
    if (url.pathname === "/callback") {
      const code = url.searchParams.get("code");
      const stateKey = url.searchParams.get("state");

      if (!code || !stateKey) {
        return new Response("Missing code or state", { status: 400 });
      }

      // Retrieve and delete the stashed OAuth request
      const stored = await env.OAUTH_KV.get(`oauth_state:${stateKey}`);
      if (!stored) {
        return new Response("Invalid or expired state", { status: 400 });
      }
      await env.OAUTH_KV.delete(`oauth_state:${stateKey}`);
      const oauthReqInfo: AuthRequest = JSON.parse(stored);

      // Exchange code for GitHub access token
      const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
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
      const user = (await userRes.json()) as { login?: string; name?: string };
      if (!user.login) {
        return new Response("Failed to fetch GitHub user", { status: 401 });
      }

      // Look up pre-existing token for this user
      const tokenMap: Record<string, string> = JSON.parse(env.TOKEN_MAP);
      const upstreamToken = tokenMap[user.login];

      if (!upstreamToken) {
        return new Response("Access denied", { status: 403 });
      }

      // Complete the OAuth flow
      const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
        request: oauthReqInfo,
        userId: user.login,
        metadata: { label: user.name || user.login },
        scope: oauthReqInfo.scope,
        props: {
          login: user.login,
          upstreamToken,
        },
      });

      return Response.redirect(redirectTo, 302);
    }

    return new Response("Not found", { status: 404 });
  },
};
