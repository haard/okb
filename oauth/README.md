# mcp-oauth-shim

A Cloudflare Worker that adds OAuth 2.1 (required by Claude.ai) in front
of an MCP server that uses pre-existing bearer tokens for auth.

## How it works

```
Claude.ai ──OAuth 2.1──▶ this Worker ──Bearer tok_xxx──▶ your MCP server
                              │                          (behind cloudflared)
                              │
                         GitHub login
                         ↓ maps to ↓
                      pre-existing token
```

The Worker:
1. Presents spec-compliant OAuth 2.1 endpoints (DCR, PKCE, token, etc.)
2. Redirects users to GitHub for identity
3. Looks up the GitHub username in a static token map
4. Proxies all MCP traffic to your upstream server with the real bearer token

Your MCP server stays unchanged — it still just validates bearer tokens.

## Setup

### 1. Create a GitHub OAuth App

Go to https://github.com/settings/developers → **New OAuth App**

- **Homepage URL**: `https://mcp-oauth-shim.<your-subdomain>.workers.dev`
- **Callback URL**: `https://mcp-oauth-shim.<your-subdomain>.workers.dev/callback`

Note the Client ID and generate a Client Secret.

### 2. Create the KV namespace

```bash
wrangler kv namespace create OAUTH_KV
```

Copy the returned `id` into `wrangler.toml`.

### 3. Set secrets

```bash
wrangler secret put GITHUB_CLIENT_ID
wrangler secret put GITHUB_CLIENT_SECRET

# JSON mapping of GitHub usernames to your pre-existing tokens
wrangler secret put TOKEN_MAP
# paste: {"fredrikhaard":"tok_abc123","otheruser":"tok_def456"}

wrangler secret put MCP_UPSTREAM_URL
# paste: https://your-mcp-server.example.com
```

### 4. Deploy

```bash
npm install
wrangler deploy
```

### 5. Add to Claude.ai

In Claude.ai → Settings → Integrations → Add:

```
https://mcp-oauth-shim.<your-subdomain>.workers.dev/sse
```

Claude.ai will discover the OAuth endpoints, redirect you to GitHub,
and you're connected.

## Files

| File | Purpose |
|---|---|
| `src/index.ts` | Worker entry — wires OAuthProvider |
| `src/github-handler.ts` | `/authorize` → GitHub, `/callback` → token lookup |
| `src/types.ts` | `Env` and `UserProps` types |

## Notes

- The `workers-oauth-provider` library only stores **hashes** of tokens
  in KV, never plaintext.
- `compatibility_flags = ["global_fetch_strictly_public"]` is required
  for SSRF protection — don't remove it.
- If your MCP server uses SSE at a different path than `/sse` or `/mcp`,
  adjust `apiRoute` in `index.ts`.
- For local dev: create a second GitHub OAuth app pointing at
  `http://localhost:8788/callback` and use a `.dev.vars` file.
