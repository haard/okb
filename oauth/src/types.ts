export interface AuthRequest {
  clientId: string;
  redirectUri: string;
  scope: string[];
  [key: string]: unknown;
}

export interface Env {
  OAUTH_KV: KVNamespace;

  // The OAuthProvider binding — injected automatically by the library
  OAUTH_PROVIDER: {
    parseAuthRequest(request: Request): Promise<AuthRequest>;
    completeAuthorization(options: {
      request: AuthRequest;
      userId: string;
      metadata?: { label: string };
      scope: string[];
      props: UserProps;
    }): Promise<{ redirectTo: string }>;
  };

  // GitHub OAuth App credentials (set via wrangler secret)
  GITHUB_CLIENT_ID: string;
  GITHUB_CLIENT_SECRET: string;

  // Your actual MCP server behind cloudflared
  MCP_UPSTREAM_URL: string;

  // JSON string: { "github_username": "tok_xxx", ... }
  // Set via wrangler secret put TOKEN_MAP
  TOKEN_MAP: string;
}

// Stored as the authenticated user's "props" after OAuth completes
export interface UserProps {
  login: string;          // GitHub username
  upstreamToken: string;  // The pre-existing token for this user
}
