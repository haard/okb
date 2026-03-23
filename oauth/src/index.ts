import { OAuthProvider } from "@cloudflare/workers-oauth-provider";
import { WorkerEntrypoint } from "cloudflare:workers";
import githubHandler from "./github-handler";
import type { Env, UserProps } from "./types";

// API handler — receives only authenticated requests.
// The library validates the OAuth token and puts the user's props
// (set during completeAuthorization) on this.ctx.props.
class McpProxyHandler extends WorkerEntrypoint<Env> {
  async fetch(request: Request): Promise<Response> {
    const props = (this.ctx as unknown as { props: UserProps }).props;
    const upstream = new URL(this.env.MCP_UPSTREAM_URL);

    const url = new URL(request.url);
    url.hostname = upstream.hostname;
    url.protocol = upstream.protocol;
    url.port = upstream.port;

    const headers = new Headers(request.headers);
    headers.set("Authorization", `Bearer ${props.upstreamToken}`);

    return fetch(url.toString(), {
      method: request.method,
      headers,
      body: request.body,
    });
  }
}

export default new OAuthProvider<Env, UserProps>({
  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/token",
  clientRegistrationEndpoint: "/register",

  apiRoute: ["/mcp", "/sse"],
  apiHandler: McpProxyHandler,

  defaultHandler: githubHandler,
});
