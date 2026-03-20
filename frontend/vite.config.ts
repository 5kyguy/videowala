import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

function normalizePublicHost(raw: string): string {
  return raw
    .trim()
    .replace(/^https?:\/\//, "")
    .split("/")[0]
    .replace(/:\d+$/, "");
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const publicHost = normalizePublicHost(env.VITE_DEV_PUBLIC_HOST ?? "");
  /** Where FastAPI listens (dev/preview proxy only). Browser never sees this when using `/api` base URL. */
  const apiProxyTarget = (env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000").trim() || "http://127.0.0.1:8000";
  const apiProxy = {
    "/api": {
      target: apiProxyTarget,
      changeOrigin: true,
      rewrite: (p: string) => p.replace(/^\/api/, "")
    }
  };

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      host: true,
      proxy: apiProxy,
      ...(publicHost
        ? {
            origin: `https://${publicHost}`,
            allowedHosts: [publicHost],
            hmr: {
              protocol: "wss",
              host: publicHost,
              clientPort: 443
            }
          }
        : {})
    },
    preview: {
      host: true,
      proxy: apiProxy
    },
    test: {
      environment: "jsdom",
      setupFiles: "./src/test-setup.ts"
    }
  };
});
