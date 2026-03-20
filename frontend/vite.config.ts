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

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      host: true,
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
    test: {
      environment: "jsdom",
      setupFiles: "./src/test-setup.ts"
    }
  };
});
