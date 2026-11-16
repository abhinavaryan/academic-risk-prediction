import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV !== "production";

// Production builds are a static export (served by Vercel's CDN or by FastAPI in Docker).
// In development, /api is proxied to the local FastAPI server on :8000.
const config: NextConfig = {
  output: isDev ? undefined : "export",
  trailingSlash: true,
  images: { unoptimized: true },
  poweredByHeader: false,
  reactStrictMode: true,
  ...(isDev
    ? {
        async rewrites() {
          return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
        },
      }
    : {}),
};

export default config;
