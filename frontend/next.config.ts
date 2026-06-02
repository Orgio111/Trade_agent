import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  async rewrites() {
    // Use env vars for service discovery (Docker Compose vs local dev)
    const apiTarget = process.env.API_TARGET || "http://localhost:8001";
    const wsTarget = process.env.WS_TARGET || "http://localhost:8082";
    return [
      {
        source: "/api/:path*",
        destination: `${apiTarget}/api/v1/:path*`,
      },
      {
        source: "/ws",
        destination: `${wsTarget}/ws`,
      },
    ];
  },
};

export default nextConfig;
