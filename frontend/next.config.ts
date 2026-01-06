import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export for self-hosted deployment
  output: "export",

  // Disable image optimization (not available in static export)
  images: {
    unoptimized: true,
  },

  // Trailing slashes for cleaner nginx routing
  trailingSlash: true,
};

export default nextConfig;
