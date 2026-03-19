import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  transpilePackages: ['better-auth'],
  productionBrowserSourceMaps: false,
  experimental: {
    optimizePackageImports: ['@mui/material', '@mui/icons-material'],
  },
  // API proxying is handled by middleware.ts (step 3 in Next.js resolution
  // order), which rewrites non-SSE /api/v1/* requests to the backend.
  // SSE-streaming endpoints (/ask, /research, /invocations/*/stream) are
  // excluded from middleware proxying and served by App Router route handlers
  // in src/app/api/v1/ that add no-buffer headers for streaming.
};

export default nextConfig;
