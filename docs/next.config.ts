import { createMDX } from 'fumadocs-mdx/next';
import path from 'path';

const withMDX = createMDX();

const nextConfig = withMDX({
  output: 'export',
  trailingSlash: true,
  basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? '',
  images: { unoptimized: true },
  // Silence "inferred workspace root" warning from sibling lockfiles
  outputFileTracingRoot: path.resolve(__dirname),
});

export default nextConfig;
