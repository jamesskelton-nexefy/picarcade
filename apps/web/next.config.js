/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  typescript: {
    // During build, Next.js will type check your project
    ignoreBuildErrors: false,
  },
  eslint: {
    // ESLint will run during builds
    ignoreDuringBuilds: false,
  },
}

module.exports = nextConfig 