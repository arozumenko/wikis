import { betterAuth } from 'better-auth';
import { prismaAdapter } from 'better-auth/adapters/prisma';
import { apiKey } from '@better-auth/api-key';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const dbProvider = process.env.DATABASE_URL?.startsWith('postgres') ? 'postgresql' : 'sqlite';

export const auth = betterAuth({
  database: prismaAdapter(prisma, {
    provider: dbProvider as 'sqlite' | 'postgresql',
  }),
  emailAndPassword: {
    enabled: true,
  },
  socialProviders: {
    ...(process.env.GITHUB_CLIENT_ID && {
      github: {
        clientId: process.env.GITHUB_CLIENT_ID,
        clientSecret: process.env.GITHUB_CLIENT_SECRET!,
      },
    }),
    ...(process.env.GOOGLE_CLIENT_ID && {
      google: {
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      },
    }),
  },
  plugins: [
    apiKey({
      schema: {
        apikey: {
          modelName: 'apiKey',
          fields: {
            referenceId: 'userId',
          },
        },
      },
      defaultPrefix: 'wikis_',
      defaultKeyLength: 48,
      keyExpiration: {
        defaultExpiresIn: 90 * 24 * 60 * 60 * 1000, // 90 days
      },
      rateLimit: {
        enabled: false,
      },
    }),
  ],
  session: {
    expiresIn: 24 * 60 * 60, // 24 hours in seconds
    cookieCache: {
      enabled: true,
      maxAge: 5 * 60, // 5 min cache
    },
  },
  advanced: {
    // HTTP = no secure cookies, HTTPS = secure cookies.
    // BETTER_AUTH_URL must be set (e.g. http://localhost:3000 or https://app.example.com).
    useSecureCookies: (process.env.BETTER_AUTH_URL ?? '').startsWith('https://'),
  },
  trustedOrigins: [
    'http://localhost:3000',
    process.env.BETTER_AUTH_URL ?? '',
    process.env.FRONTEND_URL ?? '',
  ].filter(Boolean),
});
