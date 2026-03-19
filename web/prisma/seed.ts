import { PrismaClient } from '@prisma/client';
import { hashPassword } from 'better-auth/crypto';
import { randomUUID } from 'crypto';

const prisma = new PrismaClient();
const dbProvider = process.env.DATABASE_URL?.startsWith('postgres') ? 'PostgreSQL' : 'SQLite';

async function main() {
  console.log(`Seeding database (${dbProvider})...`);
  const userCount = await prisma.user.count();
  if (userCount > 0) {
    console.log(`Database already has ${userCount} user(s), skipping seed.`);
    return;
  }

  // Try sign-up via API first (if server is running)
  const baseUrl = process.env.BETTER_AUTH_URL ?? 'http://localhost:3000';

  try {
    const resp = await fetch(`${baseUrl}/api/auth/sign-up/email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'admin',
        email: 'admin@wikis.dev',
        password: 'changeme123',
      }),
    });

    if (resp.ok) {
      console.warn('⚠️  Default admin created (admin@wikis.dev / changeme123) — change password immediately!');
      return;
    }
    console.log('API sign-up failed, falling back to direct DB insert...');
  } catch {
    console.log('Auth server not running. Using direct DB insert...');
  }

  // Direct insert using Better Auth's own hashPassword
  const userId = randomUUID();
  const accountId = randomUUID();
  const hashedPassword = await hashPassword('changeme123');

  await prisma.user.create({
    data: {
      id: userId,
      name: 'admin',
      email: 'admin@wikis.dev',
      emailVerified: false,
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  });

  await prisma.account.create({
    data: {
      id: accountId,
      accountId: userId,
      providerId: 'credential',
      userId: userId,
      password: hashedPassword,
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  });

  console.warn('⚠️  Default admin created (admin@wikis.dev / changeme123) — change password immediately!');
}

main()
  .catch((e) => {
    console.error('Seed failed:', e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
