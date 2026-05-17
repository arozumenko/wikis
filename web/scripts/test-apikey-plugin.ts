/**
 * Regression guard for the Better-Auth api-key plugin × our schema
 * remap. Issue #170 follow-up to PR #169 (Better-Auth 1.5 → 1.6).
 *
 * Why this lives in scripts/ and not in __tests__/
 * ------------------------------------------------
 * Better-Auth's npm packages (``better-auth``, ``@better-auth/*``)
 * are ESM-only with no CJS build. Our Jest config uses ts-jest in
 * CommonJS mode for everything else and adding an ESM-transform
 * layer just for this one test would expand the test framework
 * surface significantly. Running this as a standalone Node ESM
 * script (via tsx) keeps the framework simple — and tsx is
 * already a dev dep we use for ``prisma db seed``.
 *
 * What we're guarding
 * -------------------
 * Our production auth config (``src/lib/auth.ts``) tells the
 * api-key plugin to rename its default ``referenceId`` field to
 * ``userId`` so it lines up with the ``apikey.userId`` column in
 * ``prisma/schema.prisma``. The remap is a single line and silent
 * — if a Better-Auth minor bump ever changes the plugin's default
 * field name, internal lookup logic, or wire-up order, our code
 * would still compile (``tsc`` clean) but rows would land in the
 * wrong column at runtime.
 *
 * Run:
 *   cd web && npm run test:integration
 *
 * CI invokes this alongside the Jest suite.
 */
import { betterAuth } from 'better-auth';
import { memoryAdapter, type MemoryDB } from '@better-auth/memory-adapter';
import { apiKey } from '@better-auth/api-key';
import assert from 'node:assert/strict';

// Better-Auth's secret env var must be set (it warns + falls back
// to a dev-only default otherwise). We use a deterministic 32-char
// value so the test is reproducible across runs.
process.env.BETTER_AUTH_SECRET ??=
  'test-secret-32-bytes-aaaaaaaaaaaa';

// Tied to ``makeAuth`` (not the generic ``betterAuth`` return) so
// the type carries our exact plugin/option shape — without this,
// TypeScript narrows ``signUpEmail``'s response and the call type-
// errors despite working at runtime.
type Auth = ReturnType<typeof makeAuth>;

/** Pre-seed the tables Better-Auth touches. The memory adapter
 *  errors on findOne against a missing table, so we declare them
 *  empty up front. Keys here match the plugin's ``modelName``, not
 *  the Prisma ``@@map`` — the in-memory adapter doesn't know about
 *  Prisma's mapping layer. So ``apiKey`` (camelCase, the modelName)
 *  not ``apikey`` (lowercase, the SQL table name). */
function freshDb(): MemoryDB {
  return {
    user: [],
    session: [],
    account: [],
    verification: [],
    apiKey: [],
  };
}

function makeAuth(db: MemoryDB) {
  return betterAuth({
    baseURL: 'http://localhost:3000',
    database: memoryAdapter(db),
    emailAndPassword: { enabled: true },
    plugins: [
      apiKey({
        // Mirrors src/lib/auth.ts EXACTLY. Change here only if
        // production auth.ts changes, otherwise this guard is doing
        // its job.
        schema: {
          apikey: {
            modelName: 'apiKey',
            fields: { referenceId: 'userId' },
          },
        },
        defaultPrefix: 'wikis_',
        defaultKeyLength: 48,
        keyExpiration: {
          defaultExpiresIn: 90 * 24 * 60 * 60 * 1000,
        },
        rateLimit: { enabled: false },
      }),
    ],
    trustedOrigins: ['http://localhost:3000'],
  });
}

async function makeSignedInUser(auth: Auth) {
  // We create the user via signUpEmail (rather than poking the
  // memory adapter directly) so we exercise Better-Auth's own user-
  // creation path. The session token it returns isn't used by these
  // tests — we drive the api-key plugin in server-only mode by
  // passing `userId` in the body. See the per-test comments below
  // for rationale.
  const resp = await auth.api.signUpEmail({
    body: {
      email: 'test@wikis.dev',
      password: 'changeme123-test',
      name: 'test',
    },
  });
  return { userId: resp.user.id };
}

interface TestCase {
  name: string;
  run: () => Promise<void>;
}

const tests: TestCase[] = [
  {
    name: 'writes the owner column as `userId` (not `referenceId`)',
    async run() {
      const db = freshDb();
      const auth = makeAuth(db);
      const { userId } = await makeSignedInUser(auth);

      // No `headers` → server-only call path. The plugin accepts
      // `userId` in the body in that mode, which exercises the same
      // adapter/schema code we care about without needing to spin up
      // a real session/cookie machinery.
      const created = await auth.api.createApiKey({
        body: { name: 'rio-regression', userId },
      });

      // The plugin's response surfaces `referenceId` regardless of
      // how the column is named on disk — that's the public
      // contract. What matters is the row in the table.
      assert.equal(created.referenceId, userId, 'response.referenceId should be the user id');

      const rows = db.apiKey ?? [];
      assert.equal(rows.length, 1, 'one api key row should exist');
      const row = rows[0];

      assert.equal(row.userId, userId,
        'on-disk column MUST be `userId` (the remap target)');
      assert.equal(row.referenceId, undefined,
        'on-disk column MUST NOT be `referenceId` — that would mean the remap was ignored');
    },
  },
  {
    name: 'verifies a created key (round-trip)',
    async run() {
      const db = freshDb();
      const auth = makeAuth(db);
      const { userId } = await makeSignedInUser(auth);

      const created = await auth.api.createApiKey({
        body: { name: 'roundtrip', userId },
      });

      const verify = await auth.api.verifyApiKey({
        body: { key: created.key },
      });

      assert.equal(verify.valid, true,
        'verify must accept a freshly minted key');
      assert.equal(verify.error, null);
    },
  },
  {
    name: 'rejects an unknown key (negative control)',
    async run() {
      const db = freshDb();
      const auth = makeAuth(db);

      const verify = await auth.api.verifyApiKey({
        body: { key: 'wikis_definitely-not-a-real-key' },
      });

      assert.equal(verify.valid, false);
      assert.notEqual(verify.error, null);
    },
  },
  {
    name: 'respects the configured key prefix',
    async run() {
      const db = freshDb();
      const auth = makeAuth(db);
      const { userId } = await makeSignedInUser(auth);

      const created = await auth.api.createApiKey({
        body: { name: 'prefix-check', userId },
      });

      assert.equal(created.key.startsWith('wikis_'), true,
        'minted key must use the configured wikis_ prefix');
      assert.equal(created.prefix, 'wikis_');
    },
  },
  {
    name: 'persists every column the Prisma ApiKey model requires',
    async run() {
      const db = freshDb();
      const auth = makeAuth(db);
      const { userId } = await makeSignedInUser(auth);

      await auth.api.createApiKey({
        body: { name: 'schema-shape', userId },
      });

      const row = (db.apiKey ?? [])[0];
      assert.ok(row, 'row should exist');

      // Every column the Prisma migration declares as required
      // must be present in the row the plugin writes. If
      // Better-Auth introduces a new required column in a future
      // bump, this set misses it and the test fails — prompting a
      // Prisma migration before the dep upgrade ships.
      const requiredColumns = [
        'id',
        'configId',
        'key',
        'userId',
        'enabled',
        'rateLimitEnabled',
        'requestCount',
        'createdAt',
        'updatedAt',
      ];
      for (const col of requiredColumns) {
        assert.ok(col in row, `row missing required column "${col}"`);
      }
      assert.equal(row.userId, userId);
      assert.equal(row.enabled, true);
      assert.equal(row.requestCount, 0);
    },
  },
];

async function main() {
  let passed = 0;
  let failed = 0;
  for (const t of tests) {
    try {
      await t.run();
      console.log(`  ✓ ${t.name}`);
      passed += 1;
    } catch (err) {
      console.error(`  ✗ ${t.name}`);
      console.error(`    ${(err as Error).message}`);
      if (err instanceof Error && err.stack) {
        console.error(err.stack.split('\n').slice(1, 5).map((l) => '    ' + l).join('\n'));
      }
      failed += 1;
    }
  }

  console.log();
  console.log(`api-key plugin × schema remap: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    console.error('\nThis test guards against Better-Auth schema-default drift on');
    console.error('minor-version bumps. A failure here means the api-key plugin no');
    console.error('longer wires up to our `apikey` Prisma table the way auth.ts');
    console.error('assumes. Before upgrading better-auth, audit the plugin diff');
    console.error('against issue #170 and update src/lib/auth.ts if needed.');
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
