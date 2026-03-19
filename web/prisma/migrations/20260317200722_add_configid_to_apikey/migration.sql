-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_apikey" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "configId" TEXT NOT NULL DEFAULT 'default',
    "name" TEXT,
    "start" TEXT,
    "prefix" TEXT,
    "key" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "refillAmount" INTEGER,
    "refillInterval" INTEGER,
    "lastRefillAt" DATETIME,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "rateLimitEnabled" BOOLEAN NOT NULL DEFAULT false,
    "rateLimitTimeWindow" INTEGER,
    "rateLimitMax" INTEGER,
    "requestCount" INTEGER NOT NULL DEFAULT 0,
    "remaining" INTEGER,
    "lastRequest" DATETIME,
    "expiresAt" DATETIME,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    "permissions" TEXT,
    "metadata" TEXT,
    CONSTRAINT "apikey_userId_fkey" FOREIGN KEY ("userId") REFERENCES "user" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);
INSERT INTO "new_apikey" ("createdAt", "enabled", "expiresAt", "id", "key", "lastRefillAt", "lastRequest", "metadata", "name", "permissions", "prefix", "rateLimitEnabled", "rateLimitMax", "rateLimitTimeWindow", "refillAmount", "refillInterval", "remaining", "requestCount", "start", "updatedAt", "userId") SELECT "createdAt", "enabled", "expiresAt", "id", "key", "lastRefillAt", "lastRequest", "metadata", "name", "permissions", "prefix", "rateLimitEnabled", "rateLimitMax", "rateLimitTimeWindow", "refillAmount", "refillInterval", "remaining", "requestCount", "start", "updatedAt", "userId" FROM "apikey";
DROP TABLE "apikey";
ALTER TABLE "new_apikey" RENAME TO "apikey";
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
