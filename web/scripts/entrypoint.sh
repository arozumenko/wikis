#!/bin/sh
# Startup script: prepare Prisma for the target DB, run migrations, start app.
set -e

cd /app

# 1. Patch schema.prisma for the correct provider
./scripts/prisma-prepare.sh

# 2. Generate Prisma client for the detected provider
npx prisma generate

# 3. Run migrations using the provider-specific migration set
if [ -d prisma/migrations-active ]; then
  # Temporarily swap in the correct migrations directory
  mv prisma/migrations prisma/migrations-sqlite-orig
  cp -r prisma/migrations-active prisma/migrations
  npx prisma migrate deploy 2>/dev/null || true
  rm -rf prisma/migrations
  mv prisma/migrations-sqlite-orig prisma/migrations
else
  npx prisma migrate deploy 2>/dev/null || true
fi

# 4. Seed default admin user (skips if users already exist)
npx prisma db seed 2>/dev/null || true

# 5. Start the app
exec npm start
