#!/bin/sh
# Patches schema.prisma datasource based on DATABASE_URL and selects the
# correct migrations directory for the target provider.
#
# SQLite (default) when DATABASE_URL is absent or starts with file:
# PostgreSQL when DATABASE_URL starts with postgresql:// or postgres://

set -e

cd "$(dirname "$0")/.."

SCHEMA="prisma/schema.prisma"
DATABASE_URL="${DATABASE_URL:-file:../data/users.db}"

if echo "$DATABASE_URL" | grep -q "^postgresql://\|^postgres://"; then
  PROVIDER="postgresql"
else
  PROVIDER="sqlite"
fi

# Patch only the datasource block (lines between "datasource db {" and "}")
# This avoids matching url fields in model definitions.
if [ "$PROVIDER" = "postgresql" ]; then
  sed -i.bak \
    '/^datasource db {/,/^}/ {
      s|provider = "sqlite"|provider = "postgresql"|
      s|url.*=.*"file:.*"|url      = env("DATABASE_URL")|
    }' "$SCHEMA"
  # Use PostgreSQL migrations
  if [ -d prisma/migrations-pg ]; then
    rm -rf prisma/migrations-active
    cp -r prisma/migrations-pg prisma/migrations-active
  fi
else
  sed -i.bak \
    '/^datasource db {/,/^}/ {
      s|provider = "postgresql"|provider = "sqlite"|
      s|url.*=.*env("DATABASE_URL")|url      = "file:../data/users.db"|
    }' "$SCHEMA"
  # Use SQLite migrations
  if [ -d prisma/migrations ]; then
    rm -rf prisma/migrations-active
    cp -r prisma/migrations prisma/migrations-active
  fi
fi

rm -f "${SCHEMA}.bak"

echo "Prisma provider: $PROVIDER"
