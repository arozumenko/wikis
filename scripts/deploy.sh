#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK=/tmp/kodan-deploy.lock
[ -f "$LOCK" ] && echo 'Deploy already running' && exit 0
trap 'rm -f $LOCK' EXIT
touch "$LOCK"

git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" != "$REMOTE" ]; then
  echo "Updating $LOCAL → $REMOTE"
  git pull origin main
  docker compose build
  docker compose up -d
  echo "Deployed $REMOTE at $(date)"
else
  echo "Already up to date ($LOCAL)"
fi
