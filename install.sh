#!/usr/bin/env bash
#
# Wikis — one-command installer
# Usage: curl -fsSL https://raw.githubusercontent.com/arozumenko/wikis/main/install.sh | bash
#
set -euo pipefail

RAW_URL="https://raw.githubusercontent.com/arozumenko/wikis/main"
INSTALL_DIR="${WIKIS_DIR:-./wikis}"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}▸${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}!${NC} $1"; }
err()   { echo -e "${RED}✗${NC} $1" >&2; }
header(){ echo -e "\n${BOLD}$1${NC}"; }

# ── Ensure we can read user input (works when piped via curl) ─────
if [ -t 0 ]; then
  INPUT_FD=0
else
  if ! exec 3</dev/tty 2>/dev/null; then
    err "Cannot open terminal for input. Run with: bash <(curl -fsSL URL)"
    exit 1
  fi
  INPUT_FD=3
fi

prompt() {
  local _prompt="$1" _var="$2"
  printf "%s" "$_prompt"
  IFS= read -r "$_var" <&${INPUT_FD}
}

prompt_secret() {
  local _prompt="$1" _var="$2"
  printf "%s" "$_prompt"
  IFS= read -rs "$_var" <&${INPUT_FD}
  echo ""
}

# ── Preflight ───────────────────────────────────────────────────────
header "🔍 Checking prerequisites..."

for cmd in docker openssl curl; do
  if ! command -v "$cmd" &>/dev/null; then
    err "$cmd is required but not installed."
    exit 1
  fi
done

if ! docker compose version &>/dev/null; then
  err "docker compose (v2) is required. Install Docker Desktop or the compose plugin."
  exit 1
fi

ok "All prerequisites found"

# ── Create install directory ──────────────────────────────────────
header "📦 Setting up Wikis..."

if [ -d "$INSTALL_DIR" ]; then
  warn "Directory $INSTALL_DIR already exists."
  prompt "   Overwrite? [y/N] " overwrite
  if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
    echo "Aborted."; exit 0
  fi
  rm -rf "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

info "Downloading docker-compose.yml..."
curl -fsSL "$RAW_URL/docker-compose.yml" -o docker-compose.yml
ok "Setup directory ready at $INSTALL_DIR"

# ── Port Configuration ────────────────────────────────────────────
header "🌐 Port Configuration"

echo ""
prompt "Web app port [3000]: " WEB_PORT
WEB_PORT="${WEB_PORT:-3000}"

prompt "Backend API port [8000]: " API_PORT
API_PORT="${API_PORT:-8000}"

if ! [[ "$WEB_PORT" =~ ^[0-9]+$ ]] || ! [[ "$API_PORT" =~ ^[0-9]+$ ]]; then
  err "Ports must be numbers."
  exit 1
fi

if [ "$WEB_PORT" -lt 1 ] || [ "$WEB_PORT" -gt 65535 ] || [ "$API_PORT" -lt 1 ] || [ "$API_PORT" -gt 65535 ]; then
  err "Ports must be between 1 and 65535."
  exit 1
fi

if lsof -i :"$WEB_PORT" &>/dev/null; then
  warn "Port $WEB_PORT is already in use."
  prompt "   Continue anyway? [y/N] " port_continue
  if [[ ! "$port_continue" =~ ^[Yy]$ ]]; then
    echo "Aborted. Re-run and choose a different port."; exit 0
  fi
fi

if lsof -i :"$API_PORT" &>/dev/null; then
  warn "Port $API_PORT is already in use."
  prompt "   Continue anyway? [y/N] " port_continue
  if [[ ! "$port_continue" =~ ^[Yy]$ ]]; then
    echo "Aborted. Re-run and choose a different port."; exit 0
  fi
fi

ok "Ports: web=$WEB_PORT, api=$API_PORT"

# ── LLM Provider ───────────────────────────────────────────────────
header "🤖 LLM Provider Configuration"

echo ""
echo "  1) OpenAI        (gpt-4o-mini, gpt-4o)"
echo "  2) Anthropic     (claude-sonnet-4-6, claude-haiku-4-5)"
echo "  3) Google Gemini (gemini-2.5-pro, gemini-2.0-flash)"
echo "  4) AWS Bedrock   (Claude, Titan — uses IAM or access keys)"
echo "  5) Azure OpenAI  (OpenAI-compatible endpoint)"
echo "  6) Ollama        (local, no API key needed)"
echo "  7) Skip          (configure later in .env)"
echo ""
prompt "Choose provider [1-7, default: 7]: " provider_choice

LLM_PROVIDER=""
LLM_API_KEY=""
LLM_MODEL=""
EMBEDDING_MODEL=""
AWS_REGION=""
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
LLM_API_BASE=""

case "${provider_choice:-7}" in
  1)
    LLM_PROVIDER="openai"
    LLM_MODEL="gpt-4o-mini"
    EMBEDDING_MODEL="text-embedding-3-large"
    prompt_secret "OpenAI API key: " LLM_API_KEY
    if [ -z "$LLM_API_KEY" ]; then
      warn "No API key provided. Edit .env later to add it."
    fi
    ;;
  2)
    LLM_PROVIDER="anthropic"
    LLM_MODEL="claude-sonnet-4-6"
    EMBEDDING_MODEL="text-embedding-3-large"
    prompt_secret "Anthropic API key: " LLM_API_KEY
    if [ -z "$LLM_API_KEY" ]; then
      warn "No API key provided. Edit .env later to add it."
    fi
    warn "Anthropic doesn't provide embeddings — set OPENAI_API_KEY in .env for embeddings, or use Ollama."
    ;;
  3)
    LLM_PROVIDER="gemini"
    LLM_MODEL="gemini-2.5-pro"
    EMBEDDING_MODEL="models/text-embedding-004"
    prompt_secret "Google AI API key: " LLM_API_KEY
    if [ -z "$LLM_API_KEY" ]; then
      warn "No API key provided. Edit .env later to add it."
    fi
    ;;
  4)
    LLM_PROVIDER="bedrock"
    LLM_MODEL="us.anthropic.claude-sonnet-4-6-20250514-v1:0"
    EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
    LLM_API_KEY="not-needed"
    prompt "AWS region [us-east-1]: " AWS_REGION
    AWS_REGION="${AWS_REGION:-us-east-1}"
    echo ""
    info "Bedrock can use IAM roles, instance profiles, or explicit keys."
    prompt "AWS access key ID (leave empty for IAM role): " AWS_ACCESS_KEY_ID
    if [ -n "$AWS_ACCESS_KEY_ID" ]; then
      prompt_secret "AWS secret access key: " AWS_SECRET_ACCESS_KEY
    else
      info "Using default AWS credential chain (IAM role, env vars, ~/.aws/credentials)."
    fi
    ;;
  5)
    LLM_PROVIDER="custom"
    echo ""
    info "Azure OpenAI uses an OpenAI-compatible endpoint."
    prompt "Azure OpenAI endpoint (e.g. https://YOUR.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT/): " LLM_API_BASE
    prompt_secret "Azure OpenAI API key: " LLM_API_KEY
    prompt "Model deployment name [gpt-4o-mini]: " LLM_MODEL
    LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
    EMBEDDING_MODEL="text-embedding-3-large"
    if [ -z "$LLM_API_BASE" ] || [ -z "$LLM_API_KEY" ]; then
      warn "Missing endpoint or key. Edit .env later to complete setup."
    fi
    ;;
  6)
    LLM_PROVIDER="ollama"
    LLM_MODEL="llama3.2"
    EMBEDDING_MODEL="nomic-embed-text"
    LLM_API_KEY="not-needed"
    info "Make sure Ollama is running: ollama pull llama3.2 && ollama pull nomic-embed-text"
    ;;
  7|"")
    info "Skipping LLM config — edit .env before generating wikis."
    ;;
  *)
    warn "Invalid choice, skipping LLM config — edit .env later."
    ;;
esac

# ── Generate JWT keys ──────────────────────────────────────────────
header "🔐 Generating authentication keys..."

KEYS_DIR="$(pwd)/.keys"
mkdir -p "$KEYS_DIR"
openssl genrsa -out "$KEYS_DIR/private.pem" 2048 2>/dev/null
openssl rsa -in "$KEYS_DIR/private.pem" -pubout -out "$KEYS_DIR/public.pem" 2>/dev/null

JWT_PRIVATE_KEY=$(cat "$KEYS_DIR/private.pem")
JWT_PUBLIC_KEY=$(cat "$KEYS_DIR/public.pem")
AUTH_SECRET=$(openssl rand -hex 32)

ok "RS256 key pair generated"

# ── Patch docker-compose ports if non-default ─────────────────────
if [ "$WEB_PORT" != "3000" ] || [ "$API_PORT" != "8000" ]; then
  info "Updating docker-compose.yml with custom ports..."
  if [ "$API_PORT" != "8000" ]; then
    sed -i.bak "s/\"8000:8000\"/\"${API_PORT}:8000\"/" docker-compose.yml
  fi
  if [ "$WEB_PORT" != "3000" ]; then
    sed -i.bak "s/\"3000:3000\"/\"${WEB_PORT}:3000\"/" docker-compose.yml
  fi
  rm -f docker-compose.yml.bak
fi

# ── Write .env ─────────────────────────────────────────────────────
header "📝 Writing configuration..."

cat > .env <<ENVFILE
# ═══════════════════════════════════════════════════════════════════
# Wikis Configuration — generated by install.sh
# ═══════════════════════════════════════════════════════════════════

# LLM
LLM_PROVIDER=${LLM_PROVIDER}
LLM_API_KEY=${LLM_API_KEY}
LLM_MODEL=${LLM_MODEL}
EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENVFILE

# Conditional provider-specific vars
if [ -n "$LLM_API_BASE" ]; then
  echo "LLM_API_BASE=${LLM_API_BASE}" >> .env
fi

if [ "$LLM_PROVIDER" = "bedrock" ]; then
  cat >> .env <<ENVFILE

# AWS (Bedrock)
AWS_REGION=${AWS_REGION}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENVFILE
fi

cat >> .env <<ENVFILE

# Storage
STORAGE_BACKEND=local
STORAGE_PATH=/app/data/artifacts
CACHE_DIR=/app/data/cache

# Auth
BETTER_AUTH_URL=http://localhost:${WEB_PORT}
AUTH_SECRET=${AUTH_SECRET}
# Uncomment to enable OAuth:
# GITHUB_CLIENT_ID=
# GITHUB_CLIENT_SECRET=
# GOOGLE_CLIENT_ID=
# GOOGLE_CLIENT_SECRET=

# JWT (cross-service auth)
JWT_PRIVATE_KEY="${JWT_PRIVATE_KEY}"
JWT_PUBLIC_KEY="${JWT_PUBLIC_KEY}"

# Internal
BACKEND_URL=http://backend:8000
WIKIS_BACKEND_URL=http://localhost:${API_PORT}
LOG_LEVEL=INFO
NODE_ENV=production
ENVFILE

chmod 600 .env
ok "Configuration written to .env"

# ── Start services ─────────────────────────────────────────────────
header "🚀 Starting Wikis..."

if ! docker compose pull; then
  warn "Failed to pull images. Will try to start with cached images..."
fi
docker compose up -d

info "Waiting for services to be healthy..."
attempt=0
max_attempts=30
while [ $attempt -lt $max_attempts ]; do
  if curl -sf http://localhost:${WEB_PORT} >/dev/null 2>&1; then
    break
  fi
  attempt=$((attempt + 1))
  sleep 2
done

if [ $attempt -eq $max_attempts ]; then
  warn "Services are still starting. Check: docker compose ps"
else
  ok "Services are running"
fi

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Wikis is ready!${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}App:${NC}       http://localhost:${WEB_PORT}"
echo -e "  ${BOLD}API:${NC}       http://localhost:${API_PORT}/docs"
echo -e "  ${BOLD}Login:${NC}     admin@wikis.dev / changeme123"
echo ""
echo -e "  ${YELLOW}Change the default password in Settings > Account after first login.${NC}"
echo ""
echo -e "  Manage:  cd $INSTALL_DIR"
echo -e "           docker compose ps      # status"
echo -e "           docker compose logs -f  # logs"
echo -e "           docker compose down     # stop"
echo ""
