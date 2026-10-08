#!/bin/bash
# deploy.sh — Push this project from your Mac to the homelab and restart services.
# Usage: ./deploy.sh
#
# Prerequisites:
#   - Tailscale running on both Mac and homelab (hostname: homelab)
#   - SSH access to homelab (ssh homelab works)
#   - Docker + Docker Compose installed on homelab (run setup-homelab.sh first)

set -e

REMOTE_HOST="aufaa@homelab"
REMOTE_DIR="/mnt/data/projects/cerina"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔄 Syncing files to homelab..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude 'traefik/acme.json' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$LOCAL_DIR/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "🚀 Restarting services on homelab..."
ssh "$REMOTE_HOST" "cd ${REMOTE_DIR} && docker compose down && docker compose up -d --build"

echo ""
echo "✅ Deployed! Check status with:"
echo "   ssh homelab 'docker compose -f ~/aufaim-portfolio/docker-compose.yaml ps'"
echo ""
echo "🌐 Access the site at: http://homelab (or your Tailscale IP)"
echo "   Run 'tailscale ip -4' on the homelab to get the IP if needed."
