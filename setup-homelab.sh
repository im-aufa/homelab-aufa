#!/bin/bash
# setup-homelab.sh — One-time setup on the HP Pavilion homelab server.
# Run this ONCE on the homelab, not on your Mac.
# Usage: ssh homelab 'bash -s' < setup-homelab.sh

set -e

echo "📦 Installing Docker..."
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"

echo "✅ Docker installed. You may need to log out and back in for group changes."
echo ""

echo "📁 Creating project directory..."
mkdir -p ~/aufaim-portfolio

echo ""
echo "✅ Homelab setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy your .env file to the homelab:"
echo "     scp /path/to/your/.env homelab:~/aufaim-portfolio/.env"
echo ""
echo "  2. Run deploy.sh from your Mac:"
echo "     chmod +x deploy.sh && ./deploy.sh"
echo ""
echo "  3. (Optional) Set up tailscale serve for HTTPS:"
echo "     tailscale serve https / http://localhost:80"
echo "     → Your site will be at https://homelab.[tailnet].ts.net"
echo ""
echo "  4. Load your PDFs into the RAG:"
echo "     - Copy PDFs to the homelab: scp your-docs/*.pdf homelab:~/"
echo "     - Move them into the docker volume:"
echo "       ssh homelab 'docker cp ~/your-doc.pdf aufaim-portfolio-chat-api-1:/app/data/'"
echo "     - Trigger processing via the API (after logging in):"
echo "       POST http://homelab/api/process  body: {\"local_folder_path\": \"/app/data\"}"
