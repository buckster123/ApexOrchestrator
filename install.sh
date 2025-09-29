#!/bin/bash

# install.sh: Auto-install for Apex Orchestrator on fresh Raspberry Pi OS Bookworm (Pi-5)
# Run as: sudo bash install.sh (from project dir)
# Logs to install.log

set -e  # Exit on error
LOGFILE="install.log"
exec > >(tee -a ${LOGFILE})
exec 2> >(tee -a ${LOGFILE} >&2)

echo "=== Apex Orchestrator Install Script ==="
echo "Date: $(date)"
echo "Starting on: $(uname -a)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if running on Pi-5/Bookworm
if ! grep -q "Raspberry Pi 5" /proc/cpuinfo; then
    warn "Not detected as Pi-5; proceeding anyway (ensure 64-bit Bookworm)."
fi
if ! grep -q "bookworm" /etc/os-release; then
    warn "Not Bookworm; some deps may fail."
fi

# Step 1: System Update
log "Updating system..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install apt dependencies
log "Installing apt packages..."
sudo apt install -y \
    build-essential \
    cmake \
    libgit2-dev \
    python3-pip \
    python3-venv \
    git \
    clang-tools \
    golang-go \
    php-cli \
    composer \
    curl \
    ntp \
    || error "Apt install failed. Check log."

# Step 3: Install php-cs-fixer
log "Installing php-cs-fixer..."
sudo composer global require friendsofphp/php-cs-fixer || warn "Composer global failed; PHP linting disabled."

# Step 4: Install Rust (for rustfmt)
log "Installing Rust (rustup)..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup component add rustfmt || warn "Rustfmt add failed; Rust linting disabled."

# Step 5: Create virtual environment
log "Setting up Python venv..."
python3 -m venv venv
source venv/bin/activate

# Step 6: Install pip dependencies (with torch CPU for Pi-5)
log "Installing pip packages..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only for ARM64
pip install \
    python-dotenv \
    openai \
    passlib \
    pygit2 \
    requests \
    streamlit \
    pyyaml \
    beautifulsoup4 \
    black \
    chromadb \
    jsbeautifier \
    ntplib \
    numpy \
    sentence-transformers \
    sqlparse \
    tiktoken \
    || error "Pip install failed. Torch may need retry."

# Generate requirements.txt
pip freeze > requirements.txt
log "requirements.txt generated."

# Step 7: Project setup
log "Setting up project directories..."
mkdir -p prompts sandbox chroma_db
touch chatapp.db .env

# Create .env template
cat > .env << EOF
# Add your keys here
XAI_API_KEY=your_xai_key_here
LANGSEARCH_API_KEY=your_langsearch_key_here
EOF
warn "Edit .env with your API keys before running!"

# Step 8: Test install
log "Running quick test..."
source venv/bin/activate
python -c "import streamlit; print('Streamlit OK')" || warn "Streamlit import failed."
python -c "from sentence_transformers import SentenceTransformer; print('Embeddings OK')" || warn "Embeddings failed (first load slow)."
python -c "import chromadb; print('ChromaDB OK')" || warn "ChromaDB failed."

# Step 9: Run script
log "Install complete! Run with: source venv/bin/activate && streamlit run your_script.py"
log "Activate venv: source venv/bin/activate"
log "Deactivate: deactivate"

echo "=== Install finished. Check ${LOGFILE} for details. ==="
