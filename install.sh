#!/bin/bash

# Install script for ApexOrchestrator on Raspberry Pi 5 (Bookworm)

set -e  # Exit on error

echo "Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3-dev python3-pip python3-venv libgit2-dev libatlas-base-dev clang-format golang-go rustc cargo php-cli curl php-cs-fixer

# Optional php-cs-fixer via composer if apt fails
if ! command -v php-cs-fixer &> /dev/null; then
  echo "Installing php-cs-fixer via composer..."
  curl -sS https://getcomposer.org/installer | php
  php composer.phar global require friendsofphp/php-cs-fixer
fi

echo "Cloning repository..."
git clone https://github.com/buckster123/ApexOrchestrator.git || true
cd ApexOrchestrator

echo "Creating fresh virtual environment..."
rm -rf venv # Remove old venv if exists (caution: destructive)
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip and installing wheel..."
pip install --upgrade pip wheel

echo "Installing pip dependencies..."
pip install python-dotenv beautifulsoup4 black openai passlib sentence-transformers chromadb jsbeautifier ntplib numpy pygit2 requests sqlparse streamlit tiktoken pyyaml tqdm ecdsa scipy pandas matplotlib sympy mpmath statsmodels PuLP astropy qutip control biopython pubchempy dendropy rdkit pyscf pygame chess mido midiutil networkx torch python-snappy

echo "Setup complete! Add your API keys to .env and run: streamlit run chat_mk3.py"
echo "For agent isolation, create separate venvs: python3 -m venv venv-apexcoder && ... (repeat for each agent; install deps in each)"
