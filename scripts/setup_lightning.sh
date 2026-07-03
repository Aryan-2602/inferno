#!/bin/bash
# Run this first thing inside Lightning Studio
git clone https://github.com/YOUR_USERNAME/inferno.git
cd inferno
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install datasets  # for wikitext-2 perplexity eval
python scripts/check_gpu.py  # confirm GPU is healthy before benchmarking
