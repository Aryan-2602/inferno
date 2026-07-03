# INFERNO — KV Cache Compression & Continuous Batching Engine

## Project Overview
Inferno is a from-scratch LLM inference optimization engine implementing KV cache 
quantization and continuous batching on top of HuggingFace Transformers. 
Every performance claim must be measured and reproducible. No theoretical numbers.

## Goals
- Phase 1: Baseline benchmark + INT8 KV cache quantization with correctness validation
- Phase 2: Continuous batching with chunk-size tuning
- Phase 3: GPU validation on rented hardware (RunPod/Lambda)

## Environment
- Python 3.12
- Virtual environment: venv (always activate before running anything)
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Model: Qwen2.5-0.5B (primary), Llama-3.2-1B (secondary validation)

## Project Structure
inferno/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── inferno/
│   ├── __init__.py
│   ├── baseline.py        # HuggingFace naive generate() baseline
│   ├── cache.py           # KV cache quantization (INT8 first, INT4 later)
│   ├── engine.py          # Continuous batching scheduler
│   └── utils.py           # Timing, memory measurement, logging helpers
├── benchmarks/
│   ├── bench_baseline.py  # Baseline throughput + memory benchmark
│   ├── bench_cache.py     # Quantized cache vs baseline benchmark
│   └── bench_batching.py  # Continuous batching throughput benchmark
├── tests/
│   ├── test_cache.py      # Correctness tests for quantization
│   ├── test_engine.py     # Scheduler logic tests
│   └── test_baseline.py   # Sanity checks on baseline
└── results/
    └── (benchmark outputs go here as JSON + charts)

## Coding Standards
- Type hints on every function signature
- Docstring on every function explaining what it does and why
- No magic numbers — all hyperparameters as named constants at top of file
- Every benchmark script saves results to results/ as JSON before printing
- Never delete a benchmark result — append with timestamp

## Testing Rules
- Write tests alongside each module, not after
- Tests must run on CPU with no GPU required
- Each test must have a clear docstring explaining what correctness property it checks
- Run tests with: `pytest tests/ -v`

## Benchmarking Rules
- Always measure: tokens/sec, peak memory (MB), and perplexity where applicable
- Always run baseline and optimized in the same script for fair comparison
- Print a comparison table at the end of every benchmark run
- Save raw results as JSON to results/ with timestamp

## Claude Code Behavior
- Read and understand the relevant module fully before editing it
- Write tests for each function before or immediately after implementing it
- After completing any phase, print a summary of what was built and what the tests cover
- Never skip a test because it is hard to write — flag it and explain why instead
- When implementing math (quantization, attention), add a comment explaining the formula
- If a design decision has a tradeoff, add a # TRADEOFF: comment explaining it