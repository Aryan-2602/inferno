"""
bench_baseline.py — Baseline throughput and memory benchmark for Inferno.

Runs HuggingFace generate() on the configured model and measures tokens/sec
and peak memory (MB). Results are saved to results/ as JSON with a timestamp
before being printed as a summary table.
"""
