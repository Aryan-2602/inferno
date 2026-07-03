"""
bench_batching.py — Continuous batching throughput benchmark for Inferno.

Measures throughput (tokens/sec) of the continuous batching engine across
a sweep of chunk sizes. Runs baseline sequential generation alongside for
comparison. Saves results to results/ as JSON and prints a comparison table.
"""
