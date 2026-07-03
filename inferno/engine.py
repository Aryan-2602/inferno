"""
engine.py — Continuous batching scheduler for Inferno.

Implements a scheduler that dynamically groups in-flight requests into batches,
inserting new sequences mid-generation to maximize GPU utilization.
Chunk size is a tunable parameter; throughput is measured across chunk-size sweeps.
"""
