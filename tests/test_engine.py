"""
test_engine.py — Scheduler logic tests for the continuous batching engine in Inferno.

Verifies that the scheduler correctly enqueues requests, groups them into batches
of the configured chunk size, and drains the queue without dropping sequences.
All tests run on CPU with synthetic data; no GPU required.
"""
