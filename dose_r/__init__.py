"""DOSE-R: replication and extension harness for the Synthio DOSE benchmark.

Workstream 2 owns the *runner* layer: adapters, latency/cost capture, and the
tradeoff blueprint. The scoring layer is a pluggable interface that Workstream 1's
hybrid judge (phonetic-distance + audio-LLM panel) drops into.
"""

__version__ = "0.1.0"
