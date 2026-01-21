"""
Performance and benchmarking tests.
Track response times, throughput, and batch processing efficiency.
"""

import time

import pytest
from config import OPENAI_MODEL
from helpers import classify_batch, classify_with_tokens, measure_latencies


@pytest.mark.parametrize("size", [5, 10, 20])
def test_response_time_benchmark(openai_client, sentiment_dataset, size):
    """Benchmark response times at different test sizes"""

    latencies = measure_latencies(openai_client, OPENAI_MODEL, sentiment_dataset[:size])
    elapsed = sum(latencies)
    avg_time = elapsed / size

    assert avg_time < 3.0, f"size={size}: avg {avg_time:.3f}s > 3.0s"


def test_batch_vs_sequential_performance(openai_client, sentiment_dataset):
    """Compare batch vs sequential processing performance"""

    test_cases = sentiment_dataset[:10]

    # Sequential processing
    sequential_latencies = measure_latencies(openai_client, OPENAI_MODEL, test_cases)
    sequential_time = sum(sequential_latencies)

    # Batch processing
    start_batch = time.time()
    classify_batch(openai_client, OPENAI_MODEL, test_cases, batch_size=10)
    batch_time = time.time() - start_batch

    speedup = sequential_time / batch_time

    assert batch_time < sequential_time, "Batch should be faster"
    assert speedup >= 3.0, f"Speedup {speedup:.2f}x < 3x"


def test_token_usage_tracking(openai_client, sentiment_dataset):
    """Verify token usage stays within expected bounds"""

    tokens = classify_with_tokens(openai_client, OPENAI_MODEL, sentiment_dataset[:10])

    total_tokens = tokens["input_tokens"] + tokens["output_tokens"]
    avg_total = total_tokens / 10

    assert avg_total < 50, f"Token usage too high: {avg_total:.1f}"
