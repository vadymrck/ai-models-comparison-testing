"""
Performance and benchmarking tests.
Track response times, throughput, and regression detection.
"""

import json
import os
import time
from datetime import datetime

import pytest
from config import OPENAI_MODEL, OPENAI_MODEL_COMPARE
from helpers import calculate_cost, call_with_delay, classify_sentiment


def test_response_time_benchmark(openai_client, sentiment_dataset):
    """Benchmark response times across different test sizes"""

    test_sizes = [5, 10, 20]
    results = {}

    for size in test_sizes:
        test_cases = sentiment_dataset[:size]
        start_time = time.time()

        for case in test_cases:
            classify_sentiment(openai_client, OPENAI_MODEL, case["text"])

        elapsed = time.time() - start_time
        avg_time = elapsed / size

        results[f"{size}_samples"] = {
            "total_time": elapsed,
            "avg_time_per_request": avg_time,
            "throughput_per_minute": 60 / avg_time,
        }

    # Save benchmark results
    os.makedirs("reports", exist_ok=True)
    benchmark_file = "reports/performance_benchmark.json"

    benchmark_data = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENAI_MODEL,
        "results": results,
    }

    with open(benchmark_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    avg_time = results["10_samples"]["avg_time_per_request"]
    assert avg_time < 3.0, f"Response time {avg_time:.3f}s > 3.0s"


def test_batch_vs_sequential_performance(openai_client, sentiment_dataset):
    """Compare batch vs sequential processing performance"""

    test_cases = sentiment_dataset[:10]

    # Sequential processing
    start_sequential = time.time()
    for case in test_cases:
        classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
    sequential_time = time.time() - start_sequential

    # Batch processing
    start_batch = time.time()
    batch_prompt = "Classify each review as positive, negative, or neutral. Respond with only the labels separated by commas.\n\n"
    for i, case in enumerate(test_cases, 1):
        batch_prompt += f"{i}. {case['text']}\n"
    batch_prompt += "\nLabels (comma-separated):"

    call_with_delay(
        openai_client,
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": batch_prompt}],
        temperature=0,
    )
    batch_time = time.time() - start_batch

    speedup = sequential_time / batch_time

    assert batch_time < sequential_time, "Batch should be faster"
    assert speedup >= 3.0, f"Speedup {speedup:.2f}x < 3x"


def test_model_regression_detection(openai_client, sentiment_dataset):
    """Detect regression in model performance"""

    test_cases = sentiment_dataset[:15]
    predictions = []
    ground_truth = []

    for case in test_cases:
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])

    correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    current_accuracy = correct / len(test_cases)

    # Save results
    results_file = "reports/regression_tracking.json"
    current_result = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENAI_MODEL,
        "accuracy": current_accuracy,
        "test_size": len(test_cases),
    }

    history = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            history = json.load(f)

    history.append(current_result)
    history = history[-10:]

    os.makedirs("reports", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(history, f, indent=2)

    if len(history) > 1:
        previous_accuracy = history[-2]["accuracy"]
        diff = current_accuracy - previous_accuracy
        assert diff > -0.15, f"Regression: {current_accuracy:.1%} vs {previous_accuracy:.1%}"


def test_concurrent_requests_handling(openai_client, sentiment_dataset):
    """Test handling of concurrent requests"""

    import concurrent.futures

    test_cases = sentiment_dataset[:10]

    def classify_single(case):
        """Helper for concurrent classification"""
        return classify_sentiment(openai_client, OPENAI_MODEL, case["text"])

    start_time = time.time()

    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        predictions = list(executor.map(classify_single, test_cases))

    concurrent_time = time.time() - start_time

    # Compare with sequential
    start_sequential = time.time()
    sequential_predictions = [classify_single(case) for case in test_cases]
    sequential_time = time.time() - start_sequential

    speedup = sequential_time / concurrent_time

    # Verify results are still accurate
    ground_truth = [case["label"] for case in test_cases]
    concurrent_correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    concurrent_accuracy = concurrent_correct / len(test_cases)

    sequential_correct = sum(
        1 for p, t in zip(sequential_predictions, ground_truth) if p == t
    )
    sequential_accuracy = sequential_correct / len(test_cases)

    # Concurrent should be meaningfully faster and maintain accuracy
    assert concurrent_time < sequential_time, "Concurrent should be faster"
    assert (
        speedup >= 1.5
    ), f"Insufficient speedup: {speedup:.2f}x (expected at least 1.5x with 3 workers)"
    assert (
        concurrent_accuracy >= 0.60
    ), f"Concurrent accuracy too low: {concurrent_accuracy:.1%}"
    assert (
        concurrent_accuracy >= sequential_accuracy * 0.9
    ), f"Concurrent accuracy degraded: {concurrent_accuracy:.1%} vs sequential {sequential_accuracy:.1%}"


def test_token_usage_tracking(openai_client, sentiment_dataset):
    """Track token usage across tests"""

    test_cases = sentiment_dataset[:10]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for case in test_cases:
        _, response = classify_sentiment(
            openai_client, OPENAI_MODEL, case["text"], return_raw_response=True
        )

        # Extract token usage
        usage = response.usage
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        total_tokens += usage.total_tokens

    avg_total = total_tokens / len(test_cases)

    # Calculate cost using helper
    costs = calculate_cost(total_prompt_tokens, total_completion_tokens, OPENAI_MODEL)

    # Save token tracking
    token_data = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENAI_MODEL,
        "test_size": len(test_cases),
        "total_tokens": total_tokens,
        "avg_tokens_per_request": avg_total,
        "total_cost": costs["total_cost"],
        "cost_per_request": costs["total_cost"] / len(test_cases),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/token_usage.json", "w") as f:
        json.dump(token_data, f, indent=2)

    # Assert reasonable token usage
    assert avg_total < 50, f"Token usage too high: {avg_total:.1f}"


def test_error_rate_monitoring(openai_client, sentiment_dataset):
    """Monitor error rates and failures"""

    test_cases = sentiment_dataset[:20]

    errors = []
    successful = 0
    failed = 0

    for i, case in enumerate(test_cases):
        try:
            prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])

            # Check if prediction is valid
            if prediction not in ["positive", "negative", "neutral"]:
                errors.append(
                    {
                        "index": i,
                        "text": case["text"][:50],
                        "error": "Invalid prediction format",
                        "prediction": prediction,
                    }
                )
                failed += 1
            else:
                successful += 1

        except Exception as e:
            errors.append({"index": i, "text": case["text"][:50], "error": str(e)})
            failed += 1

    error_rate = failed / len(test_cases)

    # Save error tracking
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENAI_MODEL,
        "total_tests": len(test_cases),
        "successful": successful,
        "failed": failed,
        "error_rate": error_rate,
        "errors": errors,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/error_tracking.json", "w") as f:
        json.dump(error_data, f, indent=2)

    # Format error details for assertion message
    error_details = ""
    if errors:
        error_details = "\n" + "\n".join(
            f"  [{e['index']}] {e['error']}: {e['text']}..." for e in errors[:5]
        )

    # Error rate should be very low
    assert error_rate < 0.05, f"Error rate {error_rate:.1%}{error_details}"


@pytest.mark.skip(reason="Dataset too small for meaningful model comparison")
def test_model_version_comparison(openai_client, sentiment_dataset):
    """Compare different model versions automatically"""

    models = [OPENAI_MODEL, OPENAI_MODEL_COMPARE]

    test_cases = sentiment_dataset[:10]
    results = {}

    for model in models:
        predictions = []
        ground_truth = []
        start_time = time.time()

        for case in test_cases:
            prediction = classify_sentiment(openai_client, model, case["text"])
            predictions.append(prediction)
            ground_truth.append(case["label"])

        elapsed = time.time() - start_time

        correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
        accuracy = correct / len(test_cases)

        results[model] = {
            "accuracy": accuracy,
            "time": elapsed,
            "avg_time": elapsed / len(test_cases),
        }

    # Save comparison
    comparison_data = {"timestamp": datetime.now().isoformat(), "models": results}

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_comparison_automated.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Both models should meet minimum standards
    for model, metrics in results.items():
        assert (
            metrics["accuracy"] >= 0.75
        ), f"{model} accuracy {metrics['accuracy']:.1%} < 75%"

        assert (
            metrics["avg_time"] < 3.0
        ), f"{model} avg time {metrics['avg_time']:.2f}s > 3s"
