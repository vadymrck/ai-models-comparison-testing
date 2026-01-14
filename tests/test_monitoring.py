"""
Performance and benchmarking tests.
Track response times, throughput, and regression detection.
"""

import pytest
import time
import json
import os
from datetime import datetime
from config import OPENAI_MODEL, OPENAI_MODEL_COMPARE
from helpers import call_with_delay, classify_sentiment


def test_response_time_benchmark(openai_client, sentiment_dataset):
    """Benchmark response times across different test sizes"""

    print("\n  ⚡ Running response time benchmark...\n")

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

        print(
            f"  {size:2} samples: {elapsed:.2f}s total, {avg_time:.3f}s avg, {60/avg_time:.1f} req/min"
        )

    # Save benchmark results
    os.makedirs("reports", exist_ok=True)
    benchmark_file = "reports/performance_benchmark.json"

    benchmark_data = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENAI_MODEL,
        "results": results,
    }

    # Load previous results if they exist
    previous_results = None
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            previous_results = json.load(f)

    # Save current results
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n  💾 Benchmark saved to: {benchmark_file}")

    # Compare with previous results
    if previous_results:
        print("\n  📊 Comparison with previous run:")
        prev_avg = previous_results["results"]["10_samples"]["avg_time_per_request"]
        curr_avg = results["10_samples"]["avg_time_per_request"]
        diff_pct = ((curr_avg - prev_avg) / prev_avg) * 100

        if diff_pct > 0:
            print(f"  ⚠️  Slower by {diff_pct:.1f}% ({prev_avg:.3f}s → {curr_avg:.3f}s)")
        else:
            print(
                f"  ✓ Faster by {abs(diff_pct):.1f}% ({prev_avg:.3f}s → {curr_avg:.3f}s)"
            )

    # Assert reasonable performance
    assert results["10_samples"]["avg_time_per_request"] < 3.0, "Response time too slow"

    print("\n✅ PASSED - Benchmark complete")


def test_batch_vs_sequential_performance(openai_client, sentiment_dataset):
    """Compare batch vs sequential processing performance"""

    print("\n  🏁 Comparing batch vs sequential processing...\n")

    test_cases = sentiment_dataset[:10]

    # Sequential processing
    print("  Testing sequential processing...")
    start_sequential = time.time()

    for case in test_cases:
        classify_sentiment(openai_client, OPENAI_MODEL, case["text"])

    sequential_time = time.time() - start_sequential

    # Batch processing
    print("  Testing batch processing...")
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

    print("\n  📊 Results:")
    print(
        f"  Sequential: {sequential_time:.2f}s ({sequential_time/len(test_cases):.3f}s per item)"
    )
    print(
        f"  Batch:      {batch_time:.2f}s ({batch_time/len(test_cases):.3f}s per item)"
    )
    print(f"  Speedup:    {speedup:.2f}x faster")

    # Batch should be significantly faster (at least 3x)
    assert batch_time < sequential_time, "Batch processing should be faster"
    assert (
        speedup >= 3.0
    ), f"Batch processing not efficient enough: only {speedup:.2f}x faster (expected at least 3x)"

    print(f"\n✅ PASSED - Batch processing is {speedup:.1f}x faster")


def test_model_regression_detection(openai_client, sentiment_dataset):
    """Detect regression in model performance"""

    print("\n  🔍 Running regression detection...\n")

    test_cases = sentiment_dataset[:15]

    predictions = []
    ground_truth = []

    for case in test_cases:
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])

    # Calculate current accuracy
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

    # Load history
    history = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            history = json.load(f)

    # Add current result
    history.append(current_result)

    # Keep only last 10 runs
    history = history[-10:]

    # Save updated history
    os.makedirs("reports", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Current accuracy: {current_accuracy:.1%}")

    # Check for regression
    if len(history) > 1:
        previous_accuracy = history[-2]["accuracy"]
        diff = current_accuracy - previous_accuracy

        print(f"  Previous accuracy: {previous_accuracy:.1%}")
        print(f"  Change: {diff:+.1%}")

        if diff < -0.10:  # More than 10% drop
            print("  ⚠️  WARNING: Significant regression detected!")
        elif diff > 0:
            print("  ✓ Improvement detected")
        else:
            print("  ✓ Stable performance")

        # Fail if major regression
        assert diff > -0.15, f"Major regression: {diff:.1%} drop"

    print("\n✅ PASSED - No major regression detected")


def test_concurrent_requests_handling(openai_client, sentiment_dataset):
    """Test handling of concurrent requests"""

    print("\n  🔀 Testing concurrent request handling...\n")

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

    print(f"  📊 Concurrent: {concurrent_time:.2f}s")
    print(f"  📊 Sequential: {sequential_time:.2f}s")
    print(f"  📊 Speedup: {speedup:.2f}x")

    # Verify results are still accurate
    ground_truth = [case["label"] for case in test_cases]
    concurrent_correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    concurrent_accuracy = concurrent_correct / len(test_cases)

    sequential_correct = sum(
        1 for p, t in zip(sequential_predictions, ground_truth) if p == t
    )
    sequential_accuracy = sequential_correct / len(test_cases)

    print(f"  ✓ Concurrent accuracy: {concurrent_accuracy:.1%}")
    print(f"  ✓ Sequential accuracy: {sequential_accuracy:.1%}")

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

    print(
        f"\n✅ PASSED - Concurrent processing {speedup:.1f}x faster with accuracy maintained"
    )


def test_token_usage_tracking(openai_client, sentiment_dataset):
    """Track token usage across tests"""

    print("\n  🎫 Tracking token usage...\n")

    test_cases = sentiment_dataset[:10]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for case in test_cases:
        response = call_with_delay(
            openai_client,
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify as: positive, negative, or neutral\n\n{case['text']}\n\nSentiment:",
                }
            ],
            temperature=0,
        )

        # Extract token usage
        usage = response.usage
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        total_tokens += usage.total_tokens

    total_prompt_tokens / len(test_cases)
    total_completion_tokens / len(test_cases)
    avg_total = total_tokens / len(test_cases)

    print("  📊 Token Usage Statistics:")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average per request: {avg_total:.1f} tokens")

    # Calculate cost (GPT-4o-mini pricing)
    input_cost = (total_prompt_tokens / 1_000_000) * 0.150
    output_cost = (total_completion_tokens / 1_000_000) * 0.600
    total_cost = input_cost + output_cost

    print("\n  💰 Cost Analysis:")
    print(f"  Input cost: ${input_cost:.6f}")
    print(f"  Output cost: ${output_cost:.6f}")
    print(f"  Total cost: ${total_cost:.6f}")

    # Extrapolate to full dataset
    full_cost = total_cost * (len(sentiment_dataset) / len(test_cases))
    print(f"  Full dataset cost estimate: ${full_cost:.4f}")

    # Save token tracking
    token_data = {
        "timestamp": datetime.now().isoformat(),
        "model": OPENAI_MODEL,
        "test_size": len(test_cases),
        "total_tokens": total_tokens,
        "avg_tokens_per_request": avg_total,
        "total_cost": total_cost,
        "cost_per_request": total_cost / len(test_cases),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/token_usage.json", "w") as f:
        json.dump(token_data, f, indent=2)

    print("\n  💾 Token usage saved to: reports/token_usage.json")

    # Assert reasonable token usage
    assert avg_total < 50, f"Token usage too high: {avg_total:.1f}"

    print("\n✅ PASSED - Token usage tracked")


def test_error_rate_monitoring(openai_client, sentiment_dataset):
    """Monitor error rates and failures"""

    print("\n  📉 Monitoring error rates...\n")

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
    success_rate = successful / len(test_cases)

    print("  📊 Error Rate Analysis:")
    print(f"  Successful: {successful}/{len(test_cases)} ({success_rate:.1%})")
    print(f"  Failed: {failed}/{len(test_cases)} ({error_rate:.1%})")

    if errors:
        print("\n  ⚠️  Errors encountered:")
        for err in errors[:3]:  # Show first 3
            print(f"    - Index {err['index']}: {err['error']}")

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

    print("\n  💾 Error tracking saved to: reports/error_tracking.json")

    # Error rate should be very low
    assert error_rate < 0.05, f"Error rate too high: {error_rate:.1%}"

    print("\n✅ PASSED - Error rate acceptable")


@pytest.mark.skip(reason="Dataset too small for meaningful model comparison")
def test_model_version_comparison(openai_client, sentiment_dataset):
    """Compare different model versions automatically"""

    print("\n  🔄 Comparing model versions...\n")

    models = [OPENAI_MODEL, OPENAI_MODEL_COMPARE]

    test_cases = sentiment_dataset[:10]
    results = {}

    for model in models:
        print(f"  Testing {model}...")

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

        print(f"    Accuracy: {accuracy:.1%}, Time: {elapsed:.2f}s")

    # Print comparison
    print("\n  📊 Model Comparison:")
    print(f"  {'Model':<20} {'Accuracy':<12} {'Total Time':<12} {'Avg Time'}")
    print(f"  {'-'*60}")

    for model, metrics in results.items():
        print(
            f"  {model:<20} {metrics['accuracy']:<12.1%} {metrics['time']:<12.2f} {metrics['avg_time']:.3f}s"
        )

    # Save comparison
    comparison_data = {"timestamp": datetime.now().isoformat(), "models": results}

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_comparison_automated.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    print("\n  💾 Comparison saved to: reports/model_comparison_automated.json")

    # Both models should meet minimum standards
    for model, metrics in results.items():
        # Higher threshold based on actual performance
        assert (
            metrics["accuracy"] >= 0.75
        ), f"{model} accuracy too low: {metrics['accuracy']:.1%} (expected at least 75%)"

        # Reasonable time limit (3 seconds per request max)
        assert (
            metrics["avg_time"] < 3.0
        ), f"{model} too slow: {metrics['avg_time']:.2f}s per request (expected < 3s)"

    print("\n✅ PASSED - Model comparison complete")
