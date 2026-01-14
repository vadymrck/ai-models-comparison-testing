import pytest
import math
import time

from config import OPENAI_MODEL, OPENAI_MODEL_COMPARE
from helpers import call_with_delay, classify_sentiment, compute_metrics
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


def test_sentiment_classification_basic(openai_client, sentiment_dataset):
    """Model can classify sentiment correctly"""

    test_cases = sentiment_dataset[:5]

    print(f"\n  Testing sentiment classification on {len(test_cases)} examples:")

    correct = 0
    for case in test_cases:
        # Use helper!
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        actual = case["label"]

        is_correct = prediction == actual
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(
            f"  {status} Predicted: {prediction:8} | Actual: {actual:8} | Text: {case['text'][:50]}..."
        )

    accuracy = correct / len(test_cases)
    print(f"\n  📊 Accuracy: {correct}/{len(test_cases)} = {accuracy:.2%}")

    assert accuracy >= 0.80, f"Accuracy too low: {accuracy:.2%}"

    print("\n✅ PASSED - Basic classification working")


def test_sentiment_classification_full_metrics(openai_client, sentiment_dataset):
    """Compute precision, recall, F1 on full dataset"""

    print(f"\n  🔍 Testing on full dataset ({len(sentiment_dataset)} examples)...")
    print("  ⏳ This will take ~30-60 seconds...\n")

    predictions = []
    ground_truth = []

    for i, case in enumerate(sentiment_dataset, 1):
        # Use helper!
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])

        # Progress indicator
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(sentiment_dataset)} complete")

    # Use helper for metrics!
    metrics = compute_metrics(predictions, ground_truth)

    # Confusion matrix
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"])

    # Print detailed results
    print(f"\n  {'='*60}")
    print("  📊 METRICS REPORT")
    print(f"  {'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    print(f"  {'='*60}")

    print("\n  📈 Confusion Matrix:")
    print("                Predicted")
    print("              Pos  Neg  Neu")
    print(f"  Actual Pos  {cm[0][0]:3}  {cm[0][1]:3}  {cm[0][2]:3}")
    print(f"         Neg  {cm[1][0]:3}  {cm[1][1]:3}  {cm[1][2]:3}")
    print(f"         Neu  {cm[2][0]:3}  {cm[2][1]:3}  {cm[2][2]:3}")

    # Show some errors
    print("\n  ❌ Misclassifications:")
    errors = []
    for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
        if pred != true:
            errors.append(
                {
                    "text": sentiment_dataset[i]["text"][:60],
                    "predicted": pred,
                    "actual": true,
                }
            )

    if errors:
        for err in errors[:5]:
            print(
                f"     Predicted {err['predicted']:8} (actually {err['actual']:8}): {err['text']}..."
            )
    else:
        print("     None! Perfect classification! 🎉")

    # Assertions
    assert metrics["accuracy"] > 0.85, f"Accuracy too low: {metrics['accuracy']:.3f}"
    assert metrics["f1"] > 0.85, f"F1-score too low: {metrics['f1']:.3f}"

    print(f"\n✅ PASSED - F1-Score: {metrics['f1']:.3f}")

    return metrics


def test_per_class_metrics(openai_client, sentiment_dataset):
    """Analyze performance per sentiment class"""

    print("\n  Analyzing per-class performance...\n")

    predictions = []
    ground_truth = []

    for case in sentiment_dataset:
        # Use helper!
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])

    # Use helper for metrics!
    metrics = compute_metrics(predictions, ground_truth)

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        metrics["y_true"], metrics["y_pred"], labels=[0, 1, 2], zero_division=0
    )

    classes = ["Positive", "Negative", "Neutral"]

    print(
        f"  {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}"
    )
    print(f"  {'-'*65}")

    for i, class_name in enumerate(classes):
        print(
            f"  {class_name:<12} {precision[i]:<12.3f} {recall[i]:<12.3f} {f1[i]:<12.3f} {support[i]}"
        )

    print(f"  {'-'*65}")

    # Check if any class is performing poorly
    min_f1 = min(f1)
    assert min_f1 > 0.75, f"Some class has F1 < 0.75: {min_f1:.3f}"

    print("\n✅ PASSED - All classes performing adequately")


@pytest.fixture(params=[OPENAI_MODEL, OPENAI_MODEL_COMPARE])
def model_name(request):
    """Fixture that provides different model names"""
    return request.param


def test_compare_models(openai_client, sentiment_dataset, model_name):
    """Compare performance across different models"""

    print(f"\n  🤖 Testing model: {model_name}")
    print(f"  {'='*60}")

    # Use first 15 examples for faster comparison
    test_cases = sentiment_dataset[:15]

    predictions = []
    ground_truth = []

    for case in test_cases:
        # Use helper with dynamic model!
        prediction = classify_sentiment(openai_client, model_name, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])

    # Use helper for metrics!
    metrics = compute_metrics(predictions, ground_truth)

    print(f"\n  📊 Results for {model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    print(f"  {'='*60}")

    # Both models should perform reasonably well
    assert metrics["f1"] > 0.85, f"{model_name} F1-score too low: {metrics['f1']:.3f}"

    print(f"\n✅ PASSED - {model_name}: F1={metrics['f1']:.3f}")


@pytest.fixture(params=[0.0, 0.5, 1.0])
def temperature_value(request):
    """Fixture that provides different temperature values"""
    return request.param


def test_temperature_impact_on_accuracy(
    openai_client, sentiment_dataset, temperature_value
):
    """Does temperature affect classification accuracy?"""

    print(f"\n  🌡️  Testing temperature: {temperature_value}")
    print(f"  {'='*60}")

    test_cases = sentiment_dataset

    predictions = []
    ground_truth = []

    for case in test_cases:
        # Use helper with temperature parameter!
        prediction = classify_sentiment(
            openai_client, OPENAI_MODEL, case["text"], temperature=temperature_value
        )
        predictions.append(prediction)
        ground_truth.append(case["label"])

    # Calculate accuracy
    correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    accuracy = correct / len(test_cases)

    print(
        f"\n  📊 Temp={temperature_value}: Accuracy = {accuracy:.1%} ({correct}/{len(test_cases)})"
    )
    print(f"  {'='*60}")

    # Higher temperature might be slightly less accurate
    min_accuracy = 0.60 if temperature_value > 0.5 else 0.70
    assert (
        accuracy >= min_accuracy
    ), f"Accuracy too low at temp={temperature_value}: {accuracy:.1%}"

    print(f"\n✅ PASSED - Temp={temperature_value}: Accuracy={accuracy:.1%}")


def test_classification_latency(openai_client, sentiment_dataset):
    """Measure average response time for classification"""

    print("\n  ⏱️  Measuring classification latency...")

    test_cases = sentiment_dataset[:10]
    latencies = []

    for case in test_cases:
        start_time = time.time()

        # Use helper!
        classify_sentiment(openai_client, OPENAI_MODEL, case["text"])

        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print("\n  📊 Latency Statistics:")
    print(f"  Average: {avg_latency:.3f}s")
    print(f"  Min:     {min_latency:.3f}s")
    print(f"  Max:     {max_latency:.3f}s")

    # Assert reasonable performance
    assert avg_latency < 5.0, f"Average latency too high: {avg_latency:.3f}s"
    assert max_latency < 10.0, f"Max latency too high: {max_latency:.3f}s"

    print(f"\n✅ PASSED - Avg latency: {avg_latency:.3f}s")


def test_edge_cases(openai_client, edge_cases):
    """Test classification on edge cases and tricky examples"""

    print(f"\n  🧪 Testing OpenAI on {len(edge_cases)} edge cases:")

    results = []
    for i, case in enumerate(edge_cases, 1):
        # Use helper!
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])

        is_acceptable = prediction in case["expected"]
        status = "✓" if is_acceptable else "⚠"

        print(f"\n  {status} Case {i}: {case['description']}")
        print(f"    Text: {case['text']}")
        print(f"    Predicted: {prediction}")
        print(f"    Acceptable: {case['expected']}")

        results.append(is_acceptable)

    success_rate = sum(results) / len(results)
    print(
        f"\n  📊 Edge case success rate: {success_rate:.1%} ({sum(results)}/{len(results)})"
    )

    # At least 60% should be handled correctly (edge cases are challenging)
    assert success_rate >= 0.60, f"Edge case success rate too low: {success_rate:.1%}"

    print("\n✅ PASSED - Edge cases handled reasonably")


def test_batch_processing(openai_client, sentiment_dataset):
    """Measure efficiency of batch classification"""

    batch_size = 5
    max_cases = 10  # Use 10 for quick test, or None for full dataset

    print(f"  {'='*60}")
    if max_cases is not None and max_cases > 0:
        test_cases = sentiment_dataset[:max_cases]
        print(
            f"\n  📦 Testing batch processing on {max_cases} reviews (batch size: {batch_size})..."
        )
    else:
        test_cases = sentiment_dataset
        print(
            f"\n  📦 Testing batch processing on FULL dataset ({len(test_cases)} reviews, batch size: {batch_size})..."
        )

    total_cases = len(test_cases)
    num_batches = math.ceil(total_cases / batch_size)

    all_predictions = []
    all_ground_truth = []

    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total_cases)
        batch_cases = test_cases[batch_start:batch_end]
        if not batch_cases:
            continue

        # Build batch prompt
        batch_prompt = "Classify each review as positive, negative, or neutral. Respond with only the labels separated by commas.\n\n"
        for i, case in enumerate(batch_cases, 1):
            batch_prompt += f"{i}. {case['text']}\n"
        batch_prompt += "\nLabels (comma-separated):"

        response = call_with_delay(
            openai_client,
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip().lower()
        predicted_labels = [label.strip() for label in answer.split(",")]

        print(f"\n  Batch {batch_idx+1}/{num_batches}:")
        print(f"    Reviews: {batch_start+1}-{batch_end}")
        print(f"    Response: {answer}")
        print(f"    Parsed labels: {predicted_labels}")

        if len(predicted_labels) < len(batch_cases):
            print("    ⚠️  Warning: Fewer labels than reviews in this batch!")

        # Store results
        for pred, case in zip(predicted_labels, batch_cases):
            all_predictions.append(pred)
            all_ground_truth.append(case["label"])

    total_time = time.time() - start_time
    print(f"\n  ⏱️  Total time: {total_time:.2f}s for {total_cases} reviews")
    print(f"  ⏱️  Average per review: {total_time/total_cases:.3f}s")

    # Normalize predictions
    norm_preds = []
    for p in all_predictions:
        if p not in ["positive", "negative", "neutral"]:
            if "positive" in p:
                p = "positive"
            elif "negative" in p:
                p = "negative"
            elif "neutral" in p:
                p = "neutral"
            else:
                p = "neutral"
        norm_preds.append(p)

    # Use helper for metrics!
    metrics = compute_metrics(norm_preds, all_ground_truth)

    print("\n  📊 Batch Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")

    # Show errors
    errors = []
    for i, (pred, true) in enumerate(zip(norm_preds, all_ground_truth)):
        if pred != true:
            errors.append(
                {
                    "text": test_cases[i]["text"][:60],
                    "predicted": pred,
                    "actual": true,
                }
            )
    if errors:
        print("\n  ⚠️ Example misclassifications:")
        for err in errors[:5]:
            print(
                f"     Predicted {err['predicted']:8} (actually {err['actual']:8}): {err['text']}..."
            )
    else:
        print("\n  No misclassifications! 🎉")

    # Assertions
    min_acc = 0.65 if total_cases > 20 else 0.60
    min_f1 = 0.65 if total_cases > 20 else 0.60
    assert (
        metrics["accuracy"] > min_acc
    ), f"Batch accuracy too low: {metrics['accuracy']:.3f}"
    assert metrics["f1"] > min_f1, f"Batch F1-score too low: {metrics['f1']:.3f}"

    print("\n✅ PASSED - Batch processing working")
