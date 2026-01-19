"""
Classification metrics and performance tests:
- Accuracy, precision, recall, F1-score
- Class balance and misclassification analysis
- Model comparison (OpenAI vs Anthropic)
- Temperature and latency effects
- Edge case and batch processing evaluation
"""

import math
import time

import pytest
from config import OPENAI_MODEL, OPENAI_MODEL_COMPARE
from helpers import (
    call_with_delay,
    classify_sentiment,
    compute_metrics,
    format_failures,
)
from sklearn.metrics import precision_recall_fscore_support


def test_sentiment_classification_basic(openai_client, sentiment_dataset):
    """Model can classify sentiment correctly"""

    test_cases = sentiment_dataset[:5]
    failures = []

    for case in test_cases:
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        actual = case["label"]
        if prediction != actual:
            failures.append(
                {"text": case["text"], "predicted": prediction, "expected": actual}
            )

    accuracy = (len(test_cases) - len(failures)) / len(test_cases)
    assert accuracy >= 0.80, f"Accuracy {accuracy:.2%}\n{format_failures(failures)}"


def test_sentiment_classification_full_metrics(openai_client, sentiment_dataset):
    """Compute precision, recall, F1 on full dataset"""

    predictions = []
    ground_truth = []
    failures = []

    for case in sentiment_dataset:
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])
        if prediction != case["label"]:
            failures.append(
                {
                    "text": case["text"],
                    "predicted": prediction,
                    "expected": case["label"],
                }
            )

    metrics = compute_metrics(predictions, ground_truth)

    assert (
        metrics["accuracy"] > 0.85
    ), f"Accuracy {metrics['accuracy']:.3f}\n{format_failures(failures)}"
    assert (
        metrics["f1"] > 0.85
    ), f"F1-score {metrics['f1']:.3f}\n{format_failures(failures)}"


def test_per_class_metrics(openai_client, sentiment_dataset):
    """Analyze performance per sentiment class"""

    predictions = []
    ground_truth = []

    for case in sentiment_dataset:
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])

    metrics = compute_metrics(predictions, ground_truth)

    _, _, f1, _ = precision_recall_fscore_support(
        metrics["y_true"], metrics["y_pred"], labels=[0, 1, 2], zero_division=0
    )

    classes = ["Positive", "Negative", "Neutral"]
    min_f1 = min(f1)

    # Build detailed message for assertion
    class_details = "\n".join(
        f"  {classes[i]}: F1={f1[i]:.3f}" for i in range(len(classes))
    )
    assert min_f1 > 0.75, f"Class F1 < 0.75: {min_f1:.3f}\n{class_details}"


@pytest.fixture(params=[OPENAI_MODEL, OPENAI_MODEL_COMPARE])
def model_name(request):
    """Fixture that provides different model names"""
    return request.param


def test_compare_models(openai_client, sentiment_dataset, model_name):
    """Compare performance across different models"""

    test_cases = sentiment_dataset[:15]
    predictions = []
    ground_truth = []

    for case in test_cases:
        prediction = classify_sentiment(openai_client, model_name, case["text"])
        predictions.append(prediction)
        ground_truth.append(case["label"])

    metrics = compute_metrics(predictions, ground_truth)
    assert metrics["f1"] > 0.85, f"{model_name} F1={metrics['f1']:.3f}"


@pytest.fixture(params=[0.0, 0.5, 1.0])
def temperature_value(request):
    """Fixture that provides different temperature values"""
    return request.param


def test_temperature_impact_on_accuracy(
    openai_client, sentiment_dataset, temperature_value
):
    """Does temperature affect classification accuracy?"""

    predictions = []
    ground_truth = []

    for case in sentiment_dataset:
        prediction = classify_sentiment(
            openai_client, OPENAI_MODEL, case["text"], temperature=temperature_value
        )
        predictions.append(prediction)
        ground_truth.append(case["label"])

    correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    accuracy = correct / len(sentiment_dataset)

    min_accuracy = 0.60 if temperature_value > 0.5 else 0.70
    assert accuracy >= min_accuracy, f"temp={temperature_value}: {accuracy:.1%}"


def test_classification_latency(openai_client, sentiment_dataset):
    """Measure average response time for classification"""

    test_cases = sentiment_dataset[:10]
    latencies = []

    for case in test_cases:
        start_time = time.time()
        classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        latencies.append(time.time() - start_time)

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    assert avg_latency < 5.0, f"avg={avg_latency:.3f}s, max={max_latency:.3f}s"
    assert max_latency < 10.0, f"max={max_latency:.3f}s"


def test_edge_cases(openai_client, edge_cases):
    """Test classification on edge cases and tricky examples"""

    failures = []
    for case in edge_cases:
        prediction = classify_sentiment(openai_client, OPENAI_MODEL, case["text"])
        if prediction not in case["expected"]:
            failures.append(
                {
                    "text": case["text"],
                    "predicted": prediction,
                    "expected": str(case["expected"]),
                }
            )

    success_rate = (len(edge_cases) - len(failures)) / len(edge_cases)
    assert success_rate >= 0.60, f"{success_rate:.1%}\n{format_failures(failures)}"


def test_batch_processing(openai_client, sentiment_dataset):
    """Measure efficiency of batch classification"""

    from helpers import normalize_sentiment

    batch_size = 5
    test_cases = sentiment_dataset[:10]
    total_cases = len(test_cases)
    num_batches = math.ceil(total_cases / batch_size)

    all_predictions = []
    all_ground_truth = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total_cases)
        batch_cases = test_cases[batch_start:batch_end]

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

        for pred, case in zip(predicted_labels, batch_cases):
            all_predictions.append(normalize_sentiment(pred))
            all_ground_truth.append(case["label"])

    metrics = compute_metrics(all_predictions, all_ground_truth)

    assert metrics["accuracy"] > 0.60, f"Batch accuracy: {metrics['accuracy']:.3f}"
    assert metrics["f1"] > 0.60, f"Batch F1: {metrics['f1']:.3f}"
