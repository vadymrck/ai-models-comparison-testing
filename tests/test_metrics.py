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
    classify_cases,
    classify_dataset,
    classify_sentiment,
    compute_metrics,
    format_failures,
    map_dataset_to_cases,
)
from sklearn.metrics import precision_recall_fscore_support


def test_sentiment_classification_basic(openai_client, sentiment_dataset):
    """Model can classify sentiment correctly"""

    cases = map_dataset_to_cases(sentiment_dataset[:5])
    success_rate, failures = classify_cases(openai_client, OPENAI_MODEL, cases)

    assert (
        success_rate >= 0.80
    ), f"Accuracy {success_rate:.2%}\n{format_failures(failures)}"


def test_sentiment_classification_full_metrics(openai_client, sentiment_dataset):
    """Compute precision, recall, F1 on full dataset"""

    result = classify_dataset(openai_client, OPENAI_MODEL, sentiment_dataset)

    assert (
        result["accuracy"] > 0.85
    ), f"Accuracy {result['accuracy']:.3f}\n{format_failures(result['failures'])}"
    assert (
        result["f1"] > 0.85
    ), f"F1-score {result['f1']:.3f}\n{format_failures(result['failures'])}"


def test_per_class_metrics(openai_client, sentiment_dataset):
    """Analyze performance per sentiment class"""

    result = classify_dataset(openai_client, OPENAI_MODEL, sentiment_dataset)
    metrics = compute_metrics(result["predictions"], result["ground_truth"])

    _, _, f1, _ = precision_recall_fscore_support(
        metrics["y_true"], metrics["y_pred"], labels=[0, 1, 2], zero_division=0
    )

    classes = ["Positive", "Negative", "Neutral"]
    min_f1 = min(f1)

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

    result = classify_dataset(openai_client, model_name, sentiment_dataset[:15])
    assert result["f1"] > 0.85, f"{model_name} F1={result['f1']:.3f}"


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

    # edge_cases uses 'expected' key with list of acceptable values
    success_rate, failures = classify_cases(openai_client, OPENAI_MODEL, edge_cases)
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
