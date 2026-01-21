"""
Classification metrics and performance tests:
- Accuracy, precision, recall, F1-score
- Class balance and misclassification analysis
- Model comparison (OpenAI vs Anthropic)
- Temperature and latency effects
- Edge case and batch processing evaluation
"""

import pytest
from config import OPENAI_MODEL, OPENAI_MODEL_COMPARE
from helpers import (
    classify_batch,
    classify_cases,
    classify_dataset,
    compute_metrics,
    format_failures,
    map_dataset_to_cases,
    measure_latencies,
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


@pytest.mark.parametrize("model_name", [OPENAI_MODEL, OPENAI_MODEL_COMPARE])
def test_compare_models(openai_client, sentiment_dataset, model_name):
    """Compare performance across different models"""

    result = classify_dataset(openai_client, model_name, sentiment_dataset[:15])
    assert result["f1"] > 0.85, f"{model_name} F1={result['f1']:.3f}"


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
def test_temperature_impact_on_accuracy(openai_client, sentiment_dataset, temperature):
    """Does temperature affect classification accuracy?"""

    result = classify_dataset(
        openai_client, OPENAI_MODEL, sentiment_dataset, temperature=temperature
    )

    min_accuracy = 0.60 if temperature > 0.5 else 0.70
    assert (
        result["accuracy"] >= min_accuracy
    ), f"temp={temperature}: {result['accuracy']:.1%}"


def test_classification_latency(openai_client, sentiment_dataset):
    """Measure average response time for classification"""

    latencies = measure_latencies(openai_client, OPENAI_MODEL, sentiment_dataset[:10])

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

    result = classify_batch(openai_client, OPENAI_MODEL, sentiment_dataset[:10])

    assert result["accuracy"] > 0.60, f"Batch accuracy: {result['accuracy']:.3f}"
    assert result["f1"] > 0.60, f"Batch F1: {result['f1']:.3f}"
