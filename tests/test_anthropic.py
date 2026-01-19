from config import ANTHROPIC_MODEL, OPENAI_MODEL
from helpers import classify_sentiment, compute_metrics, format_failures


def test_claude_basic_classification(anthropic_client, sentiment_dataset):
    """Claude can classify sentiment correctly"""

    test_cases = sentiment_dataset[:5]
    failures = []

    for case in test_cases:
        prediction = classify_sentiment(
            anthropic_client, ANTHROPIC_MODEL, case["text"], provider="anthropic"
        )
        if prediction != case["label"]:
            failures.append(
                {"text": case["text"], "predicted": prediction, "expected": case["label"]}
            )

    accuracy = (len(test_cases) - len(failures)) / len(test_cases)
    assert accuracy >= 0.80, f"Claude accuracy {accuracy:.1%}\n{format_failures(failures)}"


def test_claude_full_metrics(anthropic_client, sentiment_dataset):
    """Claude performance on full dataset"""

    predictions = []
    ground_truth = []
    failures = []

    for case in sentiment_dataset:
        prediction = classify_sentiment(
            anthropic_client, ANTHROPIC_MODEL, case["text"], provider="anthropic"
        )
        predictions.append(prediction)
        ground_truth.append(case["label"])
        if prediction != case["label"]:
            failures.append(
                {"text": case["text"], "predicted": prediction, "expected": case["label"]}
            )

    metrics = compute_metrics(predictions, ground_truth)

    assert metrics["f1"] > 0.85, (
        f"Claude F1={metrics['f1']:.3f}\n{format_failures(failures)}"
    )


def test_compare_openai_vs_claude(openai_client, anthropic_client, sentiment_dataset):
    """Direct comparison between OpenAI and Claude models"""

    models = [
        {"client": openai_client, "name": OPENAI_MODEL, "provider": "openai"},
        {"client": anthropic_client, "name": ANTHROPIC_MODEL, "provider": "anthropic"},
    ]

    results = {}

    for model_config in models:
        predictions = []
        ground_truth = []

        for case in sentiment_dataset:
            prediction = classify_sentiment(
                model_config["client"],
                model_config["name"],
                case["text"],
                provider=model_config["provider"],
            )
            predictions.append(prediction)
            ground_truth.append(case["label"])

        metrics = compute_metrics(predictions, ground_truth)
        results[model_config["name"]] = metrics

    for model_name, metrics in results.items():
        assert metrics["f1"] > 0.85, f"{model_name} F1={metrics['f1']:.3f}"


def test_claude_edge_cases(anthropic_client, edge_cases):
    """Claude handling of edge cases"""

    failures = []
    for case in edge_cases:
        prediction = classify_sentiment(
            anthropic_client, ANTHROPIC_MODEL, case["text"], provider="anthropic"
        )
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
