from config import ANTHROPIC_MODEL, OPENAI_MODEL
from helpers import (
    classify_cases,
    classify_dataset,
    format_failures,
    map_dataset_to_cases,
)


def test_claude_basic_classification(anthropic_client, sentiment_dataset):
    """Claude can classify sentiment correctly"""

    cases = map_dataset_to_cases(sentiment_dataset[:5])
    success_rate, failures = classify_cases(
        anthropic_client, ANTHROPIC_MODEL, cases, provider="anthropic"
    )

    assert (
        success_rate >= 0.80
    ), f"Claude accuracy {success_rate:.1%}\n{format_failures(failures)}"


def test_claude_full_metrics(anthropic_client, sentiment_dataset):
    """Claude performance on full dataset"""

    result = classify_dataset(
        anthropic_client, ANTHROPIC_MODEL, sentiment_dataset, provider="anthropic"
    )

    assert (
        result["f1"] > 0.85
    ), f"Claude F1={result['f1']:.3f}\n{format_failures(result['failures'])}"


def test_compare_openai_vs_claude(openai_client, anthropic_client, sentiment_dataset):
    """Direct comparison between OpenAI and Claude models"""

    models = [
        {"client": openai_client, "name": OPENAI_MODEL, "provider": "openai"},
        {"client": anthropic_client, "name": ANTHROPIC_MODEL, "provider": "anthropic"},
    ]

    for model_config in models:
        result = classify_dataset(
            model_config["client"],
            model_config["name"],
            sentiment_dataset,
            provider=model_config["provider"],
        )
        assert result["f1"] > 0.85, f"{model_config['name']} F1={result['f1']:.3f}"


def test_claude_edge_cases(anthropic_client, edge_cases):
    """Claude handling of edge cases"""

    # edge_cases uses 'expected' key with list of acceptable values
    success_rate, failures = classify_cases(
        anthropic_client, ANTHROPIC_MODEL, edge_cases, provider="anthropic"
    )

    assert success_rate >= 0.60, f"{success_rate:.1%}\n{format_failures(failures)}"
