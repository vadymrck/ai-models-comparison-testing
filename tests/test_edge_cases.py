"""
Edge case tests.
Verify handling of special characters, Unicode, long text, and non-English input.
"""

from config import OPENAI_MODEL
from helpers import classify_cases, format_failures


def test_special_characters_handling(openai_client):
    """Handles special characters and Unicode"""

    special_cases = [
        {"text": "Amazing!!! ⭐⭐⭐⭐⭐", "expected": "positive"},
        {"text": "Terrible... 😡😡😡", "expected": "negative"},
        {"text": "It's $100 but worth $$$$", "expected": "positive"},
        {"text": "Product: 5/10 ⭐", "expected": "neutral"},
        {"text": "Très bien! ✓", "expected": "positive"},
        {"text": "No bueno ✗", "expected": "negative"},
    ]

    success_rate, failures = classify_cases(openai_client, OPENAI_MODEL, special_cases)

    assert (
        success_rate >= 0.65
    ), f"Special chars: {success_rate:.1%}\n{format_failures(failures)}"


def test_very_long_text_handling(openai_client):
    """Handles very long reviews appropriately"""

    long_negative = (
        "I am extremely disappointed with this product. " * 10
        + "It broke immediately. Would not recommend. Waste of money. " * 20
        + "The worst purchase I've ever made."
    )

    long_positive = (
        "This is absolutely amazing! I love it so much! " * 10
        + "Best purchase ever! Highly recommend! Five stars! " * 15
        + "Perfect product!"
    )

    cases = [
        {"text": long_negative, "expected": "negative"},
        {"text": long_positive, "expected": "positive"},
    ]

    success_rate, failures = classify_cases(openai_client, OPENAI_MODEL, cases)

    assert (
        success_rate == 1.0
    ), f"Long text: {success_rate:.1%}\n{format_failures(failures)}"


def test_non_english_handling(openai_client):
    """Test handling of non-English reviews"""

    non_english_cases = [
        {"text": "Este producto es excelente!", "expected": "positive"},
        {"text": "Très mauvais produit.", "expected": "negative"},
        {"text": "Durchschnittlich, nichts Besonderes.", "expected": "neutral"},
        {"text": "Ottimo prodotto!", "expected": "positive"},
        {"text": "Это хороший продукт!", "expected": "positive"},
    ]

    success_rate, failures = classify_cases(
        openai_client, OPENAI_MODEL, non_english_cases
    )

    assert (
        success_rate >= 0.75
    ), f"Non-English: {success_rate:.1%}\n{format_failures(failures)}"
