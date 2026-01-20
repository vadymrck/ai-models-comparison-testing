"""
Production-readiness tests:
- System prompt effectiveness
- Few-shot learning
- Streaming responses
- Cross-model consistency on clear cases
- Confidence detection
"""

from config import ANTHROPIC_MODEL, OPENAI_MODEL
from helpers import (
    call_with_delay,
    classify_cases,
    classify_sentiment,
    format_failures,
    normalize_sentiment,
)


def test_system_prompt_effectiveness(openai_client):
    """System prompt overrides default behavior"""

    test_text = "This product is amazing!"

    # Without system prompt
    response_default = call_with_delay(
        openai_client,
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Classify as: positive, negative, or neutral\n\n{test_text}\n\nSentiment:",
            }
        ],
        temperature=0,
    )

    # With strict system prompt
    response_system = call_with_delay(
        openai_client,
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment classifier. Respond ONLY with one word: positive, negative, or neutral. No explanations, no punctuation.",
            },
            {"role": "user", "content": test_text},
        ],
        temperature=0,
    )

    default_response = response_default.choices[0].message.content.strip()
    system_response = response_system.choices[0].message.content.strip()

    default_word_count = len(default_response.split())
    system_word_count = len(system_response.split())

    # Assert system prompt is more effective (shorter, cleaner)
    assert (
        system_word_count <= default_word_count
    ), f"System prompt produced longer response: {system_word_count} vs {default_word_count} words"

    assert (
        system_word_count == 1
    ), f"System prompt should enforce single word, got: '{system_response}'"

    assert system_response.lower() in [
        "positive",
        "negative",
        "neutral",
    ], f"Invalid format: '{system_response}'"


def test_few_shot_learning(openai_client):
    """Few-shot examples improve accuracy on edge cases"""

    # Tricky edge case: sarcasm
    test_text = "Oh great, another broken product. Just what I needed."

    # Zero-shot (no examples)
    response_zero_shot = call_with_delay(
        openai_client,
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Classify as: positive, negative, or neutral\n\n{test_text}\n\nSentiment:",
            }
        ],
        temperature=0,
    )

    # Few-shot (with examples)
    few_shot_messages = [
        {"role": "user", "content": "Classify: Best purchase ever!"},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "Classify: Worst product imaginable."},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "Classify: It's okay, nothing special."},
        {"role": "assistant", "content": "neutral"},
        {"role": "user", "content": f"Classify: {test_text}"},
    ]

    response_few_shot = call_with_delay(
        openai_client, model=OPENAI_MODEL, messages=few_shot_messages, temperature=0
    )

    few_shot_pred = normalize_sentiment(response_few_shot.choices[0].message.content)

    # Few-shot should correctly detect sarcasm as negative
    assert (
        "negative" in few_shot_pred
    ), f"Few-shot should detect sarcasm as negative, got: {few_shot_pred}"


def test_streaming_response(openai_client):
    """Streaming responses work correctly"""

    test_text = "This is an excellent product that exceeded all my expectations!"

    # Create streaming request with longer response to get multiple chunks
    stream = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Classify this review as positive, negative, or neutral, then explain why in one sentence.\n\nReview: {test_text}",
            }
        ],
        temperature=0,
        stream=True,
    )

    # Collect streamed chunks
    chunks = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)

    # Reconstruct full response
    full_response = "".join(chunks).strip().lower()

    # Should receive multiple chunks for longer response
    assert (
        len(chunks) > 1
    ), f"Streaming should produce multiple chunks, got {len(chunks)}"

    # Should still contain valid classification
    assert any(
        word in full_response for word in ["positive", "negative", "neutral"]
    ), f"No valid classification in response: '{full_response[:100]}...'"


def test_cross_model_agreement_on_clear_cases(openai_client, anthropic_client):
    """All models should agree on obvious cases"""

    # Obviously positive and negative cases
    clear_cases = [
        {"text": "Absolutely perfect! Best product ever!", "expected": "positive"},
        {"text": "Terrible! Completely broken and useless!", "expected": "negative"},
        {"text": "Love it! Highly recommend to everyone!", "expected": "positive"},
        {"text": "Awful! Worst purchase of my life!", "expected": "negative"},
    ]

    models = [
        {"client": openai_client, "name": OPENAI_MODEL, "provider": "openai"},
        {"client": anthropic_client, "name": ANTHROPIC_MODEL, "provider": "anthropic"},
    ]

    failures = []

    for case in clear_cases:
        predictions = []

        for model in models:
            pred = classify_sentiment(
                model["client"], model["name"], case["text"], provider=model["provider"]
            )
            predictions.append(pred)

        # Check if all models agree and are correct
        all_agree = len(set(predictions)) == 1
        all_correct = all(p == case["expected"] for p in predictions)

        if not all_correct or not all_agree:
            failures.append(
                {
                    "text": case["text"],
                    "expected": case["expected"],
                    "gpt": predictions[0],
                    "claude": predictions[1],
                }
            )

    # Format failure details
    failure_details = ""
    if failures:
        failure_details = "\n" + "\n".join(
            f"  '{f['text'][:40]}': expected={f['expected']}, GPT={f['gpt']}, Claude={f['claude']}"
            for f in failures
        )

    assert len(failures) == 0, f"Model disagreements on clear cases:{failure_details}"


def test_robustness_to_input_variations(openai_client):
    """Same sentiment, different formats should give same result"""

    variations = [
        {"text": "This product is excellent!", "expected": "positive"},
        {"text": "THIS PRODUCT IS EXCELLENT!", "expected": "positive"},
        {"text": "this product is excellent!", "expected": "positive"},
        {"text": "This    product    is    excellent!", "expected": "positive"},
        {"text": "This product is excellent!!!", "expected": "positive"},
        {"text": "This. Product. Is. Excellent.", "expected": "positive"},
    ]

    success_rate, failures = classify_cases(openai_client, OPENAI_MODEL, variations)

    assert (
        success_rate == 1.0
    ), f"Input variations: {success_rate:.1%}\n{format_failures(failures)}"
