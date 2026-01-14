"""
Production-readiness tests:
- System prompt effectiveness
- Few-shot learning
- Streaming responses
- Cross-model consistency on clear cases
- Confidence detection
"""

from config import ANTHROPIC_MODEL, OPENAI_MODEL
from helpers import call_with_delay, classify_sentiment, normalize_sentiment


def test_system_prompt_effectiveness(openai_client):
    """System prompt overrides default behavior"""

    print("\n  🎯 Testing system prompt effectiveness...\n")

    # Custom system prompt enforcing strict format
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

    print(f"  Without system prompt: '{default_response}' ({default_word_count} words)")
    print(f"  With system prompt:    '{system_response}' ({system_word_count} words)")

    # Assert system prompt is more effective (shorter, cleaner)
    assert (
        system_word_count <= default_word_count
    ), "System prompt should produce shorter responses"

    assert system_word_count == 1, "System prompt should enforce single word response"

    assert system_response.lower() in [
        "positive",
        "negative",
        "neutral",
    ], "System prompt should enforce exact format"

    print("\n✅ PASSED - System prompt works effectively")


def test_few_shot_learning(openai_client):
    """Few-shot examples improve accuracy on edge cases"""

    print("\n  📚 Testing few-shot learning effectiveness...\n")

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

    zero_shot_pred = normalize_sentiment(response_zero_shot.choices[0].message.content)
    few_shot_pred = normalize_sentiment(response_few_shot.choices[0].message.content)

    print(f"  Test text: '{test_text}'")
    print(f"  Zero-shot prediction: {zero_shot_pred}")
    print(f"  Few-shot prediction:  {few_shot_pred}")

    # Few-shot should correctly detect sarcasm as negative
    assert (
        "negative" in few_shot_pred
    ), f"Few-shot learning should detect sarcasm correctly, got: {few_shot_pred}"

    print("\n✅ PASSED - Few-shot learning works correctly")


def test_streaming_response(openai_client):
    """Streaming responses work correctly"""

    print("\n  📡 Testing streaming response handling...\n")

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

    print(f"  Received {len(chunks)} chunks")
    print(f"  Full response: '{full_response}'")

    # Should receive multiple chunks for longer response
    assert (
        len(chunks) > 1
    ), f"Streaming should produce multiple chunks, got only {len(chunks)}"

    # Should still contain valid classification
    assert any(
        word in full_response for word in ["positive", "negative", "neutral"]
    ), "Streaming should produce valid classification"

    print(f"\n✅ PASSED - Streaming works correctly with {len(chunks)} chunks")


def test_cross_model_agreement_on_clear_cases(openai_client, anthropic_client):
    """All models should agree on obvious cases"""

    print("\n  🤝 Testing cross-model agreement on clear cases...\n")

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

    agreements = 0
    total_cases = len(clear_cases)

    for case in clear_cases:
        predictions = []

        for model in models:
            pred = classify_sentiment(
                model["client"], model["name"], case["text"], provider=model["provider"]
            )
            predictions.append(pred)

        # Check if all models agree
        all_agree = len(set(predictions)) == 1
        all_correct = all(p == case["expected"] for p in predictions)

        print(f"  '{case['text'][:40]}...'")
        print(f"     Expected: {case['expected']}")
        print(f"     GPT: {predictions[0]}, Claude: {predictions[1]}")

        # Both models should classify obvious cases correctly
        assert (
            all_correct
        ), f"Both models should classify obvious case correctly: '{case['text'][:50]}' - Expected: {case['expected']}, Got: GPT={predictions[0]}, Claude={predictions[1]}"

        # Models should agree on obvious cases
        assert (
            all_agree
        ), f"Models disagree on obvious case: '{case['text'][:50]}' - GPT={predictions[0]}, Claude={predictions[1]}"

        agreements += 1
        print("     ✓ Both models agree and correct")

    print(f"\n  📊 Cross-model agreement: 100% ({agreements}/{total_cases})")

    print("\n✅ PASSED - Models show good agreement")


def test_robustness_to_input_variations(openai_client):
    """Same sentiment, different formats should give same result"""

    print("\n  🔄 Testing robustness to input variations...\n")

    # Same sentiment expressed differently
    variations = [
        "This product is excellent!",
        "THIS PRODUCT IS EXCELLENT!",
        "this product is excellent!",
        "This    product    is    excellent!",
        "This product is excellent!!!",
        "This. Product. Is. Excellent.",
    ]

    predictions = []

    for text in variations:
        pred = classify_sentiment(openai_client, OPENAI_MODEL, text)
        predictions.append(pred)
        print(f"  '{text[:40]:<40}' → {pred}")

    # All should be positive
    all_positive = all(p == "positive" for p in predictions)
    unique_predictions = len(set(predictions))

    print(f"\n  📊 Consistency: {unique_predictions} unique prediction(s)")

    assert all_positive, "All variations should be classified as positive"
    assert unique_predictions == 1, "Should be consistent across formatting"

    print("\n✅ PASSED - Robust to input variations")
