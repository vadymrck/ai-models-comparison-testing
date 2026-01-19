import time

from anthropic import RateLimitError as AnthropicRateLimitError
from openai import RateLimitError
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def normalize_sentiment(prediction: str) -> str:
    """Normalize model output to: positive, negative, or neutral."""
    prediction = prediction.strip().lower()
    if prediction not in ["positive", "negative", "neutral"]:
        if "positive" in prediction:
            return "positive"
        elif "negative" in prediction:
            return "negative"
        return "neutral"
    return prediction


def call_with_delay(client, **kwargs):
    """Call OpenAI API with a delay to avoid rate limits."""
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            time.sleep(0.15)  # 150ms between calls (~400 RPM max)
            return response
        except RateLimitError as e:
            print(f"\nRate limit hit, retrying in 1s: {e}")
            time.sleep(1.0)


def call_claude_with_delay(client, **kwargs):
    """Call Anthropic Claude API with a delay to avoid rate limits."""
    while True:
        try:
            response = client.messages.create(**kwargs)
            time.sleep(0.15)  # 150ms between calls
            return response
        except AnthropicRateLimitError as e:
            print(f"\nRate limit hit, retrying in 1s: {e}")
            time.sleep(1.0)


def classify_sentiment(
    client, model, text, temperature=0, provider="openai", return_raw_response=False
):
    """
    Classify sentiment using OpenAI or Anthropic API.
    Returns normalized prediction: 'positive', 'negative', or 'neutral'.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        text: Text to classify
        temperature: Temperature setting (0-1)
        provider: "openai" or "anthropic"
        return_raw_response: If True, returns (prediction, response) tuple

    Returns:
        str: Normalized sentiment ('positive', 'negative', 'neutral')
        OR tuple: (prediction, raw_response) if return_raw_response=True
    """
    prompt = f"Classify as: positive, negative, or neutral\n\n{text}\n\nSentiment:"

    if provider == "anthropic":
        response = call_claude_with_delay(
            client,
            model=model,
            max_tokens=10,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        prediction = response.content[0].text
    else:  # openai
        response = call_with_delay(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        prediction = response.choices[0].message.content

    normalized = normalize_sentiment(prediction)

    if return_raw_response:
        return normalized, response
    return normalized


def compute_metrics(predictions, ground_truth):
    """
    Compute classification metrics from predictions and ground truth.
    Returns dict with accuracy, precision, recall, F1, and numeric labels.
    """
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    y_true = [label_map[label] for label in ground_truth]
    y_pred = [label_map[pred] for pred in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def format_failures(failures, limit=5):
    """
    Format failure details for assertion messages.

    Args:
        failures: List of dicts with 'text', 'predicted', 'expected' keys
        limit: Max number of failures to show

    Returns:
        str: Formatted failure details
    """
    lines = []
    for f in failures[:limit]:
        text = f.get("text", "")[:50]
        lines.append(f"  {f['predicted']} != {f['expected']}: {text}...")
    remaining = len(failures) - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return "\n".join(lines)


def calculate_cost(input_tokens, output_tokens, model):
    """
    Calculate API cost based on token usage and model pricing.

    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        model: Model name (must exist in config.PRICING)

    Returns:
        dict with input_cost, output_cost, and total_cost
    """
    from config import PRICING

    pricing = PRICING.get(model, {"input": 0, "output": 0})

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
