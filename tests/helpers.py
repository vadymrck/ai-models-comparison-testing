import time
from openai import RateLimitError
from anthropic import RateLimitError as AnthropicRateLimitError
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


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
            print(f"\n⚠️  Rate limit hit, retrying in 1s: {e}")
            time.sleep(1.0)


def call_claude_with_delay(client, **kwargs):
    """Call Anthropic Claude API with a delay to avoid rate limits."""
    while True:
        try:
            response = client.messages.create(**kwargs)
            time.sleep(0.15)  # 150ms between calls
            return response
        except AnthropicRateLimitError as e:
            print(f"\n⚠️  Rate limit hit, retrying in 1s: {e}")
            time.sleep(1.0)


def classify_sentiment(client, model, text, temperature=0, provider="openai"):
    """
    Classify sentiment using OpenAI or Anthropic API.
    Returns normalized prediction: 'positive', 'negative', or 'neutral'.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        text: Text to classify
        temperature: Temperature setting (0-1)
        provider: "openai" or "anthropic"
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

    return normalize_sentiment(prediction)


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
