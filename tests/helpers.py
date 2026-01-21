import time
from typing import Any

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


def call_with_delay(client: Any, **kwargs: Any) -> Any:
    """Call OpenAI API with a delay to avoid rate limits."""
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            time.sleep(0.15)  # 150ms between calls (~400 RPM max)
            return response
        except RateLimitError as e:
            print(f"\nRate limit hit, retrying in 1s: {e}")
            time.sleep(1.0)


def call_claude_with_delay(client: Any, **kwargs: Any) -> Any:
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
    client: Any,
    model: str,
    text: str,
    temperature: float = 0,
    provider: str = "openai",
    return_raw_response: bool = False,
) -> str | tuple[str, Any]:
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


def compute_metrics(predictions: list[str], ground_truth: list[str]) -> dict[str, Any]:
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


def map_dataset_to_cases(dataset: list[dict]) -> list[dict]:
    """Convert dataset items (with 'label') to test cases (with 'expected')."""
    return [{"text": item["text"], "expected": item["label"]} for item in dataset]


def classify_cases(
    client: Any, model: str, cases: list[dict], provider: str = "openai"
) -> tuple[float, list[dict]]:
    """
    Classify test cases and return success rate and failures.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        cases: List of dicts with 'text' and 'expected' keys
               (expected can be string or list of acceptable values)
        provider: "openai" or "anthropic"

    Returns:
        tuple: (success_rate, failures list)
    """
    failures = []
    for case in cases:
        prediction = classify_sentiment(client, model, case["text"], provider=provider)
        expected = case["expected"]
        is_correct = (
            prediction == expected
            if isinstance(expected, str)
            else prediction in expected
        )
        if not is_correct:
            failures.append(
                {
                    "text": case["text"],
                    "predicted": prediction,
                    "expected": str(expected),
                }
            )

    success_rate = (len(cases) - len(failures)) / len(cases)
    return success_rate, failures


def classify_dataset(
    client: Any,
    model: str,
    dataset: list[dict],
    provider: str = "openai",
    temperature: float = 0,
) -> dict[str, Any]:
    """
    Classify a dataset and return full metrics.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        dataset: List of dicts with 'text' and 'label' keys
        provider: "openai" or "anthropic"
        temperature: Temperature setting (0-1)

    Returns:
        dict with accuracy, precision, recall, f1, predictions, ground_truth, failures
    """
    predictions = []
    ground_truth = []
    failures = []

    for case in dataset:
        prediction = classify_sentiment(
            client, model, case["text"], temperature=temperature, provider=provider
        )
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

    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "predictions": predictions,
        "ground_truth": ground_truth,
        "failures": failures,
    }


def classify_repeated(
    client: Any, model: str, text: str, runs: int = 5, provider: str = "openai"
) -> list[str]:
    """
    Classify the same text multiple times to test consistency.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        text: Text to classify repeatedly
        runs: Number of times to classify
        provider: "openai" or "anthropic"

    Returns:
        list of predictions
    """
    return [
        classify_sentiment(client, model, text, provider=provider) for _ in range(runs)
    ]


def measure_latencies(
    client: Any, model: str, dataset: list[dict], provider: str = "openai"
) -> list[float]:
    """
    Measure classification latency for each item in dataset.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        dataset: List of dicts with 'text' key
        provider: "openai" or "anthropic"

    Returns:
        list of latencies in seconds
    """
    latencies = []
    for case in dataset:
        start = time.time()
        classify_sentiment(client, model, case["text"], provider=provider)
        latencies.append(time.time() - start)
    return latencies


def classify_batch(
    client: Any, model: str, dataset: list[dict], batch_size: int = 5
) -> dict[str, Any]:
    """
    Classify multiple items in a single prompt (batch classification).

    Args:
        client: OpenAI client
        model: Model name
        dataset: List of dicts with 'text' and 'label' keys
        batch_size: Number of items per batch

    Returns:
        dict with accuracy, f1, predictions, ground_truth
    """
    all_predictions = []
    all_ground_truth = [case["label"] for case in dataset]

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        prompt = "Classify each review as positive, negative, or neutral. Respond with only the labels separated by commas.\n\n"
        prompt += "\n".join(f"{j}. {case['text']}" for j, case in enumerate(batch, 1))
        prompt += "\n\nLabels (comma-separated):"

        response = call_with_delay(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip().lower()
        predictions = [normalize_sentiment(p.strip()) for p in answer.split(",")]
        all_predictions.extend(predictions)

    metrics = compute_metrics(all_predictions, all_ground_truth)
    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "predictions": all_predictions,
        "ground_truth": all_ground_truth,
    }


def classify_with_tokens(
    client: Any, model: str, dataset: list[dict], provider: str = "openai"
) -> dict[str, int]:
    """
    Classify dataset and return token usage totals.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        dataset: List of dicts with 'text' key
        provider: "openai" or "anthropic"

    Returns:
        dict with input_tokens and output_tokens
    """
    total_input = 0
    total_output = 0

    for case in dataset:
        _, response = classify_sentiment(
            client, model, case["text"], provider=provider, return_raw_response=True
        )
        total_input += response.usage.prompt_tokens
        total_output += response.usage.completion_tokens

    return {"input_tokens": total_input, "output_tokens": total_output}


def format_failures(failures: list[dict], limit: int = 5) -> str:
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


def get_response(client: Any, model: str, prompt: str, provider: str = "openai") -> str:
    """Send a single prompt and return lowercased response."""
    return collect_responses(client, model, [prompt], provider)[0]


def collect_responses(
    client: Any, model: str, prompts: list[str], provider: str = "openai"
) -> list[str]:
    """
    Send multiple prompts and collect responses.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        prompts: List of prompt strings
        provider: "openai" or "anthropic"

    Returns:
        list of response strings (lowercased)
    """
    responses = []
    for prompt in prompts:
        if provider == "anthropic":
            response = call_claude_with_delay(
                client,
                model=model,
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            responses.append(response.content[0].text.lower())
        else:
            response = call_with_delay(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            responses.append(response.choices[0].message.content.lower())
    return responses


def verify_qa_pairs(
    client: Any,
    model: str,
    qa_pairs: list[tuple[str, str]],
    provider: str = "openai",
) -> tuple[float, list[dict]]:
    """
    Verify question-answer pairs where expected value should be contained in response.

    Args:
        client: OpenAI or Anthropic client
        model: Model name
        qa_pairs: List of (question, expected_answer) tuples
        provider: "openai" or "anthropic"

    Returns:
        tuple: (success_rate, failures list)
    """
    failures = []
    for question, expected in qa_pairs:
        prompt = f"{question} Answer with just the number."

        if provider == "anthropic":
            response = call_claude_with_delay(
                client,
                model=model,
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text.strip()
        else:
            response = call_with_delay(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = response.choices[0].message.content.strip()

        if expected not in answer:
            failures.append({"question": question, "expected": expected, "got": answer})

    success_rate = (len(qa_pairs) - len(failures)) / len(qa_pairs)
    return success_rate, failures


def calculate_cost(
    input_tokens: int, output_tokens: int, model: str
) -> dict[str, float]:
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
