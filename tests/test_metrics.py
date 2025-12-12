import pytest

from helpers import call_with_delay
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

# Model constant for all tests
MODEL = "gpt-4o-mini"

def test_sentiment_classification_basic(openai_client, sentiment_dataset):
    """TEST #11: Model can classify sentiment correctly"""

    # Take first 5 examples for quick test
    test_cases = sentiment_dataset[:5]

    print(f"\n  Testing sentiment classification on {len(test_cases)} examples:")

    correct = 0
    for case in test_cases:
        response = call_with_delay(
            openai_client,
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""Classify the sentiment of this review as exactly one word: positive, negative, or neutral.

Review: {case['text']}

Answer with only one word:""",
                }
            ],
            temperature=0,
        )

        prediction = response.choices[0].message.content.strip().lower()
        actual = case["label"]

        is_correct = prediction == actual
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(
            f"  {status} Predicted: {prediction:8} | Actual: {actual:8} | Text: {case['text'][:50]}..."
        )

    accuracy = correct / len(test_cases)
    print(f"\n  📊 Accuracy: {correct}/{len(test_cases)} = {accuracy:.2%}")

    assert accuracy >= 0.60, f"Accuracy too low: {accuracy:.2%}"

    print(f"\n✅ TEST #11 PASSED - Basic classification working")


def test_sentiment_classification_full_metrics(openai_client, sentiment_dataset):
    """TEST #12: Compute precision, recall, F1 on full dataset"""

    print(f"\n  🔍 Testing on full dataset ({len(sentiment_dataset)} examples)...")
    print(f"  ⏳ This will take ~30-60 seconds...\n")

    predictions = []
    ground_truth = []

    for i, case in enumerate(sentiment_dataset, 1):
        response = call_with_delay(
            openai_client,
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""Classify the sentiment as one word: positive, negative, or neutral.

Review: {case['text']}

Sentiment:""",
                }
            ],
            temperature=0,
        )
        prediction = response.choices[0].message.content.strip().lower()

        # Normalize prediction (handle variations)
        if prediction not in ["positive", "negative", "neutral"]:
            # Try to extract the sentiment from response
            if "positive" in prediction:
                prediction = "positive"
            elif "negative" in prediction:
                prediction = "negative"
            elif "neutral" in prediction:
                prediction = "neutral"
            else:
                prediction = "neutral"  # Default fallback

        predictions.append(prediction)
        ground_truth.append(case["label"])

        # Progress indicator
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(sentiment_dataset)} complete")

    # Convert to numeric labels for sklearn
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    y_true = [label_map[label] for label in ground_truth]
    y_pred = [label_map[pred] for pred in predictions]

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print detailed results
    print(f"\n  {'='*60}")
    print(f"  📊 METRICS REPORT")
    print(f"  {'='*60}")
    print(f"  Accuracy:  {accuracy:.3f} ({accuracy:.1%})")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  {'='*60}")

    print(f"\n  📈 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Pos  Neg  Neu")
    print(f"  Actual Pos  {cm[0][0]:3}  {cm[0][1]:3}  {cm[0][2]:3}")
    print(f"         Neg  {cm[1][0]:3}  {cm[1][1]:3}  {cm[1][2]:3}")
    print(f"         Neu  {cm[2][0]:3}  {cm[2][1]:3}  {cm[2][2]:3}")

    # Show some errors
    print(f"\n  ❌ Misclassifications:")
    errors = []
    for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
        if pred != true:
            errors.append(
                {
                    "text": sentiment_dataset[i]["text"][:60],
                    "predicted": pred,
                    "actual": true,
                }
            )

    if errors:
        for err in errors[:5]:  # Show first 5 errors
            print(
                f"     Predicted {err['predicted']:8} (actually {err['actual']:8}): {err['text']}..."
            )
    else:
        print(f"     None! Perfect classification! 🎉")

    # Assertions
    assert accuracy > 0.70, f"Accuracy too low: {accuracy:.3f}"
    assert f1 > 0.70, f"F1-score too low: {f1:.3f}"

    print(f"\n✅ TEST #12 PASSED - F1-Score: {f1:.3f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def test_per_class_metrics(openai_client, sentiment_dataset):
    """TEST #13: Analyze performance per sentiment class"""
    
    print(f"\n  Analyzing per-class performance...\n")
    
    predictions = []
    ground_truth = []
    
    for case in sentiment_dataset:
        response = call_with_delay(
            openai_client,
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"Classify as: positive, negative, or neutral\n\n{case['text']}\n\nSentiment:"
            }],
            temperature=0
        )
        
        prediction = response.choices[0].message.content.strip().lower()
        
        # Normalize
        if prediction not in ["positive", "negative", "neutral"]:
            if "positive" in prediction:
                prediction = "positive"
            elif "negative" in prediction:
                prediction = "negative"
            else:
                prediction = "neutral"
        
        predictions.append(prediction)
        ground_truth.append(case['label'])
    
    # Convert to numeric
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    y_true = [label_map[label] for label in ground_truth]
    y_pred = [label_map[pred] for pred in predictions]
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0
    )
    
    classes = ["Positive", "Negative", "Neutral"]
    
    print(f"  {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print(f"  {'-'*65}")
    
    for i, class_name in enumerate(classes):
        print(f"  {class_name:<12} {precision[i]:<12.3f} {recall[i]:<12.3f} {f1[i]:<12.3f} {support[i]}")
    
    print(f"  {'-'*65}")
    
    # Check if any class is performing poorly
    min_f1 = min(f1)
    assert min_f1 > 0.60, f"Some class has F1 < 0.60: {min_f1:.3f}"
    
    print(f"\n✅ TEST #13 PASSED - All classes performing adequately")