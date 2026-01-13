import pytest
from helpers import classify_sentiment, compute_metrics

# Model constants for all tests
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
OPEN_AI_MODEL = "gpt-4o-mini"


def test_claude_basic_classification(anthropic_client, sentiment_dataset):
    """Claude can classify sentiment correctly"""
    
    print(f"\n  🤖 Testing Claude on basic classification...")
    
    test_cases = sentiment_dataset[:5]
    
    predictions = []
    ground_truth = []
    
    for case in test_cases:
        prediction = classify_sentiment(
            anthropic_client,
            ANTHROPIC_MODEL,
            case["text"],
            provider="anthropic"
        )
        predictions.append(prediction)
        ground_truth.append(case["label"])
        
        print(f"  Text: {case['text'][:50]}...")
        print(f"  Predicted: {prediction}, Actual: {case['label']}")
    
    correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    accuracy = correct / len(test_cases)

    print(f"\n  📊 Claude Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")

    assert accuracy >= 0.80, f"Claude accuracy too low: {accuracy:.1%}"

    print(f"\n✅ PASSED - Claude basic classification working")


def test_claude_full_metrics(anthropic_client, sentiment_dataset):
    """Claude performance on full dataset"""

    print(f"\n  🔍 Testing Claude on full dataset ({len(sentiment_dataset)} examples)...")
    print(f"  ⏳ This will take ~30-60 seconds...\n")

    predictions = []
    ground_truth = []
    test_data = []

    for i, case in enumerate(sentiment_dataset, 1):
        prediction = classify_sentiment(
            anthropic_client,
            ANTHROPIC_MODEL,
            case["text"],
            provider="anthropic"
        )
        predictions.append(prediction)
        ground_truth.append(case["label"])
        test_data.append(case["text"])

        if i % 5 == 0:
            print(f"  Progress: {i}/{len(sentiment_dataset)} complete")

    metrics = compute_metrics(predictions, ground_truth)

    print(f"\n  {'='*60}")
    print(f"  📊 CLAUDE METRICS REPORT")
    print(f"  {'='*60}")
    print(f"  Model: {ANTHROPIC_MODEL}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    print(f"  {'='*60}")

    # Detailed diagnostic info if performance is poor
    if metrics['f1'] < 0.85:
        from sklearn.metrics import confusion_matrix, classification_report

        print(f"\n  ⚠️  LOW F1-SCORE DETECTED - DIAGNOSTIC ANALYSIS")
        print(f"  {'='*60}\n")

        # 1. Per-class breakdown
        print(f"  PER-CLASS PERFORMANCE:")
        report = classification_report(ground_truth, predictions,
                                       output_dict=True, zero_division=0)
        for label in ['positive', 'negative', 'neutral']:
            if label in report:
                print(f"    {label:8} - P:{report[label]['precision']:.2f} "
                      f"R:{report[label]['recall']:.2f} "
                      f"F1:{report[label]['f1-score']:.2f} "
                      f"(support: {int(report[label]['support'])})")

        # 2. Confusion matrix
        print(f"\n  CONFUSION MATRIX:")
        cm = confusion_matrix(ground_truth, predictions,
                             labels=['positive', 'negative', 'neutral'])
        labels = ['positive', 'negative', 'neutral']
        print(f"    {'':>12} {'Predicted →':>12}")
        print(f"    {'Actual ↓':>12} {' '.join([f'{l[:3]:>4}' for l in labels])}")
        for i, label in enumerate(labels):
            print(f"    {label:>12} {' '.join([f'{cm[i][j]:>4}' for j in range(len(labels))])}")

        # 3. Error breakdown
        print(f"\n  ERROR BREAKDOWN:")
        errors = {}
        for pred, actual in zip(predictions, ground_truth):
            if pred != actual:
                key = f"{actual} → {pred}"
                errors[key] = errors.get(key, 0) + 1

        for error_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            print(f"    {error_type:20} {count:>3} errors")

        # 4. Sample failures
        print(f"\n  SAMPLE FAILURES (first 8):")
        failures = [(t, p, a) for t, p, a in zip(test_data, predictions, ground_truth) if p != a]
        for i, (text, pred, actual) in enumerate(failures[:8], 1):
            print(f"    [{i}] {actual} → {pred}")
            print(f"        {text[:70]}{'...' if len(text) > 70 else ''}\n")

        total_errors = len(failures)
        print(f"  Total errors: {total_errors}/{len(predictions)} ({total_errors/len(predictions):.1%})")
        print(f"  {'='*60}\n")

    assert metrics['f1'] > 0.85, f"Claude F1-score too low: {metrics['f1']:.3f}"

    print(f"\n✅ PASSED - Claude F1: {metrics['f1']:.3f}")


def test_compare_openai_vs_claude(openai_client, anthropic_client, sentiment_dataset):
    """Direct comparison between OpenAI and Claude models"""
    
    print(f"\n  {'='*70}")
    print(f"  🥊 HEAD-TO-HEAD: OpenAI vs Anthropic")
    print(f"  {'='*70}\n")
    
    test_cases = sentiment_dataset
    
    models = [
        {"client": openai_client, "name": OPEN_AI_MODEL, "provider": "openai"},
        {"client": anthropic_client, "name": ANTHROPIC_MODEL, "provider": "anthropic"}
    ]
    
    results = {}
    
    for model_config in models:
        print(f"  Testing {model_config['name']}...")
        
        predictions = []
        ground_truth = []
        
        for case in test_cases:
            prediction = classify_sentiment(
                model_config["client"],
                model_config["name"],
                case["text"],
                provider=model_config["provider"]
            )
            predictions.append(prediction)
            ground_truth.append(case["label"])
        
        metrics = compute_metrics(predictions, ground_truth)
        results[model_config["name"]] = metrics
        
        print(f"  ✓ F1-Score: {metrics['f1']:.3f}\n")
    
    # Print comparison table
    print(f"  {'='*70}")
    print(f"  COMPARISON RESULTS")
    print(f"  {'='*70}\n")
    print(f"  {'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1'}")
    print(f"  {'-'*70}")
    
    for model_name, metrics in results.items():
        print(f"  {model_name:<30} "
              f"{metrics['accuracy']:<12.3f} "
              f"{metrics['precision']:<12.3f} "
              f"{metrics['recall']:<12.3f} "
              f"{metrics['f1']:.3f}")
    
    print(f"\n  {'='*70}")
    
    # Determine winner
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    f1_diff = abs(results[OPEN_AI_MODEL]["f1"] - results[ANTHROPIC_MODEL]["f1"])
    
    if f1_diff < 0.02:
        print(f"  🤝 TIE: Both models performed similarly (Δ F1: {f1_diff:.3f})")
    else:
        print(f"  🏆 WINNER: {best_model[0]} (F1: {best_model[1]['f1']:.3f})")
    
    print(f"  {'='*70}\n")
    
    # Both should meet minimum threshold
    for model_name, metrics in results.items():
        assert metrics['f1'] > 0.85, f"{model_name} F1 too low: {metrics['f1']:.3f}"
    
    print(f"\n✅ PASSED - Comparison complete")


def test_claude_edge_cases(anthropic_client, edge_cases):
    """Claude handling of edge cases"""

    print(f"\n  🧪 Testing Claude on {len(edge_cases)} edge cases:")
    
    results = []
    
    for i, case in enumerate(edge_cases, 1):
        prediction = classify_sentiment(
            anthropic_client,
            ANTHROPIC_MODEL,
            case["text"],
            provider="anthropic"
        )
        
        is_acceptable = prediction in case["expected"]
        status = "✓" if is_acceptable else "⚠"
        
        print(f"\n  {status} Case {i}: {case['description']}")
        print(f"    Text: {case['text']}")
        print(f"    Predicted: {prediction}")
        print(f"    Acceptable: {case['expected']}")
        
        results.append(is_acceptable)
    
    success_rate = sum(results) / len(results)
    print(f"\n  📊 Claude edge case success: {success_rate:.1%} ({sum(results)}/{len(results)})")

    assert success_rate >= 0.60, f"Edge case success rate too low: {success_rate:.1%}"

    print(f"\n✅ PASSED - Claude handles edge cases reasonably")