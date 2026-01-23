# AI Models Comparison Testing

A pytest-based test framework for evaluating and comparing AI model behavior across OpenAI GPT and Anthropic Claude APIs. Tests cover sentiment classification accuracy, hallucination detection, performance benchmarks, and production readiness checks.

## Features

- **Multi-provider support**: OpenAI (GPT-4o, GPT-4o-mini) and Anthropic (Claude 3.5 Haiku)
- **Sentiment classification**: Accuracy, precision, recall, F1 metrics with sklearn
- **Hallucination detection**: Tests for factual accuracy and knowledge boundaries
- **Performance monitoring**: Latency benchmarks, batch processing, token usage tracking
- **Production readiness**: Smoke tests for API health, streaming, cross-model consistency

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd ai-models-comparison-testing
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and ANTHROPIC_API_KEY

# Run tests
pytest -v
```

## Test Modules

| Module | Description |
|--------|-------------|
| `test_basic.py` | Basic API connectivity and response format validation |
| `test_metrics.py` | Classification accuracy, F1-score, per-class metrics, edge cases |
| `test_hallucinations.py` | Factual accuracy, knowledge boundaries, made-up content detection |
| `test_advanced.py` | Consistency, cost tracking, temperature effects |
| `test_monitoring.py` | Response time benchmarks, batch vs sequential performance, token usage |
| `test_production_readiness.py` | System prompt effectiveness, streaming, cross-model smoke tests |
| `test_anthropic.py` | Anthropic-specific API tests |

## Running Tests

```bash
# Run all tests
pytest -v

# Run specific module
pytest tests/test_metrics.py -v

# Run with output visible
pytest -v -s

# Run specific test
pytest tests/test_metrics.py::test_sentiment_classification_basic -v

# Generate HTML report
pytest --html=reports/report.html

# Run with timeout (useful for API tests)
pytest --timeout=60
```

## Project Structure

```
ai-models-comparison-testing/
├── tests/
│   ├── config.py          # Model names, pricing constants
│   ├── conftest.py        # Pytest fixtures (clients, datasets)
│   ├── helpers.py         # Shared utilities (API calls, metrics)
│   ├── test_*.py          # Test modules
├── data/
│   ├── sentiment_dataset.json  # Sentiment test dataset
│   └── edge_cases.json         # Edge case test dataset
├── reports/               # Generated test reports
├── .env                   # API keys (not committed)
├── .pre-commit-config.yaml
├── requirements.txt
└── README.md
```

## Configuration

Models and pricing are configured in `tests/config.py`:

```python
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL_COMPARE = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
```

## Helpers

Key helper functions in `tests/helpers.py`:

| Function | Purpose |
|----------|---------|
| `classify_sentiment()` | Single text classification with provider routing |
| `classify_cases()` | Batch classification with success rate and failure tracking |
| `classify_dataset()` | Full dataset evaluation with metrics |
| `measure_latencies()` | Response time measurement |
| `classify_batch()` | Multi-item single-prompt classification |
| `compute_metrics()` | Accuracy, precision, recall, F1 calculation |
| `format_failures()` | Format failure details for assertion messages |

## Development

```bash
# Format code
black tests/

# Check formatting
black --check tests/

# Pre-commit hooks (auto-format on commit)
pre-commit install
```

## Requirements

- Python 3.10+
- OpenAI API key
- Anthropic API key
- Dependencies: pytest, openai, anthropic, scikit-learn, python-dotenv

## License

MIT
