# AI QA Test Suite

Testing OpenAI GPT and Anthropic Claude APIs for behavioral consistency, hallucination detection, and classification accuracy.

## Setup

1. Clone this repo
2. Create virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Add your API keys:
   - Copy `.env.example` to `.env`
   - Add your OpenAI and Anthropic API keys

## Running Tests
```bash
# Run all tests
pytest -v

# Run with detailed output
pytest -v -s

# Generate HTML report
pytest --html=reports/report.html
```

## Project Structure

- `tests/` - Test files
- `data/` - Test datasets
- `docs/` - Documentation and demos
- `reports/` - Generated test reports

## Progress

- [ ] Week 1: Basic functionality tests (5 tests)
- [ ] Week 2: Hallucination detection (5 tests)
- [ ] Week 3: Classification & metrics (10 tests)
- [ ] Week 4: Multi-model comparison (10 tests)