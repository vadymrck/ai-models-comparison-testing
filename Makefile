.PHONY: test test-quick test-monitoring report clean install setup

# Install dependencies
install:
	pip install -r requirements.txt

# Setup project
setup: install
	@echo "✅ Project setup complete"

# Run all tests
test:
	pytest tests/ -v

# Run quick smoke test
test-quick:
	pytest tests/test_basic.py -v

# Run performance tests
test-monitoring:
	pytest tests/test_monitoring.py -v -s

# Run tests with coverage
test-coverage:
	pytest tests/ -v --cov=. --cov-report=html

# Generate HTML report
report:
	pytest tests/ --html=reports/ai-qa-test-report.html --self-contained-html -v
	@echo "\n✅ Report generated: reports/ai-qa-test-report.html"

# Clean artifacts
clean:
	rm -rf reports/*.html reports/*.json
	rm -rf .pytest_cache
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Run everything
all: clean test report
	@echo "\n✅ All tests complete! Check reports/ for results"