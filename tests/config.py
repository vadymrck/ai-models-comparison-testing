# Model constants

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL_COMPARE = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"

# Pricing per 1M tokens (as of 2025)
PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}
