import time
from openai import RateLimitError

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
