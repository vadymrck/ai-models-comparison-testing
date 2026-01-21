import json
import os

import pytest
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@pytest.fixture
def openai_client():
    """Setup OpenAI client for all tests"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=api_key)


@pytest.fixture
def anthropic_client():
    """Setup Anthropic client for all tests"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in .env")
    return Anthropic(api_key=api_key)


@pytest.fixture
def sentiment_dataset():
    """Load sentiment test dataset"""
    dataset_path = "data/sentiment_dataset.json"

    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    return data


@pytest.fixture
def edge_cases():
    """Load edge cases test dataset"""
    dataset_path = "data/edge_cases.json"

    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    return data
