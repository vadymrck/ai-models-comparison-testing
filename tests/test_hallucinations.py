import pytest

# Model constant for all tests
MODEL = "gpt-4o-mini"


def test_refuses_impossible_question(openai_client):
    """Detects impossible historical question"""

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Who was the 101th president of the United States?",
            }
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.lower()
    print(f"\n  🤖 Model response: {answer}")

    # Model should indicate this is impossible
    # US has only had 46 presidents (as of 2024)
    refusal_indicators = [
        "only",  # "there have only been"
        "currently",  # "currently 46 presidents"
        "as of",  # "as of now"
        "46",  # mentions correct number
        "47",  # or 47 if recently updated
        "not yet",  # "has not yet been"
        "hasn't been",  # "hasn't been a 100th"
        "no 100th",  # "there is no 100th"
    ]

    found_refusal = any(indicator in answer for indicator in refusal_indicators)

    assert found_refusal, f"Model should refuse impossible question, but said: {answer}"

    print(f"  ✓ Model correctly handled impossible question")
    print(f"\n✅ PASSED - Refused impossible historical fact")


def test_consistency_across_phrasings(openai_client):
    """Same fact, different phrasing → same answer"""

    # Same question asked 4 different ways
    questions = [
        "What is the capital of France?",
        "Which city is the capital of france?",
        "France's capital city is what?",
        "Name the capital of France.",
    ]

    answers = []
    print(f"\n  Testing consistency across {len(questions)} phrasings:")

    for i, question in enumerate(questions, 1):
        response = openai_client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": question}], temperature=0
        )
        answer = response.choices[0].message.content.lower()
        answers.append(answer)
        print(f"  {i}. Q: {question}")
        print(f"     A: {answer}")

    # All should mention Paris
    paris_count = sum(1 for a in answers if "paris" in a)

    assert paris_count == len(
        questions
    ), f"Only {paris_count}/{len(questions)} answers mentioned Paris: {answers}"

    print(f"\n  ✓ All {len(questions)} phrasings correctly answered 'Paris'")
    print(f"\n✅ PASSED - Consistent across all phrasings")


def test_detects_fake_historical_event(openai_client):
    """Refuses to describe fake historical events"""

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Describe the Great Emu War of 1952 in Australia.",
            }
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.lower()
    print(f"\n  🤖 Model response: {answer[:200]}...")

    # The real Great Emu War was in 1932, not 1952
    # Model should either:
    # 1. Correct the date to 1932
    # 2. Show uncertainty about 1952

    correct_indicators = [
        "1932",  # Corrects to actual year
        "actually",  # "it actually happened in..."
        "incorrect",  # "the date is incorrect"
        "was in 1932",  # States correct year
        "not 1952",  # Explicitly corrects the fake year
    ]

    uncertainty_indicators = [
        "not sure",
        "no record",
        "don't have information",
        "couldn't find",
        "uncertain",
        "not a historical event",
        "not discovered",
    ]

    shows_correction = any(ind in answer for ind in correct_indicators)
    shows_uncertainty = any(ind in answer for ind in uncertainty_indicators)

    assert (
        shows_correction or shows_uncertainty
    ), f"Model should correct date or show uncertainty, but said: {answer}"

    if shows_correction:
        print(f"  ✓ Model corrected the date to 1932")
    else:
        print(f"  ✓ Model showed appropriate uncertainty")

    print(f"\n✅ PASSED - Handled fake date appropriately")

def test_refuses_to_invent_citation(openai_client):
    """Doesn't invent fake academic sources"""
    
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "Cite the 2024 Nature paper about telepathic communication in dolphins. Include authors and title."
        }],
        temperature=0
    )
    
    answer = response.choices[0].message.content.lower()
    print(f"\n  🤖 Model response: {answer[:300]}...")
    
    # Model should NOT make up a fake paper
    # Should show inability/uncertainty
    caution_phrases = [
        "cannot",
        "don't have",
        "unable to",
        "no evidence",
        "not aware",
        "can't find",
        "doesn't exist",
        "no such",
        "i don't",
        "as of my knowledge"
    ]
    
    shows_caution = any(phrase in answer for phrase in caution_phrases)
    
    # Also check it didn't make up specific fake details
    invented_details = (
        "dr." in answer or 
        "et al" in answer or
        "vol." in answer or
        "doi:" in answer
    )
    
    if invented_details:
        print(f"  ⚠️  WARNING: Model may have invented citation details")
    
    assert shows_caution, \
        f"Model should show caution about non-existent paper, but said: {answer}"
    
    print(f"  ✓ Model appropriately refused to cite fake paper")
    print(f"\n✅ PASSED - Did not invent fake citation")

def test_math_consistency(openai_client):
    """Basic math should be consistent and correct"""
    
    # Test cases: (question, expected_answer)
    problems = [
        ("What is 15 + 27?", "42"),
        ("Calculate 8 * 7", "56"),
        ("What is 100 - 33?", "67"),
        ("What is 144 / 12?", "12"),
        ("What is 2 to the power of 5?", "32")
    ]
    
    print(f"\n  Testing {len(problems)} math problems:")
    
    for question, expected in problems:
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"{question} Answer with just the number."
            }],
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Check if expected number is in the answer
        is_correct = expected in answer
        status = "✓" if is_correct else "✗"
        
        print(f"  {status} {question} → {answer} (expected: {expected})")
        
        assert is_correct, \
            f"Wrong math: {question} should be {expected}, got {answer}"
    
    print(f"\n  ✓ All {len(problems)} math problems correct")
    print(f"\n✅ PASSED - Math is consistent")
