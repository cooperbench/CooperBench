#!/usr/bin/env python3
"""Quick smoke test to verify a model works via LiteLLM."""

import argparse
import sys

from dotenv import load_dotenv
load_dotenv()

import litellm


def test_model(model_name: str, prompt: str = "Say hello in one sentence.") -> bool:
    """Test a specific model and return True if it works."""
    print(f"Testing: {model_name}")
    print("-" * 40)

    try:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        print(f"Response: {response.choices[0].message.content}")
        print(f"Model:    {response.model}")
        print(f"Usage:    {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:300]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test a model via LiteLLM")
    parser.add_argument("models", nargs="+", help="Model name(s) to test, e.g. anthropic/MiniMax-M2.5")
    parser.add_argument("--prompt", default="Say hello in one sentence.", help="Prompt to send")
    args = parser.parse_args()

    results = {}
    for model in args.models:
        print()
        ok = test_model(model, args.prompt)
        results[model] = ok
        print()

    # Summary
    if len(results) > 1:
        print("=" * 40)
        print("SUMMARY")
        print("=" * 40)
        for model, ok in results.items():
            status = "pass" if ok else "FAIL"
            print(f"  [{status}] {model}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
