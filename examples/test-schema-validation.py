#!/usr/bin/env python3
"""Test script to validate the schema against the example output.

This confirms that examples/output.json is a valid Insight object.
Run from repo root: python examples/test-schema-validation.py
"""

import json
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import Insight


def main():
    """Validate example output against schema."""
    example_path = Path(__file__).parent / "output.json"

    print(f"[test] validating {example_path}...")

    with open(example_path) as f:
        example_data = json.load(f)

    try:
        # This will raise ValidationError if schema is violated
        insight = Insight(**example_data)
        print(f"[test] ✓ schema validation passed")
        print(f"[test] ✓ watch_full: {insight.decision.watch_full}")
        print(f"[test] ✓ confidence: {insight.decision.confidence}")
        print(f"[test] ✓ key_concepts: {len(insight.key_concepts)}")
        print(f"[test] ✓ metadata.llm_model: {insight.metadata.llm_model}")
        return 0
    except Exception as e:
        print(f"[test] ✗ validation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
