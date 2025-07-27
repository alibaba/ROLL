#!/usr/bin/env python3
"""
Test script to run the updated 1-5 scale evaluation
"""

import subprocess
import sys
import os


def test_quick_eval():
    """Test the quick evaluation script"""
    print("Testing quick evaluation (1-5 scale)...")

    cmd = [
        sys.executable,
        "quick_eval.py",
        "--ground-truth",
        "dialogue_dataset_all_v5_summarized.jsonl",
        "--model-results",
        "qwen_multicard_results.jsonl",
        "--model-name",
        "Qwen-Multicard-Test",
        "--output",
        "test_quick_results.json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Quick evaluation completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Quick evaluation failed: {e.stderr}")
        return False


def test_automated_eval():
    """Test the automated evaluation script"""
    print("\nTesting automated evaluation (1-5 scale)...")

    cmd = [
        sys.executable,
        "automated_eval.py",
        "--ground-truth",
        "dialogue_dataset_all_v5_summarized.jsonl",
        "--model-results",
        "qwen_multicard_results.jsonl",
        "--output",
        "test_automated_results.txt",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Automated evaluation completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Automated evaluation failed: {e.stderr}")
        return False


def main():
    print("=" * 60)
    print("TESTING UPDATED EVALUATION SYSTEM (1-5 SCALE)")
    print("=" * 60)

    # Check if required files exist
    required_files = ["dialogue_dataset_all_v5_summarized.jsonl", "qwen_multicard_results.jsonl"]

    for file in required_files:
        if not os.path.exists(file):
            print(f"✗ Required file not found: {file}")
            return 1

    print("✓ All required files found")

    # Test evaluations
    success = True
    success &= test_quick_eval()
    success &= test_automated_eval()

    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! 🎉")
        print("The evaluation system has been successfully updated to 1-5 scale")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED! ❌")
        print("Please check the error messages above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
