# Code Sandbox Reward Worker

The `code_sandbox_reward_worker.py` module provides functionality to evaluate code solutions in a sandbox environment and compute rewards based on test case results. It supports multiple testing modes to accommodate different types of code evaluation scenarios.

## Overview

The Code Sandbox Reward Worker evaluates code solutions by:
1. Extracting code from model responses
2. Running the code against test cases
3. Computing rewards based on test results
4. Providing detailed feedback on test failures

The worker supports both HTTP sandbox testing (remote execution) and local testing, making it flexible for different deployment scenarios.

## Test Case Types

The worker supports five different test case types, each with its own format and requirements:

### 1. Assert Testing

Used for simple assertion-based testing of code.

**Format:**
```json
[
  {
    "assert_code": "assert find_binary_numbers(2) == ['00', '01', '10']"
  },
  {
    "assert_code": "assert find_binary_numbers(3) == ['000', '001', '010', '100', '101']"
  },
  {
    "assert_code": "assert find_binary_numbers(1) == ['0', '1']"
  }
]
```

**Key Components:**
- `assert_code`: Simple assert statements that test the function

**Example Use Case:**
Testing functions with straightforward inputs and expected outputs.

### 2. Pytest Testing

Used for more complex test cases using pytest-style test functions.

**Format:**
```json
{
  "assert_code": "\n\n\ndef test_even_length_string_with_reverse_parts():\n    assert can_split_into_reverse_parts(\"abccba\") == True\n\ndef test_even_length_string_without_reverse_parts():\n    assert can_split_into_reverse_parts(\"abcdef\") == False\n\ndef test_odd_length_string():\n    assert can_split_into_reverse_parts(\"abcba\") == False\n\ndef test_empty_string():\n    assert can_split_into_reverse_parts(\"\") == True\n\ndef test_single_character_string():\n    assert can_split_into_reverse_parts(\"a\") == False\n\ndef test_string_with_mixed_cases():\n    assert can_split_into_reverse_parts(\"AbCCba\") == False\n\ndef test_palindrome_string():\n    assert can_split_into_reverse_parts(\"abccba\") == True\n    assert can_split_into_reverse_parts(\"abcdedcba\") == False"
}
```

**Key Components:**
- `assert_code`: Contains multiple test functions with assertions

**Example Use Case:**
Testing functions that require multiple test cases with different scenarios.

### 3. Input/Output Testing

Used for testing code with standard input and expected output.

**Format:**
```json
[
  {
    "stdin": "[1, 2, 3]",
    "expected_stdout": "9"
  },
  {
    "stdin": "[1, 2, 3, 4]",
    "expected_stdout": "19"
  },
  {
    "stdin": "[1, 2, 3, 4, 5]",
    "expected_stdout": "33"
  }
]
```

**Key Components:**
- `stdin`: Input to provide to the program
- `expected_stdout`: Expected output from the program

**Example Use Case:**
Testing functions that read from standard input and write to standard output.

### 4. Check-Based Testing

Used for testing code with a specific entry point and custom imports.

**Format:**

```json
{
  "assert_code": "def check(candidate):\n    assert candidate(nums = [1,3,5,6], target = 5) == 2\n    assert candidate(nums = [1,3,5,6], target = 2) == 1\n    assert candidate(nums = [1,3,5,6], target = 7) == 4\n",
  "import_prefix": "import collections\nimport string\nimport math\nimport datetime\n\nfrom typing import *\nfrom functools import *\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nfrom operator import *\nfrom math import *\n\ninf = float('inf')\n\n",
  "entry_point": "Solution().searchInsert"
}
```

**Key Components:**

- `assert_code`: Contains the test assertions
- `import_prefix`: Imports to include before the solution code
- `entry_point`: Function or method to call for testing

**Example Use Case:**
Testing LeetCode-style problems where a specific method of a class needs to be evaluated.

### 5. Text Testing

Used for validating text responses rather than code execution.

**Format:**

```json
[
  {
    "assert_code": "import re\ndef check_keyword_highlight(input_str):\n    highlights = re.findall(r'\\\\*[^\\\\n\\\\*]+\\\\*', input_str)\n    return len(highlights) >= 1\ninput_str = {response}\nres = check_keyword_highlight(input_str)\nassert res == True"
  },
  {
    "assert_code": "import re\ndef check_title(input_str):\n    pattern = r'<<[^\\\\n]+>>'\n    re_pattern = re.compile(pattern)\n    titles = re.findall(re_pattern, input_str)\n\n    for title in titles:\n        if title.lstrip('<').rstrip('>').strip():\n            return True\n    return False\ninput_str = {response}\nres = check_title(input_str)\nassert res == True"
  }
]
```

**Key Components:**

- `assert_code`: Python code that validates the text response
- `{response}`: Placeholder that gets replaced with the model's response

**Example Use Case:**
Validating formatting, structure, or content of text responses like ensuring a response has a title, highlights, or specific number of sentences.

## Data Format

When using the Code Sandbox Reward Worker, each test case should include:

1. `id`: A unique identifier for the test case
2. `prompt`: The problem statement or question
3. `case_type`: The type of test case (one of: "check_based", "text", "assert", "pytest", "input")
4. `test_case_function`: The function name to test (if applicable)
5. `test_cases`: The test cases in the appropriate format for the case type
6. `tag`: Optional tag for categorizing test cases

Example:

```json
{
  "id": "3c45c692be4866bcf8922c7825ffe0bd00e5539034725594a2e24512f44834b5",
  "domain": "code_sandbox",
  "source": "leetcode",
  "difficulty": "0",
  "prompt": "You are an expert Python programmer...",
  "case_type": "check_based",
  "test_case_function": "Solution().searchInsert",
  "test_cases": "[{\"assert_code\": \"def check(candidate):\\n    assert candidate(nums = [1,3,5,6], target = 5) == 2\\n    assert candidate(nums = [1,3,5,6], target = 2) == 1\\n    assert candidate(nums = [1,3,5,6], target = 7) == 4\\n\", \"import_prefix\": \"import collections\\nimport string\\nimport math\\nimport datetime\\n\\nfrom typing import *\\nfrom functools import *\\nfrom collections import *\\nfrom itertools import *\\nfrom heapq import *\\nfrom bisect import *\\nfrom string import *\\nfrom operator import *\\nfrom math import *\\n\\ninf = float('inf')\\n\\n\", \"entry_point\": \"Solution().searchInsert\"}]",
  "tag": "leetcode-Easy"
}
```

## Important Considerations

### Local vs. HTTP Sandbox Testing

The worker supports two testing modes:

1. **Local Testing**: Executes code locally using Python's exec/eval
   - Faster but less secure
   - Good for development and testing
   - Set `use_local=True` in the worker config

2. **HTTP Sandbox Testing**: Executes code in a remote sandbox
   - More secure but requires a sandbox service
   - Good for production use
   - Provide `code_url` in the worker config

## Usage

To use the Code Sandbox Reward Worker:

1. Create a worker configuration with the appropriate settings:

```python
from roll.pipeline.rlvr.rlvr_config import RewardConfig

config = RewardConfig(
    use_local=True,  # Set to False for HTTP sandbox
    code_url="http://your-sandbox-url.com/execute",  # Only needed for HTTP sandbox
    model_args={...}  # Model configuration
)
```

2. Initialize the worker:

```python
from roll.pipeline.rlvr.rewards.code_sandbox_reward_worker import CodeSandboxRewardWorker

worker = CodeSandboxRewardWorker(config)
```

3. Compute rewards:

```python
from roll.distributed.scheduler.protocol import DataProto

# Prepare data with prompts, responses, and test cases
data = DataProto.from_dict(...)

# Compute rewards
results = worker.compute_rewards(data)
```

