import random
import string
from typing import List, Tuple


def generate_random_cli_task() -> Tuple[str, List[str], str, str]:
    """
    Generate a random CLI task with target files and verification description.

    Returns:
        A tuple containing (task_description, target_files, action_commands, verification_description)
    """
    tasks = [
        # (
        #     "Create a test.txt file and enter Hello World",
        #     ["test.txt"],
        #     ["<answer>write_file test.txt 'Hello World'</answer>"],
        #     "Check that test.txt exists and contains 'Hello World'. Use read_file to verify the content.",
        # ),
        (
            "Create a Python script hello.py that prints 'Hello CLI'",
            ["hello.py"],
            ["<answer>write_file hello.py 'print(\"Hello CLI\")'</answer>"],
            "Check that hello.py exists and contains Python code. Verify by running the script with python3 hello.py.",
        ),
        # (
        #     "Create a directory named 'test_dir' and a file inside it named 'info.txt' with content 'Test data'",
        #     ["test_dir/info.txt"],
        #     [
        #         "<answer>run_shell_command mkdir -p test_dir</answer>",
        #         "<answer>write_file test_dir/info.txt 'Test data'</answer>"
        #     ],
        #     "Check that test_dir directory exists and contains info.txt with content 'Test data'. Use list_directory and read_file to verify.",
        # ),
        # (
        #     "Search for all Python files in the current directory",
        #     [],
        #     ["<answer>search_file_content '*.py'</answer>"],
        #     "Verify that the search returned Python files. Check the output contains .py file names.",
        # ),
        # (
        #     "Create a todo list with three items: setup, develop, test",
        #     ["todo.txt"],
        #     [
        #         "<answer>write_file todo.txt '1. Setup environment\n2. Develop features\n3. Test functionality'</answer>"
        #     ],
        #     "Check that todo.txt exists and contains three todo items. Use read_file to verify the content format.",
        # )
    ]

    return random.choice(tasks)


def validate_cli_command(command: str) -> bool:
    dangerous_commands = [
        "rm -rf /",
        "sudo",
        "shutdown",
        "reboot",
    ]

    command_lower = command.lower().strip()

    for dangerous in dangerous_commands:
        if dangerous.lower() in command_lower:
            return False

    return True


def create_iflow_call(prompt: str) -> str:
    return f'/root/.npm-global/bin/iflow -y -p "{prompt}"'


def create_iflow_search(params: str) -> str:
    json_parms = f'{{"tool_calls":[{{"id":"tool_search","function":{{"name":"web_search","arguments":"{{\\"query\\":\\"{params}\\"}}"}},"type":"function"}}]}}'
    return f"/root/.npm-global/bin/iflow -t '{json_parms}'"


def create_iflow_shell(params: str) -> str:
    json_parms = f'{{"tool_calls":[{{"id":"tool_shell","function":{{"name":"run_shell_command","arguments":"{{\\"command\\":\\"{params}\\"}}"}},"type":"function"}}]}}'
    return f"/root/.npm-global/bin/iflow -t '{json_parms}'"


def create_iflow_read_file(params: str) -> str:
    json_parms = f'{{"tool_calls":[{{"id":"tool_read_file","function":{{"name":"read_file","arguments":"{{\\"absolute_path\\":\\"{params}\\"}}"}},"type":"function"}}]}}'
    return f"/root/.npm-global/bin/iflow -t '{json_parms}'"
