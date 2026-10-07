"""CPU-only contracts for the exact runtime chat prompt rendering helpers."""
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from roll.pipeline.tinker_backend_runtime import chat_prompt as module


def test_import_does_not_require_site_packages_or_training_dependencies():
    root = Path(__file__).resolve().parents[2]
    script = (f'import sys; sys.path.insert(0, {str(root)!r}); '
              'from roll.pipeline.tinker_backend_runtime.chat_prompt import _render_prompt_token_ids; '
              'assert not any(name in sys.modules for name in ("torch", "tensordict", "megatron", "transformers"))')
    subprocess.run([sys.executable, '-S', '-c', script], check=True, env={'PATH': os.defpath})


def test_normalizes_content_and_tool_arguments_without_mutating_input():
    messages = [{'role': 'assistant', 'content': [{'text': 'a'}, {'content': 'b'}, 3],
                 'tool_calls': [{'id': 'x', 'function': {'name': 'run', 'arguments': '{"cmd":"pwd"}'}}]},
                {'role': 'assistant', 'tool_calls': [{'function': {'arguments': 'invalid'}}]},
                {'role': 'user', 'content': None}, 'unchanged']
    original = copy.deepcopy(messages)
    normalized = module._normalize_messages_for_chat_template(messages)
    assert messages == original
    assert normalized[0]['content'] == 'a\nb\n3'
    assert normalized[0]['tool_calls'][0]['function']['arguments'] == {'cmd': 'pwd'}
    assert normalized[1]['tool_calls'][0]['function']['arguments'] == 'invalid'
    assert normalized[2]['content'] == 'None'
    assert normalized[3] == 'unchanged'
    assert module._normalize_messages_for_chat_template(None) is None


@pytest.mark.parametrize('returned', [[1, '2'], {'input_ids': [[1, 2]]},
    SimpleNamespace(input_ids=[1, 2]), SimpleNamespace(tolist=lambda: [[1, 2]])])
def test_render_returns_real_ids_and_preserves_tools_and_thinking_flag(returned):
    calls = []
    def apply(messages, **kwargs):
        calls.append((messages, kwargs))
        return returned
    request = {'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}],
               'tools': [{'type': 'function', 'function': {'name': 'run'}}]}
    assert module._render_prompt_token_ids(SimpleNamespace(apply_chat_template=apply), request) == [1, 2]
    assert calls == [([{'role': 'user', 'content': 'hi'}], {
        'tools': request['tools'], 'tokenize': True, 'add_generation_prompt': True, 'enable_thinking': False})]


def test_legacy_tokenizer_without_thinking_keyword_remains_supported():
    calls = []
    def apply(messages, *, tools, tokenize, add_generation_prompt):
        calls.append(messages)
        return (11, 12)
    assert module._render_prompt_token_ids(SimpleNamespace(apply_chat_template=apply), {'messages': []}) == [11, 12]
    assert calls == [[]]


def test_render_errors_are_explicit_not_silent_token_substitutions():
    with pytest.raises(RuntimeError, match='requires a tokenizer'):
        module._render_prompt_token_ids(None, {'messages': []})
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match='must be a list'):
        module._render_prompt_token_ids(tokenizer, {'messages': 'bad'})
    with pytest.raises(RuntimeError, match='returned None'):
        module._render_prompt_token_ids(tokenizer, {'messages': []})
    tokenizer.apply_chat_template = lambda *args, **kwargs: {'missing_input_ids': []}
    with pytest.raises(RuntimeError, match='did not return input_ids'):
        module._render_prompt_token_ids(tokenizer, {'messages': []})
    def fail(*args, **kwargs):
        raise ValueError('template error')
    tokenizer.apply_chat_template = fail
    with pytest.raises(RuntimeError, match='failed to tokenize structured sample prompt: template error'):
        module._render_prompt_token_ids(tokenizer, {'messages': []})
