"""Keep ROLL's legacy warmup-ratio configuration usable with Transformers v5."""
import pytest

from mcore_adapter.training_args import TrainingArguments, Seq2SeqTrainingArguments


@pytest.mark.parametrize("arguments_cls", [TrainingArguments, Seq2SeqTrainingArguments])
@pytest.mark.parametrize("ratio,steps,expected", [(0.0, 0, 0), (0.1, 0, 11), (1.0, 0, 101), (0.5, 3, 3)])
def test_legacy_warmup_ratio_preserves_rounding_and_step_precedence(arguments_cls, ratio, steps, expected):
    args = arguments_cls(output_dir="/tmp/qwen38-warmup-test", use_cpu=True, report_to=[],
                         warmup_ratio=ratio, warmup_steps=steps)
    assert args.get_warmup_steps(101) == expected


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_invalid_legacy_warmup_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="warmup_ratio"):
        TrainingArguments(output_dir="/tmp/qwen38-warmup-test", use_cpu=True, report_to=[], warmup_ratio=ratio)


def test_native_fractional_warmup_steps_keeps_transformers_semantics():
    from transformers import TrainingArguments as HFTrainingArguments
    options = dict(output_dir="/tmp/qwen38-warmup-test", use_cpu=True, report_to=[], warmup_steps=0.25)
    expected = HFTrainingArguments(**options).get_warmup_steps(101)
    actual = TrainingArguments(**options).get_warmup_steps(101)
    assert actual == expected
