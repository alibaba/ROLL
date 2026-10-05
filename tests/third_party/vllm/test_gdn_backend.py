import pytest

from roll.utils.qwen38_gdn import configure_qwen38_gdn_backend


def test_qwen38_flash_next_defaults_to_triton_prefill_backend():
    config = {}

    configure_qwen38_gdn_backend(config, "/models/Qwen3.8-Flash-Next")

    assert config["additional_config"]["gdn_prefill_backend"] == "triton"


def test_explicit_gdn_backend_is_preserved():
    config = {"additional_config": {"gdn_prefill_backend": "flashinfer"}}

    configure_qwen38_gdn_backend(config, "/models/Qwen3.8-Flash-Next")

    assert config["additional_config"]["gdn_prefill_backend"] == "flashinfer"


def test_other_models_are_unchanged():
    config = {}

    configure_qwen38_gdn_backend(config, "/models/Qwen3.5-27B")

    assert config == {}


@pytest.mark.parametrize("path", [
    "/experiments/Qwen3.8-Flash-Next/baselines/Qwen3.5-27B",
    "/models/Qwen3.8-Flash-Next/checkpoint-99",
    "/models/unrelated-Qwen3.8-Flash-Next",
])
def test_ancestor_or_embedded_model_name_does_not_override_another_model(path):
    config = {"additional_config": {"unrelated_option": 1}}
    configure_qwen38_gdn_backend(config, path)
    assert config == {"additional_config": {"unrelated_option": 1}}


@pytest.mark.parametrize("path", ["Qwen/Qwen3.8-Flash-Next", "/models/Qwen3.8_Flash-Next/",
                                  "/models/Qwen3.8-Flash-Next-BF16"])
def test_model_leaf_and_repository_names_select_the_default(path):
    config = {}
    configure_qwen38_gdn_backend(config, path)
    assert config == {"additional_config": {"gdn_prefill_backend": "triton"}}
