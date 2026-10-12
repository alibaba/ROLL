"""M1 (transformers side): make ``AutoConfig`` recognise ``qwen4_exp``.

M1's other half registers the architecture with *mca*'s config registry. That is a
separate registry from ``transformers``, and several things go through the
transformers one:

  * ``mcore_adapter/tools/convert.py`` calls ``AutoConfig.from_pretrained`` at the
    END of a conversion -- so without this, a 336 GiB conversion runs to completion
    and then raises.
  * ROLL's data/tokenizer paths call it too.

Without registration:

    ValueError: The checkpoint you are trying to load has model type `qwen4_exp`
    but Transformers does not recognize this architecture.

The config below is intentionally a *carrier*, not a reimplementation: it keeps
whatever the checkpoint's ``config.json`` contains, including the nested
``text_config`` and ``vision_config``, and does not validate or default fields.
Anything that needs a specific field reads it through the mca template's dotted
keys (``text_config.hidden_size`` and friends), which is where the real mapping
lives.

Why not ``trust_remote_code``: the checkpoint ships no modelling code, so there is
nothing remote to trust -- the architecture is implemented on our side (M2/M3/M6/M7).

Usage -- import once, before anything calls AutoConfig:

    from register_hf_config import register_qwen4_exp
    register_qwen4_exp()
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class Qwen4ExpTextConfig(PretrainedConfig):
    """Carrier for the checkpoint's ``text_config`` block.

    Keeps every key verbatim. ``PretrainedConfig`` would otherwise drop unknown
    fields, and this architecture has many that transformers has never seen
    (``hc_count``, ``indexer_budget``, ``ngram_vocab_size_base``, ...).
    """

    model_type = "qwen4_exp_text"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(**kwargs)


class Qwen4ExpVisionConfig(PretrainedConfig):
    """Carrier for ``vision_config``. Present so the field survives a round trip;
    the vision tower is not used in language-model training."""

    model_type = "qwen4_exp_vision"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(**kwargs)


class Qwen4ExpConfig(PretrainedConfig):
    """Top-level config: holds the nested sub-configs plus top-level keys."""

    model_type = "qwen4_exp"
    sub_configs = {
        "text_config": Qwen4ExpTextConfig,
        "vision_config": Qwen4ExpVisionConfig,
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Qwen4ExpTextConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = Qwen4ExpVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(**kwargs)


def register_qwen4_exp(force: bool = False) -> bool:
    """Register the config classes with ``AutoConfig``. Idempotent.

    Returns True if a registration happened, False if it was already present.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    already = "qwen4_exp" in CONFIG_MAPPING
    if already and not force:
        return False

    for model_type, cls in (
        ("qwen4_exp_text", Qwen4ExpTextConfig),
        ("qwen4_exp_vision", Qwen4ExpVisionConfig),
        ("qwen4_exp", Qwen4ExpConfig),
    ):
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:
            # already registered (possibly by an earlier import in this process)
            pass
    return True


__all__ = [
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
    "register_qwen4_exp",
]
