"""M1 + M3 for Qwen3.8-Flash-Next (HF model_type ``qwen4_exp``).

M1 = a config class registered under "qwen4_exp" so transformers/mca can read
     the checkpoint's config at all.
M3 = a model class that routes construction to mcore's hybrid-attention spec
     builder rather than the default all-full-attention path.

M2 (the weight conversion template) is NOT here yet; without it conversion still
fails. These two alone make the model *constructible*, which is what unblocks
writing and testing M2.

Importing this package performs the registration, mirroring how the bundled
architectures do it (see mcore_adapter/models/__init__.py).
"""

# Register with transformers' AutoConfig as well as mca's own registry: they are
# separate, and tools/convert.py + ROLL's data paths go through the former.
from .register_hf_config import register_qwen4_exp  # noqa: F401

register_qwen4_exp()

from . import template_qwen4_exp  # noqa: F401  (registers the M2 template)
from .config_qwen4_exp import Qwen4ExpConfig
from .modeling_qwen4_exp import Qwen4ExpModel
from .qsa import qsa_indexer_kl_loss, select_causal_blocks
from .ple_layer import FrozenNGramEmbedding, PLELayer


__all__ = [
    "FrozenNGramEmbedding",
    "PLELayer",
    "Qwen4ExpConfig",
    "Qwen4ExpModel",
    "qsa_indexer_kl_loss",
    "select_causal_blocks",
]
