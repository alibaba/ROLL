"""Preserve Flash-Next's external frozen N-gram parameters across level-2 sleep."""

import re

import torch


_MODEL_TYPES = {"qwen4_exp", "qwen4_exp_text", "qwen3_8_flash_next", "qwen3_8_flash_next_text"}
_TABLE = re.compile(r"(?:^|\.)layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight$")


def configure_frozen_ngram_loading(hf_config, load_config):
    """Load external tables that are absent from the actor's parameter stream.

    ROLL normally initializes full-parameter rollout workers with dummy weights.
    Flash-Next's frozen tables instead live outside named_parameters on the
    training side. They must be read from the original checkpoint once, before
    the first full weight synchronization and subsequent sleep snapshots.
    """
    if getattr(hf_config, "model_type", None) not in _MODEL_TYPES:
        return
    config = getattr(hf_config, "text_config", hf_config)
    if getattr(config, "ple_layer_ids", None) and load_config.load_format == "dummy":
        load_config.load_format = "auto"


@torch.no_grad()
def restore_ngram_context_offsets(model_runner, hf_config):
    """Rebuild PLE input metadata discarded with native level-2 weights.

    vLLM's V2 runner creates model_state inside the weights CuMem pool.
    Its offsets are not nn.Module buffers, so native buffer restoration misses
    them. Other PLE input workspaces are overwritten by prepare_inputs; these
    constant offsets must be restored in place to preserve captured addresses.
    """
    if getattr(hf_config, "model_type", None) not in _MODEL_TYPES:
        return
    state = getattr(model_runner, "model_state", None)
    if state is None or not getattr(state, "uses_ngram_embedding", False):
        return
    offsets = getattr(state, "ngram_context_offsets", None)
    length = getattr(state, "ngram_context_len", None)
    if (type(length) is not int or length < 1 or not isinstance(offsets, torch.Tensor)
            or offsets.shape != (length,) or offsets.dtype != torch.int64):
        raise ValueError("Unsupported Flash-Next N-gram context offsets")
    offsets.copy_(torch.arange(-length, 0, dtype=offsets.dtype, device=offsets.device))


def _tables(model, layer_ids):
    tables, found_layers = {}, []
    for name, parameter in model.named_parameters():
        match = _TABLE.search(name)
        if match is None:
            continue
        if (parameter.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or parameter.device.type == "meta" or parameter.ndim != 2
                or not parameter.is_contiguous()):
            raise ValueError(f"Unsupported frozen N-gram table storage: {name}")
        tables[name] = parameter
        found_layers.append(int(match.group(1)))
    if set(found_layers) != layer_ids or len(found_layers) != len(layer_ids):
        raise ValueError("Frozen N-gram tables do not match configured PLE layers; "
                         "level-2 sleep requires complete GPU-resident tables")
    return tables


@torch.no_grad()
def _copy_chunks(destination, source, chunk_bytes):
    elements = max(1, chunk_bytes // source.element_size())
    destination, source = destination.view(-1), source.view(-1)
    for start in range(0, source.numel(), elements):
        destination[start:start + elements].copy_(source[start:start + elements])


class FrozenNGramSleepState:
    @classmethod
    def capture(cls, model, hf_config, *, chunk_bytes=64 * 1024 * 1024, lora_enabled=False):
        """Back up only rank-local frozen tables, before native weight discard.

        The trainable backbone must subsequently be supplied by ROLL's full
        weight update. LoRA-only updates cannot restore a discarded backbone.
        CPU-offloaded PLE and partial pipeline stages are not supported here.
        """
        if getattr(hf_config, "model_type", None) not in _MODEL_TYPES:
            return None
        if lora_enabled:
            raise ValueError("Flash-Next level-2 sleep requires full model weight synchronization; use level 1 for LoRA")
        if chunk_bytes < 4:
            raise ValueError("N-gram transfer chunk_bytes must be at least 4")
        config = getattr(hf_config, "text_config", hf_config)
        configured = getattr(config, "ple_layer_ids", None)
        if configured is None or any(type(index) is not int or index < 1 for index in configured):
            raise ValueError("Invalid N-gram PLE layer configuration")
        layer_ids = {index - 1 for index in configured}
        tables = _tables(model, layer_ids)
        if not tables:
            return None
        state = cls()
        state._layer_ids, state._chunk_bytes = layer_ids, chunk_bytes
        state._weights = {}
        for name, parameter in tables.items():
            # Pageable storage avoids a second full-size pinning allocation.
            saved = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
            _copy_chunks(saved, parameter.detach(), chunk_bytes)
            state._weights[name] = saved
        return state

    @property
    def cpu_bytes(self):
        return sum(value.numel() * value.element_size() for value in self._weights.values())

    def restore(self, model):
        """Restore tables after native weight wake, before marking weights ready."""
        tables = _tables(model, self._layer_ids)
        if set(tables) != set(self._weights):
            raise ValueError("N-gram parameter names changed during sleep")
        for name, parameter in tables.items():
            saved = self._weights[name]
            if parameter.shape != saved.shape or parameter.dtype != saved.dtype:
                raise ValueError(f"N-gram parameter geometry changed during sleep: {name}")
        for name, parameter in tables.items():
            _copy_chunks(parameter, self._weights[name], self._chunk_bytes)
        # Retain the snapshot on failure, and release it only after every
        # blocking copy completed. No original GPU tensor is kept by state.
        self._weights.clear()
