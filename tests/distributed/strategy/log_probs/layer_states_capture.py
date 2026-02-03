import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

_capture_info = None


class LayerStatesCapture:

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._update_from_env_or_meta()
            self._initialized = True

    def _update_from_env_or_meta(self):
        """Update capture settings from environment variables or global _capture_info."""
        global _capture_info

        # First check global _capture_info (set from meta_info)
        if _capture_info is not None:
            self.save_dir = _capture_info.get("save_dir")
            self.prefix = _capture_info.get("prefix", "capture")
            self.global_step = _capture_info.get("step", 0)
            self.batch_idx = _capture_info.get("batch_idx", 0)
        else:
            # Fall back to environment variables
            self.save_dir = os.getenv("LAYER_STATES_SAVE_DIR", None)
            self.prefix = os.getenv("LAYER_STATES_PREFIX", "capture")
            self.global_step = int(os.getenv("LAYER_STATES_STEP", "0"))
            self.batch_idx = int(os.getenv("LAYER_STATES_BATCH", "0"))

        self.enabled = self.save_dir is not None

    def update_from_meta_info(self, meta_info: Dict):
        """Update capture settings from DataProto meta_info."""
        global _capture_info
        if "_capture_layer_states" in meta_info:
            _capture_info = meta_info["_capture_layer_states"]
            self._update_from_env_or_meta()
        else:
            _capture_info = None
            self._update_from_env_or_meta()

    def save_tensor(self, tensor: torch.Tensor, name: str, subdir: str = ""):
        """Save a tensor to disk if capture is enabled."""
        # Refresh settings before each save
        self._update_from_env_or_meta()

        if not self.enabled:
            return

        # Optional: gather CP (Ulysses) sharded sequence tensors before saving.
        # This is meant for debugging context-parallel divergence:
        # - We only gather common "sequence-shaped" tensors (ndim == 3), e.g. (bs, seq, hidden)
        # - We concatenate on dim=1 by default (the seq dimension)
        # - We save only on rank0 to avoid duplicate files
        #
        # Enable with:
        # - LAYER_STATES_CP_GATHER=1
        # Optional knobs:
        # - LAYER_STATES_CP_GATHER_DIM (default: 1)
        # - LAYER_STATES_CP_GATHER_SAVE_LOCAL=1 (also save local shard under original name)
        do_cp_gather = os.getenv("LAYER_STATES_CP_GATHER", "0") == "1"
        gather_dim = int(os.getenv("LAYER_STATES_CP_GATHER_DIM", "1"))
        save_local = os.getenv("LAYER_STATES_CP_GATHER_SAVE_LOCAL", "0") == "1"

        gathered_tensor: torch.Tensor | None = None
        if (
            do_cp_gather
            and isinstance(tensor, torch.Tensor)
            and tensor.ndim == 3
            and dist.is_available()
            and dist.is_initialized()
        ):
            try:
                # Prefer the dedicated CP/Ulysses group if available; otherwise fall back to WORLD.
                try:
                    from roll.utils.context_parallel.globals import (
                        get_ulysses_group,
                    )  # local import for test-only util

                    group = get_ulysses_group()
                except Exception:
                    group = dist.group.WORLD

                world = dist.get_world_size(group=group)
                if world > 1 and gather_dim < tensor.ndim:
                    # Assumes equal shapes across ranks for the gathered dim (true for padded CP and non-rmpad tests).
                    parts = [torch.empty_like(tensor) for _ in range(world)]
                    dist.all_gather(parts, tensor, group=group)
                    gathered_tensor = torch.cat(parts, dim=gather_dim)

                    if dist.get_rank(group=group) != 0:
                        # Non-zero ranks participate in all_gather but do not write files.
                        return
            except Exception:
                # Never fail training/tests due to debug capture logic.
                gathered_tensor = None

        save_path = Path(self.save_dir)
        if subdir:
            save_path = save_path / subdir
        save_path.mkdir(parents=True, exist_ok=True)

        if gathered_tensor is not None:
            if save_local:
                local_path = save_path / f"{self.prefix}_step{self.global_step}_batch{self.batch_idx}_{name}.pt"
                torch.save(tensor.cpu().detach(), local_path)

            file_path = save_path / f"{self.prefix}_step{self.global_step}_batch{self.batch_idx}_{name}_gathered.pt"
            torch.save(gathered_tensor.cpu().detach(), file_path)
        else:
            file_path = save_path / f"{self.prefix}_step{self.global_step}_batch{self.batch_idx}_{name}.pt"
            torch.save(tensor.cpu().detach(), file_path)

    def save_dict(self, data: Dict[str, Any], name: str, subdir: str = ""):
        """Save a dictionary of tensors."""
        # Refresh settings before each save
        self._update_from_env_or_meta()

        if not self.enabled:
            return

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                self.save_tensor(value, f"{name}_{key}", subdir)
            elif isinstance(value, dict):
                self.save_dict(value, f"{name}_{key}", subdir)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    for i, tensor in enumerate(value):
                        self.save_tensor(tensor, f"{name}_{key}_{i}", subdir)


# Global instance
_capture = LayerStatesCapture()


def save_tensor(tensor: torch.Tensor, name: str, subdir: str = ""):
    """Convenience function to save a tensor."""
    _capture._update_from_env_or_meta()  # Refresh settings
    _capture.save_tensor(tensor, name, subdir)


def save_dict(data: Dict[str, Any], name: str, subdir: str = ""):
    """Convenience function to save a dict."""
    _capture._update_from_env_or_meta()  # Refresh settings
    _capture.save_dict(data, name, subdir)


def is_enabled() -> bool:
    """Check if capture is enabled."""
    _capture._update_from_env_or_meta()  # Refresh settings
    return _capture.enabled
