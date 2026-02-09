"""
Rollout Mock Mixin for dump/mock mechanism.

This mixin provides dump/mock functionality for schedulers to enable
deterministic testing by saving/loading DataProto objects.
"""
import os
import pickle
from typing import Optional

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


class RolloutMockMixin:
    """
    Mixin class providing rollout dump/mock functionality.

    This mixin should be used with scheduler classes that have:
    - self.config or self.pipeline_config: Configuration object with optional rollout_mock attribute
    - self.mode (str) OR self.is_val (bool): Indicating 'train' or 'val' mode

    Usage:
        # For schedulers with self.mode (like RolloutScheduler):
        class MyScheduler(RolloutMockMixin, BaseScheduler):
            def __init__(self, config, mode, ...):
                self.config = config
                self.mode = mode
                self._init_rollout_mock()
                ...

        # For schedulers with self.is_val (like DynamicSamplingScheduler):
        class MyScheduler(RolloutMockMixin, BaseScheduler):
            def __init__(self, pipeline_config, ...):
                self.pipeline_config = pipeline_config
                self.is_val = is_val
                self._init_rollout_mock()
                ...

            async def get_batch(self, ...):
                # In mock mode, load pre-recorded data
                if self._should_load_mock(global_step):
                    return await self._load_mock_batch(global_step)

                # Normal flow...
                batch = await self._actual_get_batch(...)

                # In dump mode, save the batch
                await self._maybe_dump_batch(batch, global_step)
                return batch
    """

    def _get_config(self):
        """Get configuration object (supports both self.config and self.pipeline_config)."""
        return getattr(self, 'config', None) or getattr(self, 'pipeline_config', None)

    def _get_mode_str(self) -> str:
        """
        Get mode string ('train' or 'val').

        Supports both self.mode (str) and self.is_val (bool) attributes.
        """
        if hasattr(self, 'mode'):
            return self.mode
        elif hasattr(self, 'is_val'):
            return 'val' if self.is_val else 'train'
        else:
            raise AttributeError("Scheduler must have either 'mode' or 'is_val' attribute")

    def _init_rollout_mock(self):
        """
        Initialize rollout mock configuration.

        Should be called in the scheduler's __init__ method after
        config and mode/is_val attributes are set.
        """
        config = self._get_config()
        if config is None:
            logger.warning("[RolloutMock] No config found, mock functionality disabled")
            self.mock_config = None
            return

        self.mock_config = getattr(config, 'rollout_mock', None)
        if self.mock_config and self.mock_config.enable:
            mode_str = self._get_mode_str()
            dump_dir = os.path.join(self.mock_config.dump_dir, mode_str)
            os.makedirs(dump_dir, exist_ok=True)
            logger.info(
                f"[RolloutMock] Rollout Mock enabled: mode={self.mock_config.mode}, "
                f"dir={self.mock_config.dump_dir}, scheduler_mode={mode_str}, format=pickle"
            )

    def _should_load_mock(self, global_step: int) -> bool:
        """
        Check if we should load mock data for this step.

        Args:
            global_step: Current training step

        Returns:
            True if mock mode is enabled and we should load data
        """
        return (
            self.mock_config
            and self.mock_config.enable
            and self.mock_config.mode == "mock"
        )

    def _should_dump_batch(self) -> bool:
        """
        Check if we should dump batches.

        Returns:
            True if dump mode is enabled
        """
        return (
            self.mock_config
            and self.mock_config.enable
            and self.mock_config.mode == "dump"
        )

    async def _maybe_dump_batch(self, batch: DataProto, global_step: int):
        """
        Dump batch if dump mode is enabled.

        Args:
            batch: DataProto to dump
            global_step: Current training step
        """
        if self._should_dump_batch():
            await self._dump_batch(batch, global_step)

    async def _dump_batch(self, batch: DataProto, global_step: int):
        """
        Dump DataProto to disk (pickle format).

        Args:
            batch: DataProto to dump
            global_step: Current training step
        """
        mode_str = self._get_mode_str()
        dump_path = os.path.join(
            self.mock_config.dump_dir,
            mode_str,
            f"step_{global_step:06d}.pkl"
        )
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)

        # Use pickle serialization (DataProto supports __getstate__/__setstate__)
        with open(dump_path, 'wb') as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = os.path.getsize(dump_path) / (1024 * 1024)
        logger.info(
            f"[RolloutMock] Dumped step {global_step}: {dump_path} "
            f"(samples={len(batch)}, size={file_size_mb:.2f}MB)"
        )

    async def _load_mock_batch(self, global_step: int) -> DataProto:
        """
        Load pre-recorded DataProto from disk (strict mode).

        Args:
            global_step: Current training step

        Returns:
            Loaded DataProto

        Raises:
            FileNotFoundError: If mock file doesn't exist
        """
        mode_str = self._get_mode_str()
        mock_path = os.path.join(
            self.mock_config.dump_dir,
            mode_str,
            f"step_{global_step:06d}.pkl"
        )

        # Strict mode: raise error if file doesn't exist
        if not os.path.exists(mock_path):
            raise FileNotFoundError(
                f"[RolloutMock] Mock file not found: {mock_path}\n"
                f"Possible reasons:\n"
                f"  1. Step {global_step} was never run in dump mode\n"
                f"  2. Incorrect dump_dir configuration: {self.mock_config.dump_dir}\n"
                f"  3. Mode mismatch (current mode: {mode_str})\n"
                f"Please run in dump mode first to ensure all step data is generated."
            )

        # Deserialize
        with open(mock_path, 'rb') as f:
            batch = pickle.load(f)

        logger.info(
            f"[RolloutMock] Loaded step {global_step}: {mock_path} "
            f"(samples={len(batch)})"
        )
        return batch
