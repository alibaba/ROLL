"""
Unit tests for ROLL custom exceptions.

This module tests the custom exception classes defined in roll/utils/exceptions.py.
"""

import pytest

from roll.utils.exceptions import (
    RollError,
    RollConfigError,
    RollConfigValidationError,
    RollConfigMissingError,
    RollConfigConflictError,
    RollDistributedError,
    RollWorkerInitError,
    RollCommunicationError,
    RollTimeoutError,
    RollModelError,
    RollModelLoadError,
    RollModelUpdateError,
    RollOOMError,
    RollDataError,
    RollDataLoadError,
    RollDataFormatError,
    RollPipelineError,
    RollPipelineInitError,
    RollPipelineStepError,
    RollCheckpointError,
    RollEnvironmentError,
    RollEnvInitError,
    RollEnvStepError,
    get_exception_by_code,
    format_error_for_logging,
    ERROR_CODE_MAP,
)


class TestRollConfigValidationError:
    """Tests for RollConfigValidationError."""

    def test_basic_validation_error(self):
        """Test basic validation error creation."""
        error = RollConfigValidationError(
            field_name="test_field",
            expected_type="positive integer",
            actual_value="-1"
        )
        
        assert error.code == 1001
        assert error.error_category == "CONFIG"
        assert "test_field" in error.message
        assert error.field_name == "test_field"
        assert error.expected_type == "positive integer"
        assert error.actual_value == "-1"

    def test_validation_error_with_custom_message(self):
        """Test validation error with custom message."""
        error = RollConfigValidationError(
            field_name="batch_size",
            expected_type="positive integer",
            actual_value="0",
            message="batch_size must be greater than 0"
        )
        
        assert "batch_size must be greater than 0" in error.message
        assert error.context["field_name"] == "batch_size"

    def test_to_dict(self):
        """Test to_dict method."""
        error = RollConfigValidationError(
            field_name="learning_rate",
            expected_type="float",
            actual_value="invalid"
        )
        
        d = error.to_dict()
        
        assert d["error_code"] == 1001
        assert d["error_category"] == "CONFIG"
        assert "learning_rate" in d["message"]
        assert d["context"]["field_name"] == "learning_rate"
        assert d["suggestion"] is not None

    def test_str_representation(self):
        """Test string representation."""
        error = RollConfigValidationError(
            field_name="test",
            expected_type="int",
            actual_value="str"
        )
        
        s = str(error)
        
        assert "[CONFIG-1001]" in s
        assert "test" in s


class TestRollConfigConflictError:
    """Tests for RollConfigConflictError."""

    def test_basic_conflict_error(self):
        """Test basic conflict error creation."""
        error = RollConfigConflictError(
            field1="field_a",
            field2="field_b",
            reason="they are mutually exclusive"
        )
        
        assert error.code == 1003
        assert error.field1 == "field_a"
        assert error.field2 == "field_b"
        assert "mutually exclusive" in error.message

    def test_conflict_error_context(self):
        """Test conflict error context."""
        error = RollConfigConflictError(
            field1="use_gpu",
            field2="use_cpu",
            reason="cannot enable both GPU and CPU mode"
        )
        
        assert error.context["field1"] == "use_gpu"
        assert error.context["field2"] == "use_cpu"
        assert error.context["reason"] == "cannot enable both GPU and CPU mode"


class TestRollDistributedError:
    """Tests for RollDistributedError."""

    def test_basic_distributed_error(self):
        """Test basic distributed error creation."""
        error = RollDistributedError(
            "Worker initialization failed",
            error_code=2001
        )
        
        assert error.code == 2001
        assert error.error_category == "DISTRIBUTED"
        assert "Worker initialization failed" in error.message

    def test_distributed_error_with_context(self):
        """Test distributed error with context."""
        error = RollDistributedError(
            "Communication timeout",
            error_code=2002,
            context={"rank": 0, "timeout": 30}
        )
        
        assert error.context["rank"] == 0
        assert error.context["timeout"] == 30


class TestRollWorkerInitError:
    """Tests for RollWorkerInitError."""

    def test_worker_init_error(self):
        """Test worker init error creation."""
        error = RollWorkerInitError(
            worker_name="actor_train",
            reason="CUDA out of memory",
            rank=0
        )
        
        assert error.code == 2001
        assert error.worker_name == "actor_train"
        assert error.rank == 0
        assert "CUDA out of memory" in error.message


class TestRollModelError:
    """Tests for RollModelError."""

    def test_basic_model_error(self):
        """Test basic model error creation."""
        error = RollModelError(
            "Model loading failed",
            error_code=3001
        )
        
        assert error.code == 3001
        assert error.error_category == "MODEL"

    def test_model_load_error(self):
        """Test model load error creation."""
        error = RollModelLoadError(
            model_path="/path/to/model",
            reason="File not found"
        )
        
        assert error.code == 3001
        assert error.model_path == "/path/to/model"
        assert "File not found" in error.message

    def test_oom_error(self):
        """Test OOM error creation."""
        error = RollOOMError(
            operation="forward pass",
            allocated_gb=24.5,
            total_gb=32.0
        )
        
        assert error.code == 3003
        assert error.operation == "forward pass"
        assert "24.5" in error.message
        assert "32.0" in error.message


class TestRollDataError:
    """Tests for RollDataError."""

    def test_basic_data_error(self):
        """Test basic data error creation."""
        error = RollDataError(
            "Invalid data format",
            error_code=4001
        )
        
        assert error.code == 4001
        assert error.error_category == "DATA"

    def test_data_load_error(self):
        """Test data load error creation."""
        error = RollDataLoadError(
            data_path="/path/to/data.json",
            reason="Invalid JSON format"
        )
        
        assert error.code == 4001
        assert error.data_path == "/path/to/data.json"

    def test_data_format_error(self):
        """Test data format error creation."""
        error = RollDataFormatError(
            expected_format="dict",
            actual_format="list",
            sample_index=42
        )
        
        assert error.code == 4002
        assert error.expected_format == "dict"
        assert error.actual_format == "list"
        assert error.sample_index == 42


class TestRollPipelineError:
    """Tests for RollPipelineError."""

    def test_basic_pipeline_error(self):
        """Test basic pipeline error creation."""
        error = RollPipelineError(
            "Pipeline step failed",
            error_code=5001
        )
        
        assert error.code == 5001
        assert error.error_category == "PIPELINE"

    def test_pipeline_init_error(self):
        """Test pipeline init error creation."""
        error = RollPipelineInitError(
            pipeline_name="RLVRPipeline",
            reason="Missing configuration"
        )
        
        assert error.code == 5001
        assert error.pipeline_name == "RLVRPipeline"

    def test_checkpoint_error(self):
        """Test checkpoint error creation."""
        error = RollCheckpointError(
            operation="save",
            checkpoint_path="/path/to/checkpoint",
            reason="Disk full"
        )
        
        assert error.code == 5003
        assert error.operation == "save"
        assert error.checkpoint_path == "/path/to/checkpoint"


class TestRollEnvironmentError:
    """Tests for RollEnvironmentError."""

    def test_basic_environment_error(self):
        """Test basic environment error creation."""
        error = RollEnvironmentError(
            "Environment step failed",
            error_code=6001
        )
        
        assert error.code == 6001
        assert error.error_category == "ENV"

    def test_env_init_error(self):
        """Test environment init error creation."""
        error = RollEnvInitError(
            env_name="SokobanEnv",
            reason="Missing dependency"
        )
        
        assert error.code == 6001
        assert error.env_name == "SokobanEnv"

    def test_env_step_error(self):
        """Test environment step error creation."""
        error = RollEnvStepError(
            env_name="FrozenLake",
            action=3,
            reason="Invalid action"
        )
        
        assert error.code == 6002
        assert error.env_name == "FrozenLake"
        assert error.action == 3


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_exception_by_code(self):
        """Test get_exception_by_code function."""
        cls = get_exception_by_code(1001)
        assert cls == RollConfigValidationError
        
        cls = get_exception_by_code(1003)
        assert cls == RollConfigConflictError
        
        cls = get_exception_by_code(9999)
        assert cls is None

    def test_format_error_for_logging(self):
        """Test format_error_for_logging function."""
        error = RollConfigValidationError(
            field_name="test",
            expected_type="int",
            actual_value="str"
        )
        
        d = format_error_for_logging(error)
        
        assert d["error_code"] == 1001
        assert d["error_category"] == "CONFIG"
        assert d["exception_type"] == "RollConfigValidationError"
        assert "timestamp" not in d  # Should not have timestamp

    def test_error_code_map_completeness(self):
        """Test that all exceptions are in ERROR_CODE_MAP."""
        expected_codes = [
            1001, 1002, 1003,  # Config errors
            2001, 2002, 2003,  # Distributed errors
            3001, 3002, 3003,  # Model errors
            4001, 4002,        # Data errors
            5001, 5002, 5003,  # Pipeline errors
            6001, 6002,        # Environment errors
        ]
        
        for code in expected_codes:
            assert code in ERROR_CODE_MAP, f"Error code {code} not in ERROR_CODE_MAP"


class TestExceptionInheritance:
    """Tests for exception inheritance."""

    def test_config_errors_inherit_from_roll_error(self):
        """Test that config errors inherit from RollError."""
        assert issubclass(RollConfigError, RollError)
        assert issubclass(RollConfigValidationError, RollConfigError)
        assert issubclass(RollConfigMissingError, RollConfigError)
        assert issubclass(RollConfigConflictError, RollConfigError)

    def test_distributed_errors_inherit_from_roll_error(self):
        """Test that distributed errors inherit from RollError."""
        assert issubclass(RollDistributedError, RollError)
        assert issubclass(RollWorkerInitError, RollDistributedError)
        assert issubclass(RollCommunicationError, RollDistributedError)
        assert issubclass(RollTimeoutError, RollDistributedError)

    def test_model_errors_inherit_from_roll_error(self):
        """Test that model errors inherit from RollError."""
        assert issubclass(RollModelError, RollError)
        assert issubclass(RollModelLoadError, RollModelError)
        assert issubclass(RollModelUpdateError, RollModelError)
        assert issubclass(RollOOMError, RollModelError)

    def test_data_errors_inherit_from_roll_error(self):
        """Test that data errors inherit from RollError."""
        assert issubclass(RollDataError, RollError)
        assert issubclass(RollDataLoadError, RollDataError)
        assert issubclass(RollDataFormatError, RollDataError)

    def test_pipeline_errors_inherit_from_roll_error(self):
        """Test that pipeline errors inherit from RollError."""
        assert issubclass(RollPipelineError, RollError)
        assert issubclass(RollPipelineInitError, RollPipelineError)
        assert issubclass(RollPipelineStepError, RollPipelineError)
        assert issubclass(RollCheckpointError, RollPipelineError)

    def test_environment_errors_inherit_from_roll_error(self):
        """Test that environment errors inherit from RollError."""
        assert issubclass(RollEnvironmentError, RollError)
        assert issubclass(RollEnvInitError, RollEnvironmentError)
        assert issubclass(RollEnvStepError, RollEnvironmentError)


class TestExceptionRaising:
    """Tests for exception raising and catching."""

    def test_catch_base_exception(self):
        """Test catching derived exception with base class."""
        with pytest.raises(RollError):
            raise RollConfigValidationError(
                field_name="test",
                expected_type="int",
                actual_value="str"
            )

    def test_catch_config_exception(self):
        """Test catching derived exception with config base class."""
        with pytest.raises(RollConfigError):
            raise RollConfigConflictError(
                field1="a",
                field2="b",
                reason="conflict"
            )

    def test_exception_message_contains_context(self):
        """Test that exception message contains context."""
        error = RollConfigValidationError(
            field_name="batch_size",
            expected_type="positive integer",
            actual_value="-5"
        )
        
        msg = str(error)
        
        assert "batch_size" in msg
        assert "positive integer" in msg
        assert "-5" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
