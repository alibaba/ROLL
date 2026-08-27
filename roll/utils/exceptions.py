"""
ROLL Custom Exceptions

This module defines custom exception classes for better error handling and debugging.
Each exception includes an error code for quick identification.

Error Code Ranges:
    1000-1999: Configuration Errors
    2000-2999: Distributed/System Errors
    3000-3999: Model Errors
    4000-4999: Data Errors
    5000-5999: Pipeline Errors
    6000-6999: Environment Errors
"""

from typing import Optional, Dict, Any


class RollError(Exception):
    """Base exception class for ROLL framework."""
    
    error_code: int = 0
    error_category: str = "GENERAL"
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        self.message = message
        self._error_code = error_code or self.error_code
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [f"[{self.error_category}-{self._error_code}] {self.message}"]
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
    
    @property
    def code(self) -> int:
        return self._error_code
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self._error_code,
            "error_category": self.error_category,
            "message": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
        }


class RollConfigError(RollError):
    """Configuration related errors."""
    
    error_category = "CONFIG"
    error_code = 1000


class RollConfigValidationError(RollConfigError):
    """Configuration validation failed."""
    
    error_code = 1001
    
    def __init__(
        self, 
        field_name: str, 
        expected_type: str, 
        actual_value: Any,
        message: Optional[str] = None,
        **kwargs
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value
        msg = message or f"Invalid configuration for field '{field_name}'"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={
                "field_name": field_name,
                "expected_type": expected_type,
                "actual_type": type(actual_value).__name__,
                "actual_value": str(actual_value)[:100],
            },
            suggestion=f"Please ensure '{field_name}' is of type {expected_type}",
            **kwargs
        )


class RollConfigMissingError(RollConfigError):
    """Required configuration field is missing."""
    
    error_code = 1002
    
    def __init__(
        self, 
        field_name: str, 
        config_path: Optional[str] = None,
        **kwargs
    ):
        self.field_name = field_name
        self.config_path = config_path
        msg = f"Required configuration field '{field_name}' is missing"
        if config_path:
            msg += f" in {config_path}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"field_name": field_name, "config_path": config_path},
            suggestion=f"Please provide a value for '{field_name}' in your configuration",
            **kwargs
        )


class RollConfigConflictError(RollConfigError):
    """Configuration fields have conflicting values."""
    
    error_code = 1003
    
    def __init__(
        self, 
        field1: str, 
        field2: str, 
        reason: str,
        **kwargs
    ):
        self.field1 = field1
        self.field2 = field2
        msg = f"Configuration conflict between '{field1}' and '{field2}': {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"field1": field1, "field2": field2, "reason": reason},
            suggestion="Please ensure these fields are compatible",
            **kwargs
        )


class RollDistributedError(RollError):
    """Distributed system related errors."""
    
    error_category = "DISTRIBUTED"
    error_code = 2000


class RollWorkerInitError(RollDistributedError):
    """Worker initialization failed."""
    
    error_code = 2001
    
    def __init__(
        self, 
        worker_name: str, 
        reason: str,
        rank: Optional[int] = None,
        **kwargs
    ):
        self.worker_name = worker_name
        self.rank = rank
        msg = f"Failed to initialize worker '{worker_name}': {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"worker_name": worker_name, "rank": rank, "reason": reason},
            suggestion="Check worker configuration and GPU availability",
            **kwargs
        )


class RollCommunicationError(RollDistributedError):
    """Inter-process communication error."""
    
    error_code = 2002
    
    def __init__(
        self, 
        src_rank: int, 
        dst_rank: int, 
        operation: str,
        reason: Optional[str] = None,
        **kwargs
    ):
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.operation = operation
        msg = f"Communication failed from rank {src_rank} to {dst_rank} during {operation}"
        if reason:
            msg += f": {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"src_rank": src_rank, "dst_rank": dst_rank, "operation": operation},
            suggestion="Check network connectivity and NCCL configuration",
            **kwargs
        )


class RollTimeoutError(RollDistributedError):
    """Operation timeout error."""
    
    error_code = 2003
    
    def __init__(
        self, 
        operation: str, 
        timeout_seconds: float,
        **kwargs
    ):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        msg = f"Operation '{operation}' timed out after {timeout_seconds}s"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"operation": operation, "timeout_seconds": timeout_seconds},
            suggestion="Consider increasing timeout or optimizing the operation",
            **kwargs
        )


class RollModelError(RollError):
    """Model related errors."""
    
    error_category = "MODEL"
    error_code = 3000


class RollModelLoadError(RollModelError):
    """Model loading failed."""
    
    error_code = 3001
    
    def __init__(
        self, 
        model_path: str, 
        reason: str,
        **kwargs
    ):
        self.model_path = model_path
        msg = f"Failed to load model from '{model_path}': {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"model_path": model_path, "reason": reason},
            suggestion="Check model path and format compatibility",
            **kwargs
        )


class RollModelUpdateError(RollModelError):
    """Model weight update failed."""
    
    error_code = 3002
    
    def __init__(
        self, 
        src_worker: str, 
        dst_worker: str, 
        reason: str,
        **kwargs
    ):
        self.src_worker = src_worker
        self.dst_worker = dst_worker
        msg = f"Failed to update model weights from {src_worker} to {dst_worker}: {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"src_worker": src_worker, "dst_worker": dst_worker, "reason": reason},
            suggestion="Check model architecture compatibility and GPU memory",
            **kwargs
        )


class RollOOMError(RollModelError):
    """Out of memory error with context."""
    
    error_code = 3003
    
    def __init__(
        self, 
        operation: str, 
        allocated_gb: Optional[float] = None,
        total_gb: Optional[float] = None,
        **kwargs
    ):
        self.operation = operation
        self.allocated_gb = allocated_gb
        self.total_gb = total_gb
        msg = f"Out of memory during '{operation}'"
        if allocated_gb and total_gb:
            msg += f" (allocated: {allocated_gb:.2f}GB / total: {total_gb:.2f}GB)"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"operation": operation, "allocated_gb": allocated_gb, "total_gb": total_gb},
            suggestion="Try reducing batch size, enabling gradient checkpointing, or using offload",
            **kwargs
        )


class RollDataError(RollError):
    """Data related errors."""
    
    error_category = "DATA"
    error_code = 4000


class RollDataLoadError(RollDataError):
    """Data loading failed."""
    
    error_code = 4001
    
    def __init__(
        self, 
        data_path: str, 
        reason: str,
        **kwargs
    ):
        self.data_path = data_path
        msg = f"Failed to load data from '{data_path}': {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"data_path": data_path, "reason": reason},
            suggestion="Check data path and format",
            **kwargs
        )


class RollDataFormatError(RollDataError):
    """Data format is invalid."""
    
    error_code = 4002
    
    def __init__(
        self, 
        expected_format: str, 
        actual_format: Optional[str] = None,
        sample_index: Optional[int] = None,
        **kwargs
    ):
        self.expected_format = expected_format
        self.actual_format = actual_format
        self.sample_index = sample_index
        msg = f"Invalid data format, expected {expected_format}"
        if actual_format:
            msg += f", got {actual_format}"
        if sample_index is not None:
            msg += f" at sample index {sample_index}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"expected_format": expected_format, "actual_format": actual_format, "sample_index": sample_index},
            suggestion=f"Ensure data follows {expected_format} format",
            **kwargs
        )


class RollPipelineError(RollError):
    """Pipeline related errors."""
    
    error_category = "PIPELINE"
    error_code = 5000


class RollPipelineInitError(RollPipelineError):
    """Pipeline initialization failed."""
    
    error_code = 5001
    
    def __init__(
        self, 
        pipeline_name: str, 
        reason: str,
        **kwargs
    ):
        self.pipeline_name = pipeline_name
        msg = f"Failed to initialize pipeline '{pipeline_name}': {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"pipeline_name": pipeline_name, "reason": reason},
            suggestion="Check pipeline configuration and dependencies",
            **kwargs
        )


class RollPipelineStepError(RollPipelineError):
    """Pipeline step execution failed."""
    
    error_code = 5002
    
    def __init__(
        self, 
        step_name: str, 
        step_index: int, 
        reason: str,
        **kwargs
    ):
        self.step_name = step_name
        self.step_index = step_index
        msg = f"Pipeline step '{step_name}' (index {step_index}) failed: {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"step_name": step_name, "step_index": step_index, "reason": reason},
            suggestion="Check step configuration and input data",
            **kwargs
        )


class RollCheckpointError(RollPipelineError):
    """Checkpoint save/load failed."""
    
    error_code = 5003
    
    def __init__(
        self, 
        operation: str, 
        checkpoint_path: str, 
        reason: str,
        **kwargs
    ):
        self.operation = operation
        self.checkpoint_path = checkpoint_path
        msg = f"Checkpoint {operation} failed for '{checkpoint_path}': {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"operation": operation, "checkpoint_path": checkpoint_path, "reason": reason},
            suggestion="Check disk space and write permissions",
            **kwargs
        )


class RollEnvironmentError(RollError):
    """Environment related errors."""
    
    error_category = "ENV"
    error_code = 6000


class RollEnvInitError(RollEnvironmentError):
    """Environment initialization failed."""
    
    error_code = 6001
    
    def __init__(
        self, 
        env_name: str, 
        reason: str,
        **kwargs
    ):
        self.env_name = env_name
        msg = f"Failed to initialize environment '{env_name}': {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"env_name": env_name, "reason": reason},
            suggestion="Check environment configuration and dependencies",
            **kwargs
        )


class RollEnvStepError(RollEnvironmentError):
    """Environment step failed."""
    
    error_code = 6002
    
    def __init__(
        self, 
        env_name: str, 
        action: Any, 
        reason: str,
        **kwargs
    ):
        self.env_name = env_name
        self.action = action
        msg = f"Environment '{env_name}' step failed with action {action}: {reason}"
        super().__init__(
            message=msg,
            error_code=self.error_code,
            context={"env_name": env_name, "action": str(action)[:50], "reason": reason},
            suggestion="Check action validity and environment state",
            **kwargs
        )


ERROR_CODE_MAP = {
    cls.error_code: cls
    for cls in [
        RollConfigValidationError,
        RollConfigMissingError,
        RollConfigConflictError,
        RollWorkerInitError,
        RollCommunicationError,
        RollTimeoutError,
        RollModelLoadError,
        RollModelUpdateError,
        RollOOMError,
        RollDataLoadError,
        RollDataFormatError,
        RollPipelineInitError,
        RollPipelineStepError,
        RollCheckpointError,
        RollEnvInitError,
        RollEnvStepError,
    ]
}


def get_exception_by_code(error_code: int) -> Optional[type]:
    """Get exception class by error code."""
    return ERROR_CODE_MAP.get(error_code)


def format_error_for_logging(error: RollError) -> Dict[str, Any]:
    """Format error for structured logging."""
    return {
        "error_code": error.code,
        "error_category": error.error_category,
        "message": error.message,
        "context": error.context,
        "suggestion": error.suggestion,
        "exception_type": type(error).__name__,
    }
