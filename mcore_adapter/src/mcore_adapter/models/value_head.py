from collections.abc import Callable

import torch
import torch.nn as nn
from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule

from .converter.dist_converter import DistConverter
from .converter.template import DropConverOp, RenameConverOp, Template


ModelList = list[MegatronModule]
ModelHook = Callable[[ModelList], ModelList | None]


class LinearForLastLayer(nn.Linear):
    """Final replicated projection head compatible with Megatron output-layer calls.

    Megatron-Core output layers receive a few runtime-only arguments. This head
    accepts those arguments for call-site compatibility while using a standard
    replicated linear projection.
    """

    def __init__(self, input_size: int, output_size: int, sequence_parallel: bool) -> None:
        """Initialize a replicated final projection.

        Args:
            input_size: Hidden dimension of the transformer output.
            output_size: Output dimension of the value/reward head.
            sequence_parallel: Whether to gather sequence-parallel activations.
        """
        super().__init__(in_features=input_size, out_features=output_size, bias=False)
        self.sequence_parallel = sequence_parallel
        if sequence_parallel:
            setattr(self.weight, "sequence_parallel", True)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Run the final projection and return Megatron-style ``(output, bias)``."""
        del weight, runtime_gather_output
        if input_.dtype != self.weight.dtype:
            input_ = input_.to(self.weight.dtype)
        logits = super().forward(input_)
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits,
                tensor_parallel_output_grad=False,
            )
        return logits, None


def create_value_head_hook(hidden_size: int, sequence_parallel: bool, output_size: int = 1) -> ModelHook:
    """Create a pre-wrap hook that replaces the final pipeline stage output head.

    Args:
        hidden_size: Hidden dimension of the transformer output.
        sequence_parallel: Whether the model uses sequence parallelism.
        output_size: Number of outputs produced by the final head.

    Returns:
        A model hook suitable for external trainer provider construction.
    """
    from megatron.core import parallel_state

    # _register_linear_for_last_layer_mapping()

    def hook(model: ModelList | MegatronModule) -> ModelList:
        model_chunks = model
        model_post_process: list[bool] = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for vp_stage in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(
                    parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
                )
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        if len(model_post_process) != len(model_chunks):
            raise ValueError(
                "Model list length and pipeline post-process list length must match. "
                f"Got {len(model_chunks)} model chunks and {len(model_post_process)} post-process flags."
            )

        for index, model_chunk in enumerate(model_chunks):
            if model_post_process[index]:
                model_chunk.output_layer = LinearForLastLayer(
                    input_size=hidden_size,
                    output_size=output_size,
                    sequence_parallel=sequence_parallel,
                )

        return model_chunks

    return hook


def make_value_model(hidden_size: int, sequence_parallel: bool) -> ModelHook:
    """Create a value-head hook compatible with existing external trainer code."""
    return create_value_head_hook(hidden_size=hidden_size, sequence_parallel=sequence_parallel)


def make_value_head_template(template: Template) -> Template:
    value_head_weight_op = RenameConverOp(hf_names=["v_head.summary.weight"], mca_names=["output_layer.weight"])
    value_head_bias_op = RenameConverOp(hf_names=["v_head.summary.bias"], mca_names=["output_layer.bias"])
    drop_op = DropConverOp(hf_names=["lm_head.weight"], mca_names=[])
    for i, op in enumerate(template.weight_converters):
        if op.mca_names == ["output_layer.weight"]:
            template.weight_converters[i] = value_head_weight_op
    template.weight_converters.append(value_head_bias_op)
    template.mca_name_to_converter["output_layer.weight"] = value_head_weight_op
    template.mca_name_to_converter["output_layer.bias"] = value_head_bias_op
    template.hf_name_to_converter["v_head.summary.weight"] = value_head_weight_op
    template.hf_name_to_converter["v_head.summary.bias"] = value_head_bias_op
    template.hf_name_to_converter["lm_head.weight"] = drop_op
    return template


def make_value_head_converter(dist_converter: DistConverter) -> DistConverter:
    dist_converter.config.duplicated_weights.append("output_layer.weight")
    return dist_converter
