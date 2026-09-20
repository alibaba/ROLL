"""Qwen4 decoder carries lexical token context through activation recomputation."""
from contextlib import nullcontext
from functools import partial

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import make_viewless_tensor

from .hyperconnection import HyperConnectionMixer, expand_to_streams
from .ple_layer import PLELayer


class Qwen4ExpTransformerBlock(TransformerBlock):
    def __init__(self, config, *args, **kwargs):
        kwargs["post_layer_norm"] = False
        super().__init__(config, *args, **kwargs)
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        if self.post_process:
            self.hyper_connection_mixer = HyperConnectionMixer(
                config.hidden_size, config.hc_count, config.hc_lowrank,
                eps=config.layernorm_epsilon, dtype=config.params_dtype, device=device,
                sequence_parallel=config.sequence_parallel)
        for layer in self.layers:
            if layer.layer_number in (config.ple_layer_ids or []):
                layer.ple = PLELayer(config.hidden_size, config.ple_embed_dim, config.hc_count,
                                     config.ple_conv_kernel_size, config.ngram_size,
                                     config.heads_per_ngram, config.ngram_vocab_size_base,
                                     config.eos_token_id, eps=config.layernorm_epsilon,
                                     dtype=config.params_dtype, device=device)

    def _run_layers(self, start, end, hidden, input_ids, valid, loss_mask, padding_mask,
                    attention_mask, rotary_pos_emb):
        for layer in self.layers[start:end]:
            if hasattr(layer, "ple"):
                if input_ids is None:
                    raise ValueError("PLE requires original input_ids")
                if self.training and self.config.checkpoint_cpu_offload:
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        hidden = tensor_parallel.checkpoint(
                            partial(self._run_ple, layer), False, hidden, input_ids, valid)
                else:
                    hidden = self._run_ple(layer, hidden, input_ids, valid)
            hidden, _ = layer(hidden, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb,
                              padding_mask=padding_mask, qsa_valid_mask=valid, qsa_loss_mask=loss_mask)
        return hidden

    def _run_ple(self, layer, hidden, input_ids, valid):
        ple_hidden = hidden
        if self.config.sequence_parallel:
            # Full-sequence PLE runs redundantly on TP ranks; scatter's
            # backward allgathers gradients, so gather backward splits.
            ple_hidden = tensor_parallel.gather_from_sequence_parallel_region(
                hidden, tensor_parallel_output_grad=False, group=self.pg_collection.tp)
        delta = layer.ple(ple_hidden.transpose(0, 1), input_ids, valid_mask=valid).transpose(0, 1)
        if self.config.sequence_parallel:
            delta = tensor_parallel.scatter_to_sequence_parallel_region(delta, group=self.pg_collection.tp)
        return hidden + delta

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None,
                ple_input_ids=None, qsa_valid_mask=None, qsa_loss_mask=None,
                padding_mask=None, inference_context=None, packed_seq_params=None,
                **kwargs):
        if inference_context is not None or packed_seq_params is not None:
            raise NotImplementedError("Qwen4 decoder supports unpacked training forwards")
        if not self.pre_process:
            hidden_states = self.input_tensor
        else:
            hidden_states = expand_to_streams(hidden_states, self.config.hc_count)
        hidden = make_viewless_tensor(hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        rng = tensor_parallel.get_cuda_rng_tracker().fork() if self.config.sequence_parallel else nullcontext()
        with rng:
            if (self.training and self.config.recompute_granularity == "full"
                    and not self.config.checkpoint_cpu_offload):
                chunk = self.config.recompute_num_layers or 1
                for start in range(0, len(self.layers), chunk):
                    run = partial(self._run_layers, start, min(start+chunk, len(self.layers)))
                    hidden = tensor_parallel.checkpoint(
                        run, False, hidden, ple_input_ids, qsa_valid_mask, qsa_loss_mask,
                        padding_mask, attention_mask, rotary_pos_emb)
            else:
                hidden = self._run_layers(0, len(self.layers), hidden, ple_input_ids,
                                          qsa_valid_mask, qsa_loss_mask, padding_mask,
                                          attention_mask, rotary_pos_emb)
        if self.post_process:
            if self.training and self.config.checkpoint_cpu_offload:
                # The final mixer's FP32 norm/gate graph is wide enough to
                # dominate the remaining activations after layer checkpoints
                # are offloaded. Recompute it from its one CPU-saved stream.
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    hidden = tensor_parallel.checkpoint(self.hyper_connection_mixer, False, hidden)
            else:
                hidden = self.hyper_connection_mixer(hidden)
        return make_viewless_tensor(hidden, requires_grad=hidden.requires_grad, keep_graph=True)
