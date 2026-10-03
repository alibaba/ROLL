"""Audit every real checkpoint name against the adapter conversion registry."""
import copy
import json
from pathlib import Path

from transformers import AutoConfig
from mcore_adapter.models.converter.template import get_template
from mcore_adapter.models.qwen4_exp import Qwen4ExpConfig


def main():
    root = Path("/data_hdd/Qwen3.8-Flash-Next")
    hf = AutoConfig.from_pretrained(root)
    template = copy.deepcopy(get_template("qwen4_exp"))
    config = template.convert_hf_to_mca_config(hf)
    template.set_mca_config_for_ops(config)
    names = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    coverage, failures = {}, []
    for name in names:
        try:
            mapped = template.hf_name_to_mca_names(name)
            if mapped:
                kind = "derived_constant" if name.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")) else "trainable"
            elif ".ple_embedding.ngram_embedding.shard_" in name:
                kind = "frozen_external"
            elif name.startswith(("mtp.", "model.visual.")):
                kind = "preserved_auxiliary"
            else:
                raise ValueError("unclassified empty conversion")
            coverage[name] = {"kind": kind, "mca_names": mapped or []}
        except Exception as error:
            failures.append({"name": name, "error": str(error)[:250]})
    summary = {"total": len(names), "counts": {kind: sum(v["kind"] == kind for v in coverage.values())
               for kind in ("trainable", "derived_constant", "frozen_external", "preserved_auxiliary")},
               "failures": failures, "config": {name: getattr(config, name) for name in (
                   "hidden_size", "num_layers", "hc_count", "kv_channels", "rotary_percent", "rotary_base",
                   "moe_router_topk", "moe_ffn_hidden_size", "moe_shared_expert_intermediate_size")}}
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
