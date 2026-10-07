from __future__ import annotations

from pathlib import Path

from roll.pipeline.tinker_backend_runtime import types


def prepare_sample_batch(
    requests: dict[str, tuple[str, types.SampleInput]],
    checkpoints_base: Path | None = None,
) -> types.PreparedSampleBatch:
    """Prepare runtime pull-mode sample actions for RollBasePipeline.

    This is intentionally a pure adapter helper. It mirrors the request/result
    backend request/result shape without importing the control-plane API,
    DB models, SQLAlchemy, or engine loop.
    """

    all_model_inputs = []
    all_sampling_params = []
    all_model_ids = []
    all_checkpoint_ids = []
    all_checkpoint_paths = []
    all_env_ids = []
    request_batch_slices = []

    needs_prompt_logprobs = any(request_data.prompt_logprobs for _, request_data in requests.values())

    for request_id, (model_id, request_data) in requests.items():
        request_start = len(all_model_inputs)
        checkpoint_path = ""
        if model_id and request_data.checkpoint_id and checkpoints_base:
            checkpoint_path = str(
                checkpoints_base / model_id / "sampler_weights" / f"{request_data.checkpoint_id}.tar.gz"
            )

        base_seed = request_data.sampling_params.seed
        for sample_idx in range(request_data.num_samples):
            all_model_inputs.append(request_data.prompt)
            sample_params = request_data.sampling_params.model_copy(
                update={"seed": base_seed + sample_idx if base_seed is not None else None}
            )
            all_sampling_params.append(sample_params)
            all_model_ids.append(model_id)
            all_checkpoint_ids.append(request_data.checkpoint_id)
            all_checkpoint_paths.append(checkpoint_path)
            all_env_ids.append(request_data.env_id)

        request_batch_slices.append(
            (request_id, model_id, request_start, len(all_model_inputs), request_data.prompt_logprobs)
        )

    return types.PreparedSampleBatch(
        all_model_inputs=all_model_inputs,
        all_sampling_params=all_sampling_params,
        all_model_ids=all_model_ids,
        all_checkpoint_ids=all_checkpoint_ids,
        all_checkpoint_paths=all_checkpoint_paths,
        all_env_ids=all_env_ids,
        needs_prompt_logprobs=needs_prompt_logprobs,
        request_batch_slices=request_batch_slices,
    )


def prepare_model_pass_batch(
    requests: dict[str, tuple[str, types.ForwardBackwardInput]],
) -> types.PreparedModelPassBatch:
    """Prepare runtime pull-mode forward/forward_backward actions."""

    all_model_inputs = []
    all_targets = []
    all_token_weights = []
    all_model_ids = []
    all_sampling_logprobs = []
    all_advantages = []
    all_loss_fns = []
    all_loss_fn_configs = []
    request_batch_slices = []

    for request_id, (model_id, request_data) in requests.items():
        if request_data.loss_fn not in types.LOSS_TYPES:
            raise ValueError(
                f"Unknown loss function {request_data.loss_fn!r}. "
                f"Must be one of: {list(types.LOSS_TYPES.keys())}"
            )
        request_start = len(all_model_inputs)
        for item in request_data.data:
            all_model_inputs.append(item.model_input)
            loss_fn_inputs = item.loss_fn_inputs
            all_targets.append(loss_fn_inputs.target_tokens.data)
            all_token_weights.append(loss_fn_inputs.weights.data)
            all_sampling_logprobs.append(loss_fn_inputs.logprobs.data)
            all_advantages.append(loss_fn_inputs.advantages.data)
            all_model_ids.append(model_id)
            all_loss_fns.append(request_data.loss_fn)
            all_loss_fn_configs.append(request_data.loss_fn_config)

        request_batch_slices.append((request_id, model_id, request_start, len(all_model_inputs)))

    return types.PreparedModelPassBatch(
        all_model_inputs=all_model_inputs,
        all_targets=all_targets,
        all_token_weights=all_token_weights,
        all_sampling_logprobs=all_sampling_logprobs,
        all_advantages=all_advantages,
        all_model_ids=all_model_ids,
        all_loss_fns=all_loss_fns,
        all_loss_fn_configs=all_loss_fn_configs,
        request_batch_slices=request_batch_slices,
    )


def init_roll_backend(backend_config: dict):
    """Initialize RollBasePipeline without importing the old Tinker backend."""

    from dacite import from_dict
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from roll.pipeline.agentic.agentic_config import AgenticConfig
    from roll.pipeline.tinker_backend_runtime.roll_backend import ROLLRuntimeBackend

    config_path = backend_config.get("config_path", "config")
    config_name = backend_config.get("config_name")
    if not config_name:
        raise ValueError("tinker_runtime.backend_config must include config_name")

    with initialize_config_dir(config_dir=str(Path(config_path).resolve()), version_base=None):
        cfg = compose(config_name=config_name)

    pipeline_config = from_dict(
        data_class=AgenticConfig,
        data=OmegaConf.to_container(cfg, resolve=True),
    )
    if backend_config.get("ray_address") != "local":
        from roll.distributed.scheduler.initialize import init

        init()
        return ROLLRuntimeBackend(pipeline_config)

    import ray

    from roll.platforms import current_platform
    from roll.utils.constants import RAY_NAMESPACE

    # address="local" always creates our own Ray instance, even if another
    # cluster is running on this host. Never disconnect an existing caller.
    if ray.is_initialized():
        raise RuntimeError("Local Tinker runtime requires a process without an existing Ray connection")
    ray_options = {
        "address": "local",
        "num_gpus": pipeline_config.num_gpus_per_node * pipeline_config.num_nodes,
        "num_cpus": 16,
        "include_dashboard": False,
        "namespace": RAY_NAMESPACE,
        "runtime_env": {"env_vars": current_platform.get_custom_env_vars()},
    }
    if backend_config.get("ray_temp_dir"):
        ray_options["_temp_dir"] = str(Path(backend_config["ray_temp_dir"]).expanduser().resolve())
    ray.init(**ray_options)
    try:
        # ROLLRuntimeBackend.close() calls ray.shutdown(), which releases
        # the local instance created above without touching external clusters.
        return ROLLRuntimeBackend(pipeline_config)
    except BaseException:
        ray.shutdown()
        raise
