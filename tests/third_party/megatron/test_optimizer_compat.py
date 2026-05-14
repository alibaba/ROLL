from roll.third_party.megatron.optimizer import _build_param_groups_and_buffers_kwargs


def test_build_param_groups_and_buffers_kwargs_supports_new_signature(monkeypatch):
    def fake_get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset,
        config,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        filter_fn,
        buffer_name,
        default_skip_embedding_weight_decay=False,
    ):
        return model_chunks, {}

    monkeypatch.setattr(
        "roll.third_party.megatron.optimizer._get_param_groups_and_buffers",
        fake_get_param_groups_and_buffers,
    )

    kwargs = _build_param_groups_and_buffers_kwargs(
        model_chunk_offset=3,
        config="cfg",
        no_weight_decay_cond="no_wd",
        scale_lr_cond="scale_lr",
        lr_mult=0.5,
        filter_fn="filter",
        buffer_name="buffers",
    )

    assert kwargs == {
        "model_chunk_offset": 3,
        "config": "cfg",
        "no_weight_decay_cond": "no_wd",
        "scale_lr_cond": "scale_lr",
        "lr_mult": 0.5,
        "filter_fn": "filter",
        "buffer_name": "buffers",
        "default_skip_embedding_weight_decay": False,
    }


def test_build_param_groups_and_buffers_kwargs_supports_old_signature(monkeypatch):
    def fake_get_param_groups_and_buffers(model_chunks, model_chunk_offset, config, filter_fn, buffer_name):
        return model_chunks, {}

    monkeypatch.setattr(
        "roll.third_party.megatron.optimizer._get_param_groups_and_buffers",
        fake_get_param_groups_and_buffers,
    )

    kwargs = _build_param_groups_and_buffers_kwargs(
        model_chunk_offset=1,
        config="cfg",
        no_weight_decay_cond="no_wd",
        scale_lr_cond="scale_lr",
        lr_mult=1.0,
        filter_fn="filter",
        buffer_name="buffers",
    )

    assert kwargs == {
        "model_chunk_offset": 1,
        "config": "cfg",
        "filter_fn": "filter",
        "buffer_name": "buffers",
    }
