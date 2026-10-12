"""Real sibling-process IPC must work with expandable training allocations."""

import multiprocessing as mp
import traceback

import pytest
import torch


def _values(device):
    return [
        ("adapter_a", torch.arange(2**20, dtype=torch.float32, device=device).reshape(1024, 1024).t()),
        ("adapter_b", torch.tensor([[1.5, -2.0], [0.0, 4.0]], dtype=torch.bfloat16, device=device)),
    ]


def _sender(connection, expandable, source_device, allocator_api):
    try:
        from roll.utils.send_recv_utils import serialize_named_weights

        torch.cuda.memory._set_allocator_settings(
            f"expandable_segments:{expandable},max_split_size_mb:128"
        )
        before = torch.cuda.memory._snapshot()["allocator_settings"]
        if allocator_api == "snapshot":
            # Exercise the older PyTorch interface using the real allocator.
            if hasattr(torch._C, "_accelerator_getAllocatorSettings"):
                delattr(torch._C, "_accelerator_getAllocatorSettings")
        weights = _values(source_device)
        for version in range(2):
            for _, tensor in weights:
                tensor.add_(version)
            payload = serialize_named_weights(weights, infer_strategy="vllm")
            assert torch.cuda.memory._snapshot()["allocator_settings"] == before
            connection.send(("payload", payload))
            assert connection.recv() == "received"
        connection.send(("complete", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _receiver(connection):
    try:
        from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
        from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket

        monkey_patch_torch_reductions()
        for version in range(2):
            payload = connection.recv()
            received = MultiprocessingSerializer.deserialize(payload)
            actual = named_tensors_from_bucket(**received)
            expected = _values("cpu")
            assert [name for name, _ in actual] == ["adapter_a", "adapter_b"]
            for (name, tensor), (_, wanted) in zip(actual, expected):
                assert tensor.is_cuda
                assert tensor.dtype == wanted.dtype
                torch.testing.assert_close(tensor.cpu(), wanted + version, rtol=0, atol=0)
            del actual, received, tensor
            torch.cuda.synchronize()
            connection.send(("received", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA IPC")
@pytest.mark.parametrize("expandable,source_device,allocator_api", [
    (True, "cuda", "native"), (False, "cuda", "native"),
    (True, "cpu", "native"), (True, "cuda", "snapshot"),
])
def test_weight_bucket_round_trips_between_sibling_processes(expandable, source_device, allocator_api):
    # A plain torch.cat allocation regresses this boundary under default Docker
    # permissions when the training allocator has expandable_segments enabled.
    context = mp.get_context("spawn")
    sender_pipe, sender_child = context.Pipe()
    receiver_pipe, receiver_child = context.Pipe()
    sender = context.Process(target=_sender, args=(sender_child, expandable, source_device, allocator_api))
    receiver = context.Process(target=_receiver, args=(receiver_child,))
    sender.start()
    receiver.start()
    sender_child.close()
    receiver_child.close()
    try:
        for _ in range(2):
            assert sender_pipe.poll(60), "sender did not produce a bucket"
            kind, payload = sender_pipe.recv()
            assert kind == "payload", payload
            receiver_pipe.send(payload)
            assert receiver_pipe.poll(60), "receiver did not acknowledge the bucket"
            kind, detail = receiver_pipe.recv()
            assert kind == "received", detail
            sender_pipe.send("received")
        assert sender_pipe.poll(30)
        kind, detail = sender_pipe.recv()
        assert kind == "complete", detail
        sender.join(20)
        receiver.join(20)
        assert sender.exitcode == receiver.exitcode == 0
    finally:
        for process in (sender, receiver):
            if process.is_alive():
                process.terminate()
            process.join(5)
        sender_pipe.close()
        receiver_pipe.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA allocator")
def test_failed_bucket_allocation_restores_training_allocator():
    from roll.utils.send_recv_utils import serialize_named_weights

    original = torch.cuda.memory._snapshot()["allocator_settings"]["PYTORCH_CUDA_ALLOC_CONF"]
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True,max_split_size_mb:128")
        before = torch.cuda.memory._snapshot()["allocator_settings"]
        with pytest.raises(RuntimeError):
            serialize_named_weights([("a", torch.ones(4, device="cuda")), ("b", torch.ones(4))], "vllm")
        assert torch.cuda.memory._snapshot()["allocator_settings"] == before
    finally:
        torch.cuda.memory._set_allocator_settings(original)
