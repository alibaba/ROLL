"""Diffusion actor worker base class.

Subclasses must implement ``forward_and_backward``.
"""

from roll.pipeline.base_worker import ActorWorker as BaseActorWorker


class ActorDiffusionWorker(BaseActorWorker):
    """Diffusion actor worker base class.

    Subclasses (ActorGRPOWorker, ActorNFTWorker) implement forward_and_backward.
    """

    def _run_strategy_train_step(self, backward_batch):
        """Run strategy.train_step with this worker's forward_and_backward."""
        return self.strategy.train_step(
            batch=backward_batch,
            forward_and_backward=self.forward_and_backward,
        )
