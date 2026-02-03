import ray

from roll.configs.base_config import PPOConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import reduce_metrics_list


class ModelUpdateGroup:
    def __init__(self, src_cluster: Cluster, tgt_cluster: Cluster, pipeline_config: PPOConfig, frequency=1):
        self.src_cluster = src_cluster
        self.tgt_cluster = tgt_cluster
        self.frequency = frequency
        self.pipeline_config = pipeline_config
        self.model_update_name = f"model_update/{self.src_cluster.cluster_name}_2_{self.tgt_cluster.cluster_name}"
        train_devices = set(src_cluster.worker_config.device_mapping or [])
        infer_devices = set(tgt_cluster.worker_config.device_mapping or [])

        assert (max(train_devices) - min(train_devices)) == (len(train_devices) - 1), f"{train_devices=} must be continuous"
        assert (max(infer_devices) - min(infer_devices)) == (len(infer_devices) - 1), f"{infer_devices=} must be continuous"

        ray.get(
            [
                train_worker.setup_model_update.remote(
                    infer_cluster=self.tgt_cluster, model_update_name=self.model_update_name
                )
                for train_worker in self.src_cluster.workers
            ]
        )

    def model_update(self, step=None):
        if step % self.frequency != 0:
            return {}

        dataprotos: list[DataProto] = ray.get(
            [
                train_worker.start_model_update.remote(model_update_name=self.model_update_name)
                for train_worker in self.src_cluster.workers
            ]
        )
        return reduce_metrics_list([dataproto.meta_info["metrics"] for dataproto in dataprotos])
