import torch
from tensordict import TensorDict

from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.step_env_manager import StepEnvManager
from roll.utils.hash_utils import compute_object_hash
from roll.utils.str_utils import contains_renderable_field


class StepConcatEnvManager(StepEnvManager):
    """
    used for gem concat: https://github.com/axon-rl/gem/blob/6bb654052358463c141093d9dc13c69ecba7ed82/gem/wrappers/wrapper_factory.py#L64C6-L64C12
    """

    def format_messages(self, rollout_cache: RolloutCache) -> DataProto:
        current_cache = rollout_cache.history[-1]
        memory_history = []
        if "history_length" in self.cfg_template:
            memory_history = rollout_cache.history[-self.cfg_template["history_length"]:-1]

        sar_history = []
        for history_step, entry in enumerate(memory_history):
            observation = entry["observation"]
            sar_history.append(observation)

        current_observation = f"{current_cache['observation']}\n{current_cache.get('suffix', '')}"
        render_dict = {"history": "\n".join(sar_history)}
        if contains_renderable_field(self.agent_template, "current_observation"):
            render_dict["current_observation"] = current_observation
        messages = []
        if self.agent_system_template is not None:
            messages.append({"role": "system", "content": self.agent_system_template})
        messages.append({"role": "user", "content": self.agent_template.format(**render_dict)})
        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        position_ids = attention_mask.cumsum(dim=-1)
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        current_cache["prompt_ids"] = prompt_ids
        current_cache['state_hash'] = compute_object_hash(current_observation)
        current_cache['messages'] = messages
        return lm_input
