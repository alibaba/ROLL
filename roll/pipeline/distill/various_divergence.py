import torch
import torch.nn as nn

from roll.pipeline.distill.distill_config import DistillConfig

from roll.utils.constants import IGNORE_INDEX
class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

class VariousDivergence:
    def __init__(self, pipeline_config: DistillConfig, padding_id=IGNORE_INDEX) -> None:
        self.kd_temperature = pipeline_config.kd_temperature
        self.teacher_temperature = pipeline_config.teacher_temperature
        self.kd_objective = pipeline_config.kd_objective
        self.padding_id = padding_id
        self.args = pipeline_config

        if self.kd_objective == "forward_kl":
            self.dist_func = self.compute_forward_kl_divergence
        elif self.kd_objective == "reverse_kl":
            self.dist_func = self.compute_reverse_kl_divergence
        elif self.kd_objective == "adaptive_kl":
            self.dist_func = self.compute_adaptive_kl_divergence
        elif self.kd_objective == "skewed_forward_kl":
            self.dist_func = self.compute_skewed_forward_kl_divergence
        elif self.kd_objective == "skewed_reverse_kl":
            self.dist_func = self.compute_skewed_reverse_kl_divergence
        elif self.kd_objective == "js_divergence":
            self.dist_func = self.compute_js_divergence
        else:
            raise NameError(f"Unsupported kd_objective for `{self.kd_objective}'")

    def __call__(self, logits, teacher_logits, labels,attention_mask=None):
        kd_loss = self.dist_func(logits, teacher_logits, labels,attention_mask=attention_mask)
        return kd_loss

    def compute_forward_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        attention_mask=None,
        reduction="mean"
    ):

        logits = logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.teacher_temperature

        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - lprobs))
        inf_mask = logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()
        elif reduction == "mean":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            num_valid_elements = (~pad_mask).sum().float()
            kld = kld.sum() / num_valid_elements

        return kld

    def compute_reverse_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        attention_mask=None,
        reduction="mean",
    ):
        logits = logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.teacher_temperature

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (probs * (lprobs - teacher_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()
        elif reduction == "mean":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            num_valid_elements = (~pad_mask).sum().float()
            kld = kld.sum() / num_valid_elements

        return kld

    def compute_adaptive_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        attention_mask=None,
        reduction="mean"
    ):
        alpha = self.args.adaptive_kl_alpha
        probs = torch.softmax(
            logits / self.kd_temperature, dim=-1, dtype=torch.float32
        )
        teacher_probs = torch.softmax(
                teacher_logits / self.teacher_temperature / self.kd_temperature, dim=-1, dtype=torch.float32
            )
        sorted_teacher_probs, sorted_idx = teacher_probs.sort(-1)
        sorted_probs = probs.gather(-1, sorted_idx)
        gap = (sorted_teacher_probs - sorted_probs).abs()
        cum_teacher_probs = torch.cumsum(sorted_teacher_probs, -1)
        tail_mask = cum_teacher_probs.le(alpha).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        fkl = self.compute_forward_kl_divergence(logits, teacher_logits, target, attention_mask=attention_mask, reduction="none")
        rkl = self.compute_reverse_kl_divergence(logits, teacher_logits, target, attention_mask=attention_mask, reduction="none")

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl

        if reduction == "sum":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            akl = akl.masked_fill_(pad_mask, 0.0)
            akl = akl.sum()
        elif reduction == "mean":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            akl = akl.masked_fill_(pad_mask, 0.0)
            num_valid_elements = (~pad_mask).sum().float()
            akl = akl.sum() / num_valid_elements

        return akl

    def compute_skewed_forward_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        attention_mask=None,
        reduction="mean"
    ):
        logits = logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.teacher_temperature

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = self.args.skew_lambda * teacher_probs + (1 - self.args.skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()
        elif reduction == "mean":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            num_valid_elements = (~pad_mask).sum().float()
            kld = kld.sum() / num_valid_elements

        return kld

    def compute_skewed_reverse_kl_divergence(
        self,
        logits,
        teacher_logits,
        target,
        attention_mask=None,
        reduction="mean"
    ):
        logits = logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.teacher_temperature

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = (1 - self.args.skew_lambda) * teacher_probs + self.args.skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        kld = (student_probs * (student_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()
        elif reduction == "mean":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            num_valid_elements = (~pad_mask).sum().float()
            kld = kld.sum() / num_valid_elements

        return kld

    def compute_js_divergence(
        self,
        logits,
        teacher_logits,
        target,
        attention_mask=None,
        reduction="mean"
    ):
        # temperature scaling
        logits = logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.teacher_temperature

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        m_probs = (probs + teacher_probs) / 2

        lprobs = torch.log(probs + 1e-9)
        teacher_lprobs = torch.log(teacher_probs + 1e-9)
        m_lprobs = torch.log(m_probs + 1e-9)

        kld1 = teacher_probs * (teacher_lprobs - m_lprobs)
        kld2 = probs * (lprobs - m_lprobs)
        kld = (kld1 + kld2) / 2
        inf_mask = logits.isinf() | teacher_logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()
        elif reduction == "mean":
            if attention_mask is None:
                pad_mask = target.eq(self.padding_id)
            else:
                pad_mask = attention_mask.eq(0)
            kld = kld.masked_fill_(pad_mask, 0.0)
            num_valid_elements = (~pad_mask).sum().float()
            kld = kld.sum() / num_valid_elements

        return kld
