from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    template: Optional[str] = field(
        default="native",
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    domain_interleave_probs: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from domains in one batch."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    file_name: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "The name of file path name for train. Conflicts with `--dataset_name`"},
    )
    eval_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of file path name for eval. Conflicts with `--eval_dataset_name`"},
    )
    dataset_type: Optional[Union[List[str], str]] = field(
        default="json",
        metadata={"help": "The dataset type, for example, json."},
    )
    tag: Optional[str] = field(default="tag", metadata={"help": "Which column in file to use as domain selection"})
    id: Optional[str] = field(default="id", metadata={"help": "Which column in file to use as id"})
    prompt: Optional[str] = field(default=None, metadata={"help": "Which column in file to use as prompt"})
    response: Optional[str] = field(default="solution", metadata={"help": "Which column in file to use as label"})
    # image: Optional[str] = field(default='image', metadata={"help": "Which column in file to use as image"})
    messages: Optional[str] = field(default=None, metadata={"help": "Which column in file to use as messages"})

    def __post_init__(self):
        assert not (
            self.prompt is not None and self.messages is not None
        ), "prompt and messages are mutually exclusive"
