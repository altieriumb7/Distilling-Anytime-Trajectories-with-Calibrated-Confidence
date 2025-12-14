from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class TokenizedItem:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    uid: str
    budget_t: int


def tokenize_one(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    response: str,
    *,
    add_eos: bool = True,
) -> Dict[str, List[int]]:
    """
    Build:
      prompt_ids = chat_template(system+user, add_generation_prompt=True)
      resp_ids   = tokenize(response, add_special_tokens=False)
      labels     = [-100]*len(prompt_ids) + resp_ids (+eos)
    """
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids

    input_ids = list(prompt_ids) + list(resp_ids)
    if add_eos and (len(input_ids) == 0 or input_ids[-1] != tokenizer.eos_token_id):
        input_ids.append(tokenizer.eos_token_id)

    labels = [-100] * len(prompt_ids) + list(resp_ids)
    if add_eos:
        labels.append(tokenizer.eos_token_id)

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollatorForCausalLM:
    """
    Pads input_ids/attention_mask/labels.
    """
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = [pad(f["input_ids"], self.tokenizer.pad_token_id) for f in features]
        attention_mask = [pad(f["attention_mask"], 0) for f in features]
        labels = [pad(f["labels"], -100) for f in features]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
