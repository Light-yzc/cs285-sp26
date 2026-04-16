from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # del batch, minibatch_size, shuffle, generator, device
    # TODO(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.
    # raise NotImplementedError("Implement iter_minibatches in the student starter.")
    bs = batch.input_ids.shape[0]
    if shuffle:
        indices = torch.randperm(bs, generator=generator)
    else:
        indices = torch.arange(0, bs)
    
    for i in range(0, bs, minibatch_size):
        if i + minibatch_size < bs: 
            cur_indices = indices[i:i+minibatch_size]
        else:
            cur_indices = indices[i:]
        # cur_task_name = [task_name for i, task_name in enumerate(batch.task_names) if i in cur_indices.tolist() ]
        # cur_completion_texts = [completion_text for i, completion_text in enumerate(batch.completion_texts) if i in cur_indices.tolist() ]

        task_names = [batch.task_names[j] for j in cur_indices.tolist()]
        completion_texts = [batch.completion_texts[j] for j in cur_indices.tolist()]

        yield RolloutBatch(
            input_ids=batch.input_ids[cur_indices],
            attention_mask=batch.attention_mask[cur_indices],
            completion_mask=batch.completion_mask[cur_indices],
            old_logprobs=batch.old_logprobs[cur_indices],
            ref_logprobs=batch.ref_logprobs[cur_indices],
            rewards=batch.rewards[cur_indices],
            advantages=batch.advantages[cur_indices],
            task_names=task_names,
            completion_texts=completion_texts, 
        ).to(device)
    