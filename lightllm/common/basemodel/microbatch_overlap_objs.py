import torch
from dataclasses import dataclass


@dataclass
class DecodeMicroBatch:
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    input_ids: torch.Tensor
    mem_indexes: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor


@dataclass
class PrefillMicroBatch:
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    input_ids: torch.Tensor
    mem_indexes: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    b_ready_cache_len: torch.Tensor
    multimodal_params: list
