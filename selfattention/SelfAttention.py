import torch.nn as nn
import torch
from pydantic.dataclasses import dataclass
from pydantic import RootModel
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Dict, List, Optional
from dataclasses import replace
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import reduce

@dataclass(config={'frozen': True, 'arbitrary_types_allowed':True})
class SelfAttentionV1(nn.Module):
    WQuery: nn.Parameter
    WKey: nn.Parameter
    WValue: nn.Parameter
    # def __init__(self, d_in,d_out):
    #     super().__init__()

def initSelfAttentionV1(ind: int, outd: int):
    q = nn.Parameter(torch.rand(ind, outd))
    k = nn.Parameter(torch.rand(ind, outd))
    v = nn.Parameter(torch.rand(ind, outd))
    attention: SelfAttentionV1 = SelfAttentionV1(q, k, v)
    return attention
    
    
def forward(attention: SelfAttentionV1, x: torch.Tensor):
    keys = x @ attention.WKey
    queries = x @ attention.WQuery
    values = x @ attention.WValue
    attentionScores = queries @ keys.T
    attentionWeights = torch.softmax(
        attentionScores / keys.shape[-1]**0.5, dim=-1
    )
    contextVector = attentionWeights @ values
    return contextVector
