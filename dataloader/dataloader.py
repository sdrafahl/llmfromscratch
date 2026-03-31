from pydantic.dataclasses import dataclass
from pydantic import RootModel
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Dict, List, Optional
from dataclasses import replace
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import reduce
import torch

@dataclass(config={'frozen': True, 'arbitrary_types_allowed':True})
class Tensors:
    inputTensor: List[torch.Tensor]
    outputTensor: List[torch.Tensor]

def getTensor(idx: int, t: Tensors):
    return t.inputTensor[idx], t.outputTensor[idx]
    
def createTensors(txt: str, contextSize: int, stride: int):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(txt, allowed_special={"<|endoftext|>"})
    tensors: Tensors = reduce(
        lambda acc, i: replace(
            acc, 
            inputTensor = acc.inputTensor + [torch.tensor(tokens[i : i + contextSize])], 
            outputTensor = acc.outputTensor + [torch.tensor(tokens[i + 1 : i + contextSize + 1])]
        ),
        range(0, len(tokens) - contextSize, stride),
        Tensors([], [])                           
    )
    return tensors

class GPTDataSet(Dataset):
    def __init__(self, tensors: Tensors):
        self.tensors = tensors
    
    def __getitem__(self, idx: int):
        return getTensor(idx, self.tensors)

    def __len__(self):
        return len(self.tensors.inputTensor)


# If context length and stride are equal, the input and output wont overlap and wont overfirt during training
def createDataLoader(data: str, batchSize = 4, shuffle=True, dropLast=True, numWorkers=0):
    tensors = createTensors(data, 256, 128)
    dataSet = GPTDataSet(tensors)
    dataLoader = DataLoader(
        dataSet,
        batch_size=batchSize,
        shuffle=shuffle,
        drop_last=dropLast,
        num_workers=numWorkers
    )
    return dataLoader
    
