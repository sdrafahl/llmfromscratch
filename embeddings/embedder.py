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
class Embedder:
    token_embedding_layer: torch.nn.Embedding
    positional_embedding_layer: torch.nn.Embedding
    # You will eventually need a positional embedding layer here too!

def createEmbedding(dl: DataLoader, seed: int = 123):
    torch.manual_seed(seed)
    
    # 1. Get the vocab size from the tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")
    vocab_size = encoding.n_vocab
    
    # 2. Pick your embedding dimension (hyperparameter)
    embedding_dim = 256 
    
    # 3. Create the PyTorch layer
    embeddingLayer = torch.nn.Embedding(vocab_size, embedding_dim)

    # 4. Create the Positional Embedding Layer
    # Unlike token embeddings that need to know all 100,277 words, 
    # positional embeddings only need to know the max length of a sequence!
    # In dataloader.py, you set contextSize to 256, so we'll use that here.
    max_context_length = 256 
    positionEmbeddingLayer = torch.nn.Embedding(max_context_length, embedding_dim)
    
    return Embedder(
        token_embedding_layer=embeddingLayer, 
        positional_embedding_layer=positionEmbeddingLayer
    )
