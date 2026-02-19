import torch 
import torch.nn as nn
import torch.nn.functional as f
import random

from Transformers_Block import Block

corpus = [
"The quick brown fox jumps over the lazy dog.",
"Artificial intelligence is transforming the way we solve complex problems.",
"Graph neural networks are powerful tools for modeling relational data.",
"Diffusion models generate images by progressively removing noise.",
"Effective machine learning systems require both strong theory and practical experimentation."
]

corpus = [s + " <END> " for s in corpus]
text = " ".join(corpus)
print(text)
