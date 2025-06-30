from VQMaster.vector_quantize_pytorch import VectorQuantize
import torch

import torch
import torchvision
import matplotlib
import numpy
import scipy
import six

vq = VectorQuantize(
    dim = 256,
    codebook_size = 512,     # codebook size
    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
    commitment_weight = 1.   # the weight on the commitment loss
)

modulestruct = vq._modules
print("Module structure:", modulestruct)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x) # (1, 1024, 256), (1, 1024), (1)

print("quantized shape:", quantized.shape)  # should be (1, 1024, 256)
print("indices shape:", indices.shape)      # should be (1, 1024)
print("commit_loss shape:", commit_loss.shape)  # should be (1,)


#================================================================================