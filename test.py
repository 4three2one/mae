import torch
import math

x= torch.rand(1, 196, 768)
mask_ratio = 0.75
N, L, D = x.shape  # batch, length, dim
len_keep = int(L * (1 - mask_ratio))

noise = torch.rand(N,int(math.sqrt(L)),int(math.sqrt(L)), device=x.device)  # noise in [0, 1]
# sort noise for each sample
ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove
ids_shuffle=ids_shuffle[:,:,:int(int(math.sqrt(L)*(1-mask_ratio)))]
addition = torch.arange(0, 14*14, 14, dtype=ids_shuffle.dtype, device=ids_shuffle.device).view(14, 1).repeat(1,3)
ids_shuffle = ids_shuffle + addition
x =ids_shuffle.flatten().reshape(1,-1)
ids_restore = torch.argsort(ids_shuffle, dim=2)

# keep the first subset
ids_keep = ids_shuffle[:, :len_keep]
x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

# generate the binary mask: 0 is keep, 1 is remove
mask = torch.ones([N, L], device=x.device)
mask[:, :len_keep] = 0
# unshuffle to get the binary mask
mask = torch.gather(mask, dim=1, index=ids_restore)