# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/shen/myproject/habitat/semantic_anticipation_2d')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import torchvision
import torch.optim as optim

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import torchvision
import torch.optim as optim
from einops import rearrange
from rl.semantic_anticipator import SemAnt2D
import gym.spaces as spaces

def softmax_2d(x):
    b, h, w = x.shape  
    x_out = F.softmax(rearrange(x, "b h w -> b (h w)"), dim=1)
    x_out = rearrange(x_out, "b (h w) -> b h w", h=h)
    return x_out

def padded_resize(x, size):
    """For an image tensor of size (bs, c, h, w), resize it such that the
    larger dimension (h or w) is scaled to `size` and the other dimension is
    zero-padded on both sides to get `size`.
    """
    h, w = x.shape[2:]
    top_pad = 0
    bot_pad = 0
    left_pad = 0
    right_pad = 0
    if h > w:
        left_pad = (h - w) // 2
        right_pad = (h - w) - left_pad
    elif w > h: 
        top_pad = (w - h) // 2
        bot_pad = (w - h) - top_pad
    x = F.pad(x, (left_pad, right_pad, top_pad, bot_pad))
    x = F.interpolate(x, size, mode="bilinear", align_corners=False)
    return x

# %%
id = 1

ground_truth = cv2.imread('../data/examples/example' + str(id) + '_map_ground_truth_semantic_part.png', 0) 
ground_truth = torchvision.transforms.functional.to_tensor(ground_truth)[np.newaxis]

inputImg = cv2.imread('../data/examples/example' + str(id) + '_map_local_costmap_part_color.png', 0) 
inputImg = torchvision.transforms.functional.to_tensor(inputImg)[np.newaxis]

# %%
inputImg.shape
# %%
def simple_mapping_loss_fn(pt_hat, pt_gt):
    occupied_hat = pt_hat[:,0] # (T*N, V, V)
    occupied_gt = pt_gt[:,0]  # (T*N, V, V)
    occupied_mapping_loss = F.binary_cross_entropy(occupied_hat, occupied_gt)
    mapping_loss = occupied_mapping_loss 
    return mapping_loss

anticipator = SemAnt2D() 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(anticipator.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0
for epoch in range(20):
    running_loss = 0.0
    for i in range(100):
    

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        Output = anticipator(inputImg)
        loss = simple_mapping_loss_fn(Output['occ_estimate'],ground_truth)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 40 == 39:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
# %%
plt.imshow(Output['occ_estimate'].detach().numpy()[0,0])
# %%
simple_mapping_loss_fn(Output['occ_estimate'],inputImg)
# %%
plt.imshow(ground_truth[0,0])
