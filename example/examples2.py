# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/shen/myproject/habitat/semantic_anticipation_2d')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from rl.semantic_anticipator import SemAnt2D
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# %%
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
class CustomDataset(Dataset):
    def __init__(self, ground_truth, costmap, num_catagories, catagories):
        self.ground_truth = ground_truth
        self.costmap = costmap
        self.num_catagories = num_catagories
        self.catagories = catagories
    
    def __len__(self):
        return len(list(self.ground_truth.keys()))
    
    def __getitem__(self,idx):
        index =  list(self.ground_truth.keys())[idx]
        gt = np.zeros((self.num_catagories, self.ground_truth[index].shape[0], self.ground_truth[index].shape[1]))
        for i in range(self.num_catagories):
            gt[i][self.ground_truth[index] > 0] = 1
        lidar = self.costmap[index]
        return lidar[np.newaxis], gt

# %%
# %%
ground_truth = np.load('../data/1/2_2_container_ground_truth_id.npz')
costmap = np.load('../data/1/2_1_container_costmap_id.npz')
num_catagories = 1
catagories = [0]
MapDataset = CustomDataset(ground_truth, costmap, num_catagories,catagories)
train_dataloader = DataLoader(MapDataset, batch_size=8, shuffle=True)
# %%
def simple_mapping_loss_fn(pt_hat, pt_gt):
    num_catagories = pt_hat.shape[1]
    mapping_loss = 0
    for i in range(num_catagories):
        occupied_hat = pt_hat[:,i] # (T*N, V, V)
        occupied_gt = pt_gt[:,i]  # (T*N, V, V)
        weight = torch.ones(occupied_gt.shape).to(pt_gt.device)
        weight[occupied_gt > 0] *= 5
        occupied_mapping_loss = F.binary_cross_entropy(occupied_hat, occupied_gt, weight)
        mapping_loss += occupied_mapping_loss 
    return mapping_loss

# %%
running_loss = 0.0
device = 'cuda:0'
batch_size = 8
ego_map_size = 60
anticipator = SemAnt2D(1,num_catagories,32).to(device)
optimizer = optim.SGD(anticipator.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):
    for i, data in  enumerate(train_dataloader, 0):
        lidar, labels = data
        lidar = lidar.type(torch.float32).to(device)
        labels = labels.type(torch.float32).to(device)
        optimizer.zero_grad()
        output = anticipator(lidar)
        loss = simple_mapping_loss_fn(output["occ_estimate"], labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/10))
            running_loss = 0.0
# %%
