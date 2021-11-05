# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.insert(0,'/home/shen/myproject/habitat/semantic_anticipation_2d')
sys.path.insert(0,'/home/m-yordanova/catkin_ws_ma/src/rosnav-imagination')

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
ground_truth = np.load('../data/1/2_2_container_ground_truth_id.npz') # len = 205
ground_truth_1 = np.load('../data/1/2_2_container_ground_truth_id.npz') # len = 205
ground_truth_2 = np.load('../data/2/2_2_container_ground_truth_id.npz') # len = 244
ground_truth_3 = np.load('../data/3/2_2_container_ground_truth_id.npz') # len = 291
ground_truth_4 = np.load('../data/4/2_2_container_ground_truth_id.npz') # len = 318
ground_truth_5 = np.load('../data/5/2_2_container_ground_truth_id.npz') # len = 342

### combine and sort all ground truth data
### all npz files have the same names inside
### extract the npy files from each npz file and then bild a single npz file from all npy files
ground_truth_ar = [] # len = 1400
for i in range(len(ground_truth)):
    ground_truth_ar.append(ground_truth['arr_' + str(i)])
for i in range(len(ground_truth_2)):
    ground_truth_ar.append(ground_truth_2['arr_' + str(i)])
for i in range(len(ground_truth_3)):
    ground_truth_ar.append(ground_truth_3['arr_' + str(i)])
for i in range(len(ground_truth_4)):
    ground_truth_ar.append(ground_truth_4['arr_' + str(i)])
for i in range(len(ground_truth_5)):
    ground_truth_ar.append(ground_truth_5['arr_' + str(i)])
np.savez('../data/all/ground_truth_all.npz', *ground_truth_ar)
ground_truth_data_ar = np.load('../data/all/ground_truth_all.npz')
#print(ground_truth_data_ar) # correct type: numpy.lib.npyio.NpzFile object
#print(ground_truth_data_ar.files) # correct structure: ['arr_0', 'arr_1', 'arr_2', ...]
#print(ground_truth_data_ar['arr_0']) # right: [[0. 0. 0. ... 7. 0. 0. ...]]
#print(ground_truth_data_ar.files[0]) # wrong: arr_0; no data inside

### combine and sort all costmap data
costmap = np.load('../data/1/2_1_container_costmap_id.npz') # len = 205
costmap_1 = np.load('../data/1/2_1_container_costmap_id.npz') # len = 205
costmap_2 = np.load('../data/2/2_1_container_costmap_id.npz') # len = 244
costmap_3 = np.load('../data/3/2_1_container_costmap_id.npz') # len = 291
costmap_4 = np.load('../data/4/2_1_container_costmap_id.npz') # len = 318
costmap_5 = np.load('../data/5/2_1_container_costmap_id.npz') # len = 342
costmap_ar = [] # len = 1400
for i in range(len(costmap_1)):
    costmap_ar.append(costmap_1['arr_' + str(i)])
for i in range(len(costmap_2)):
    costmap_ar.append(costmap_2['arr_' + str(i)])
for i in range(len(costmap_3)):
    costmap_ar.append(costmap_3['arr_' + str(i)])
for i in range(len(costmap_4)):
    costmap_ar.append(costmap_4['arr_' + str(i)])
for i in range(len(costmap_5)):
    costmap_ar.append(costmap_5['arr_' + str(i)])
np.savez('../data/all/costmap_all.npz', *costmap_ar)
costmap_data_ar = np.load('../data/all/costmap_all.npz')

num_catagories = 1
catagories = [0]
MapDataset = CustomDataset(ground_truth_data_ar, costmap_data_ar, num_catagories,catagories)
train_dataloader = DataLoader(MapDataset, batch_size=8, shuffle=False) # shuffle=True/False
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
#device = 'cuda:0'
device = 'cpu'
batch_size = 8 # 8 -> 24 -> 32
ego_map_size = 60
anticipator = SemAnt2D(1,num_catagories,32).to(device)
optimizer = optim.SGD(anticipator.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20): # increase!
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
# make sure that the observation and ground truth data pairs match
lidar, labels = next(iter(train_dataloader))
labels.shape # torch.Size([8, 1, 60, 60])
plt.imshow(labels[1,0])
plt.figure()
plt.imshow(lidar[1,0])
# %%
