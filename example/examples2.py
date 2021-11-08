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
from rl.semantic_anticipator import SemAnt2D # the model
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
class CustomDataset(Dataset): # training dataset generator for the ground truth and observation (costmap) data
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
            gt[i][self.ground_truth[index] > 0] = 1 # one layer ground truth data for now, so occupied or not
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

num_catagories = 1 # for now only one category of occupied area # expand later on!
catagories = [0] # a list!
batch_size = 8 # 8 -> 24 -> 32
MapDataset = CustomDataset(ground_truth_data_ar, costmap_data_ar, num_catagories,catagories)
train_dataloader = DataLoader(MapDataset, batch_size, shuffle=False) # shuffle=True/False
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
#device = 'cuda:0' # run on gpu
device = 'cpu' # run on cpu
ego_map_size = 60
num_import_layers = 1
num_output_layers = num_catagories # for now =1, extend later on!
network_size = 32 # 16/32/64
anticipator = SemAnt2D(num_import_layers,num_output_layers,network_size).to(device) # init the model
optimizer = optim.SGD(anticipator.parameters(), lr=0.001, momentum=0.9) # init the optimizer

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
iterations = 10000 # increase from 20 to 1000/10000/100000/1000000
for epoch in range(iterations): # training loop
    for i, data in  enumerate(train_dataloader, 0):
        lidar, labels = data
        lidar = lidar.type(torch.float32).to(device)
        labels = labels.type(torch.float32).to(device)
        optimizer.zero_grad()
        # model(x) = y1 ?= y | model(observation = lidar) = result = labels ?= ground truth labels
        output = anticipator(lidar) # put data into the anticipator (predictor)
        loss = simple_mapping_loss_fn(output["occ_estimate"], labels)
        writer.add_scalar("Loss/train", loss, epoch) # log the loss value
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/10))
            running_loss = 0.0
    if epoch % 100 == 0: # save the model every X epochs
        #path = "./temp_models/model_" + str(epoch) + ".pth" # .pt/.pth
        #torch.save({
        #            'epoch': epoch,
        #            'model_state_dict': anticipator.state_dict(),
        #            'optimizer_state_dict': optimizer.state_dict(),
        #            'loss': loss
        #            }, path)
        path2 = "./temp_models2/model_" + str(epoch) + ".pth" # .pt/.pth
        torch.save(anticipator, path2)
    # Statistics:
    # 2760 trained models (with epoch, loss etc.) for 19h as 69 GB, where a model for each epoch was saved
    # 101 trained entire models for 40 min as 2 files of 25.6 MB all together, where a model for each 100 epochs was saved
    # "./temp_models/model_0" (25.6 MB) is the double size of "./temp_models2/model_0" (12.8 MB)
writer.flush() # make sure that all pending events have been written to disk
#writer.close()
# %%
# make sure that the observation and ground truth data pairs match
lidar, labels = next(iter(train_dataloader))
labels.shape # torch.Size([8, 1, 60, 60])
plt.imshow(labels[1,0])
plt.figure()
plt.imshow(lidar[1,0])
# %%
# understand the model (anticipator) and the optimizer
print(anticipator)
print(anticipator.parameters())

# print model's (anticipator's) state_dict
print("Model's state_dict:")
for param_tensor in anticipator.state_dict():
    print(param_tensor, "\t", anticipator.state_dict()[param_tensor].size())

# print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
# %%
#save/load the entire model
path = "./temp_models/model_final.pth" # .pt/.pth
torch.save(anticipator, path)
model = torch.load(path)
# how to get the epoch, loss etc. from the saved and loaded entire model?
model.eval()
#model.train() # outputs SemAnt2D()
# %%
# test - load a couple of the saved models to see the difference
checkpoint_0 = torch.load("./temp_models/model_0.pth")
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_test_0 = checkpoint_0['epoch']
loss_test_0 = checkpoint_0['loss']
print("epoch: " + str(epoch_test_0)) # 0
print("loss: " + str(loss_test_0)) # tensor(2.6073, requires_grad=True)
#model.eval()
#model.train()

checkpoint_10 = torch.load("./temp_models/model_10.pth")
epoch_test_10 = checkpoint_10['epoch']
loss_test_10 = checkpoint_10['loss']
print("epoch: " + str(epoch_test_10)) # 10
print("loss: " + str(loss_test_10)) # 2.4177

checkpoint_19 = torch.load("./temp_models/model_19.pth")
epoch_test_19 = checkpoint_19['epoch']
loss_test_19 = checkpoint_19['loss']
print("epoch: " + str(epoch_test_19)) # 19
print("loss: " + str(loss_test_19)) # 2.3573

checkpoint_1000 = torch.load("./temp_models/model_1000.pth")
epoch_test_1000 = checkpoint_1000['epoch']
loss_test_1000 = checkpoint_1000['loss']
print("epoch: " + str(epoch_test_1000)) # 1000
print("loss: " + str(loss_test_1000)) # 2.2264

checkpoint_2000 = torch.load("./temp_models/model_2000.pth")
epoch_test_2000 = checkpoint_1000['epoch']
loss_test_2000 = checkpoint_1000['loss']
print("epoch: " + str(epoch_test_2000)) # 2000
print("loss: " + str(loss_test_2000)) # 2.2264
# %%
