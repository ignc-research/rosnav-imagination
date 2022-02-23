# %% (runs on (anaconda (base)) python environment)
import numpy as np
import matplotlib.pyplot as plt
import os
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
from torchvision.utils import save_image
import cv2

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
        # version 1:
        gt = np.zeros((self.num_catagories, self.ground_truth[index].shape[0], self.ground_truth[index].shape[1]))
        for i in range(self.num_catagories):
            gt[i][self.ground_truth[index] > 0] = 1 # one layer ground truth data for now, so occupied or not # TODO X
            #gt[i][self.ground_truth[index] == 50] = 0 # TODO X: = 0/1/2
        lidar = self.costmap[index]
        return lidar[np.newaxis], gt
        # version 2: (TODO X)
    #    gt = self.ground_truth[index]
    #    lidar = self.costmap[index]
    #    return lidar[np.newaxis], gt[np.newaxis]

class CustomDatasetSingle(Dataset): # (TODO) test dataset generator for a single npy file
    def __init__(self, ground_truth, costmap, num_catagories, catagories):
        self.ground_truth = ground_truth
        self.costmap = costmap
        self.num_catagories = num_catagories
        self.catagories = catagories
    
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        # version 1:
        gt = np.zeros((self.num_catagories, self.ground_truth.shape[0], self.ground_truth.shape[1]))
        for i in range(self.num_catagories):
            gt[i][self.ground_truth > 0] = 1 # one layer ground truth data for now, so occupied or not # TODO X
            #gt[i][self.ground_truth > 0.2] = 1 # TODO
        lidar = self.costmap
        return lidar[np.newaxis], gt
        # version 2: (TODO X)
    #    gt = self.ground_truth
    #    lidar = self.costmap
    #    return lidar[np.newaxis], gt[np.newaxis]

# %%
#cwd = os.getcwd()
#print(cwd) # get the current working directory

img_range = 100 # 60/80/100/...
img_range_str = str(img_range) + 'x' + str(img_range)
## Group Info: (TODO X)
# grey laser scan data & gt normal & laser scan always 60x60 : "grey_laser_scan_60x60" (for 80x80 and 100x100)
# semantic laser scan data & gt normal: "/semantic_robot_sync/gt_normal" (for 60x60, 80x80 and 100x100)
# semantic laser scan data & gt extended: "/semantic_robot_sync/gt_extension" (for 60x60, 80x80 and 100x100)
group = "/semantic_robot_sync/gt_extension"

#ground_truth = np.load('../data/100x100/1/2_2_container_ground_truth_id.npz')
ground_truth = np.load('../data/' + img_range_str + group + '/1/2_2_container_ground_truth_id.npz') # len = 205
ground_truth_1 = np.load('../data/' + img_range_str + group + '/1/2_2_container_ground_truth_id.npz') # len = 205
ground_truth_2 = np.load('../data/' + img_range_str + group + '/2/2_2_container_ground_truth_id.npz') # len = 244
ground_truth_3 = np.load('../data/' + img_range_str + group + '/3/2_2_container_ground_truth_id.npz') # len = 291
ground_truth_4 = np.load('../data/' + img_range_str + group + '/4/2_2_container_ground_truth_id.npz') # len = 318
ground_truth_5 = np.load('../data/' + img_range_str + group + '/5/2_2_container_ground_truth_id.npz') # len = 342
ground_truth_collect = [ground_truth_1, ground_truth_2, ground_truth_3, ground_truth_4, ground_truth_5]
### combine and sort all ground truth data
### all npz files have the same names inside
### extract the npy files from each npz file and then bild a single npz file from all npy files
ground_truth_ar = [] # len = 1400
for ground_truth in ground_truth_collect:
    for c in range(len(ground_truth)):
        ground_truth_ar.append(ground_truth['arr_' + str(c)])
np.savez('../data/' + img_range_str + group + '/all/ground_truth_all.npz', *ground_truth_ar)
ground_truth_data_ar = np.load('../data/' + img_range_str + group + '/all/ground_truth_all.npz')
#print(ground_truth_data_ar) # correct type: numpy.lib.npyio.NpzFile object
#print(ground_truth_data_ar.files) # correct structure: ['arr_0', 'arr_1', 'arr_2', ...]
#print(ground_truth_data_ar['arr_0']) # right: [[0. 0. 0. ... 7. 0. 0. ...]]
#print(ground_truth_data_ar.files[0]) # wrong: arr_0; no data inside

### combine and sort all costmap data
costmap = np.load('../data/' + img_range_str + group + '/1/2_1_container_costmap_id.npz') # len = 205
costmap_1 = np.load('../data/' + img_range_str + group + '/1/2_1_container_costmap_id.npz') # len = 205
costmap_2 = np.load('../data/' + img_range_str + group + '/2/2_1_container_costmap_id.npz') # len = 244
costmap_3 = np.load('../data/' + img_range_str + group + '/3/2_1_container_costmap_id.npz') # len = 291
costmap_4 = np.load('../data/' + img_range_str + group + '/4/2_1_container_costmap_id.npz') # len = 318
costmap_5 = np.load('../data/' + img_range_str + group + '/5/2_1_container_costmap_id.npz') # len = 342
costmap_collect = [costmap_1, costmap_2, costmap_3, costmap_4, costmap_5]
costmap_ar = [] # len = 1400
for costmap in costmap_collect:
    for c in range(len(costmap)):
        temp_npy_file = costmap['arr_' + str(c)]
        range_old = len(temp_npy_file) # 60/80/100
        step = int((img_range - range_old)/2) # (100-60)/2=20
        new_npy_file = np.full((img_range,img_range), 50) # TODO: init with 50.0 (color=unknown) instead of 0.0 (black=free)
        # Important (TODO X): if the costmap should be bigger (it should be 80x80/100x100, but it is 60x60), expand it here (with IDs!)
        if img_range != range_old:
            for i in range(img_range):
                for j in range(img_range):
                    if not((i < step or i >= (range_old + step)) or (j < step or j >= (range_old + step))): # border from all sides: 0-19 & 80-99
                        new_npy_file[i,j] = temp_npy_file[i-step,j-step] # take the id only in the middle, the inside of the borders
            costmap_ar.append(new_npy_file) # costmap['arr_' + str(c)] -> new_npy_file
        else:
            costmap_ar.append(costmap['arr_' + str(c)])
        # TODO X: if the costmap should be smaller (if it is 100x100, but should be 60x60)
np.savez('../data/' + img_range_str + group + '/all/costmap_all.npz', *costmap_ar)
costmap_data_ar = np.load('../data/' + img_range_str + group + '/all/costmap_all.npz')

# TODO X: test with costmap images with black/white/colored border:
#group = "/test/test_border_id_50" # "/test/test_border_black_free" / "/test/test_border_white" / "/test/test_border_id_50"
#ground_truth_data_ar = np.load('../data/' + img_range_str + group + '/2_2_container_ground_truth_id.npz')
#costmap_data_ar = np.load('../data/' + img_range_str + group + '/2_1_container_costmap_id.npz')

num_catagories = 1 # for now only one category of occupied area # expand later on! # TODO X: 1 -> 2 ?
catagories = [0] # a list! # TODO X: [0] -> [0,1/2] ?
batch_size = 8 # 8 -> 24 -> 32
MapDataset = CustomDataset(ground_truth_data_ar, costmap_data_ar, num_catagories, catagories)
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
ego_map_size = 60 # 60 / 80 ? not used
num_import_layers = 1 # 1
num_output_layers = num_catagories # for now =1, extend later on! # num_catagories
network_size = 32 # 16/32/64
anticipator = SemAnt2D(num_import_layers,num_output_layers,network_size).to(device) # init the model
optimizer = optim.SGD(anticipator.parameters(), lr=0.001, momentum=0.9) # init the optimizer

# %%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
iterations = 11 # increase from 11 to 101/1001/2001/10000/100000/1000000
for epoch in range(iterations): # training loop
    for i, data in  enumerate(train_dataloader, 0):
        lidar, labels = data
        lidar = lidar.type(torch.float32).to(device)
        labels = labels.type(torch.float32).to(device)
        optimizer.zero_grad()
        # model(x) = y1 ?= y | model(observation = lidar) = result = labels ?= ground truth labels
        output = anticipator(lidar) # put data into the anticipator (predictor)
        # TODO: lidar has 60x60, but output should have 100x100 => change the model
        # TODO: maybe just make the lidar data 100x100 by adding black space all around?!?
        #print(output["occ_estimate"].shape) # torch.Size([8, 1, 60, 60]) # [batch_size, channels, height, width]
        #print(output["occ_estimate"].shape) # torch.Size([8, 1, 60, 60])
        loss = simple_mapping_loss_fn(output["occ_estimate"], labels) # TODO: error
        writer.add_scalar("Loss/train", loss, epoch) # log the loss value
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/10))
            running_loss = 0.0
    if epoch % 100 == 0: # save the model every X epochs
        #path = "./models_state_dict/model_" + str(epoch) + ".pth" # .pt/.pth
        #torch.save({
        #            'epoch': epoch,
        #            'model_state_dict': anticipator.state_dict(),
        #            'optimizer_state_dict': optimizer.state_dict(),
        #            'loss': loss
        #            }, path)
        path2 = "./models/model_" + str(epoch) + ".pth" # .pt/.pth
        torch.save(anticipator, path2)
    # Statistics:
    # 2760 trained models (with epoch, loss etc.) for 19h as 69 GB, where a model for each epoch was saved
    # 101 trained entire models for 40 min as 2 files of 25.6 MB all together, where a model for each 100 epochs was saved
    # "./models_state_dict/model_0" (25.6 MB) is the double size of "./models/model_0" (12.8 MB)
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
path = "./models_state_dict/model_final.pth" # .pt/.pth
torch.save(anticipator, path)
model = torch.load(path)
# how to get the epoch, loss etc. from the saved and loaded entire model?
model.eval()
#model.train() # outputs SemAnt2D()

# %%
# test - load a couple of the saved models to see the difference
checkpoint_0 = torch.load("./models_state_dict/model_0.pth")
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_test_0 = checkpoint_0['epoch']
loss_test_0 = checkpoint_0['loss']
print("epoch: " + str(epoch_test_0)) # 0
print("loss: " + str(loss_test_0)) # tensor(2.6073, requires_grad=True)
#model.eval()
#model.train()

checkpoint_10 = torch.load("./models_state_dict/model_10.pth")
epoch_test_10 = checkpoint_10['epoch']
loss_test_10 = checkpoint_10['loss']
print("epoch: " + str(epoch_test_10)) # 10
print("loss: " + str(loss_test_10)) # 2.4177

checkpoint_1000 = torch.load("./models_state_dict/model_1000.pth")
epoch_test_1000 = checkpoint_1000['epoch']
loss_test_1000 = checkpoint_1000['loss']
print("epoch: " + str(epoch_test_1000)) # 1000
print("loss: " + str(loss_test_1000)) # 2.2264

checkpoint_2000 = torch.load("./models_state_dict/model_2000.pth")
epoch_test_2000 = checkpoint_1000['epoch']
loss_test_2000 = checkpoint_1000['loss']
print("epoch: " + str(epoch_test_2000)) # 2000
print("loss: " + str(loss_test_2000)) # 2.2264

# %%
# Test the model if the prediction is reasonable (TODO X)
# Show the output as an image and compare it with the ground truth

# get the observation (costmap/lidar) and ground truth (labels) data
lidar, labels = next(iter(train_dataloader))
lidar = lidar.type(torch.float32).to(device) # convert float to double
labels = labels.type(torch.float32).to(device)

# load the model
model_0 = torch.load("./models/model_0.pth")
model_100 = torch.load("./models/model_100.pth")
model_1000 = torch.load("./models/model_1000.pth")
checkpoint_1000 = torch.load("./models_state_dict/model_1000.pth")
checkpoint_2760 = torch.load("./models_state_dict/model_2760.pth")

# get the prediction (predicted labels): prediction = model(observation)
labels_0 = model_0(lidar)
labels_100 = model_100(lidar)
labels_1000 = model_1000(lidar)
anticipator.load_state_dict(checkpoint_1000['model_state_dict']) # ! first load the disc state to a model and then use that model to get the prediction
output_1000 = anticipator(lidar)
anticipator.load_state_dict(checkpoint_2760['model_state_dict']) # !
output_2760 = anticipator(lidar)

labels.shape # shape: torch.Size([8, 1, 60, 60])
labels_100["occ_estimate"].shape # shape: torch.Size([8, 1, 60, 60])

plt.imshow(lidar[0,0]) # observation
plt.figure()
plt.imshow(labels[0,0]) # ground truth
plt.figure()
plt.imshow(labels_0["occ_estimate"].detach()[0,0]) # prediction
plt.figure()
plt.imshow(labels_100["occ_estimate"].detach()[0,0]) # prediction
plt.figure()
plt.imshow(labels_1000["occ_estimate"].detach()[0,0]) # prediction
plt.figure()
plt.imshow(output_1000["occ_estimate"].detach()[0,0]) # prediction
plt.figure()
plt.imshow(output_2760["occ_estimate"].detach()[0,0]) # prediction

# %%
# Test the model with the test data:
# - create a new scenario (scenario 6) that the robot hasn't trained on
# - or exclude the last made scenario from the training set (scenario5), train again and then use this scenario as test dataset

anticipator.eval()
model_0.eval()
model_100.eval()

# Test the model with test data (1): an example ground truth and observation (costmap) image
# Attention: the model was trained with an ID, not with a rgb color, so transform the color of the images to ids
# img array shape: (60,60,3) with rgb color -> (60,60) with id
# for now we have only one category, so the ids could be just 0 and 1 for free and occupied

img_costmap = cv2.imread('../data_test/costmap_test1.png')
img_costmap_id = np.zeros((img_costmap.shape[0],img_costmap.shape[1]))
black_ar = [0,0,0]
for i in range(img_costmap.shape[0]):
    for j in range(img_costmap.shape[1]):
        BGR_color = [img_costmap[i, j, 0], img_costmap[i, j, 1], img_costmap[i, j, 2]]
        id_temp = 1 # one category for now, per default occupied
        if BGR_color == black_ar:
            id_temp = 0
        img_costmap_id[i,j] = id_temp
img_costmap_ar = np.asarray(img_costmap_id)
np.save('../data_test/costmap_id_test1.npy', img_costmap_ar)
img_costmap_npy = np.load('../data_test/costmap_id_test1.npy')

img_ground_truth_map = cv2.imread('../data_test/ground_truth_test1.png')
img_ground_truth_map_id = np.zeros((img_ground_truth_map.shape[0],img_ground_truth_map.shape[1]))
black_ar = [0,0,0]
for i in range(img_ground_truth_map.shape[0]):
    for j in range(img_ground_truth_map.shape[1]):
        BGR_color = [img_ground_truth_map[i, j, 0], img_ground_truth_map[i, j, 1], img_ground_truth_map[i, j, 2]]
        id_temp = 1 # one category for now, per default occupied
        if BGR_color == black_ar:
            id_temp = 0
        img_ground_truth_map_id[i,j] = id_temp
img_ground_truth_map_ar = np.asarray(img_ground_truth_map_id)
np.save('../data_test/ground_truth_id_test1.npy', img_ground_truth_map_ar)
img_ground_truth_map_npy = np.load('../data_test/ground_truth_id_test1.npy')

MapDatasetTestNPY = CustomDatasetSingle(img_ground_truth_map_npy, img_costmap_npy, num_catagories, catagories)
test_dataloader_npy = DataLoader(MapDatasetTestNPY, batch_size, shuffle=False)

lidar_test_npy, labels_test_npy = next(iter(test_dataloader_npy))
lidar_test_npy = lidar_test_npy.type(torch.float32).to(device)
labels_test_npy = labels_test_npy.type(torch.float32).to(device)

labels_prediction_npy_0 = model_0(lidar_test_npy)
labels_prediction_npy_100 = model_100(lidar_test_npy)
output_prediction_npy_2760 = anticipator(lidar_test_npy)

#print(model_100) # SemAnt2D(...)
#print(anticipator) # SemAnt2D(...)

# black & white - correct
save_image(labels_test_npy, '../data_test/labels_test.png')
save_image(lidar_test_npy, '../data_test/lidar_test.png')
# black ??? (TODO)
#save_image(labels_prediction_npy_0["occ_estimate"].detach(), '../data_test/labels_prediction_0.png')
save_image(labels_prediction_npy_100["occ_estimate"].detach(), '../data_test/labels_prediction_100.png')
save_image(output_prediction_npy_2760["occ_estimate"].detach(), '../data_test/labels_prediction_2760.png')

labels_prediction_0 = labels_prediction_npy_0["occ_estimate"].detach()
labels_prediction_0 = labels_prediction_0.type(torch.float32).to(device)
labels_prediction_100 = labels_prediction_npy_0["occ_estimate"].detach()
labels_prediction_2760 = labels_prediction_npy_0["occ_estimate"].detach()
#print(labels_prediction_0)
save_image(labels_prediction_0, '../data_test/labels_prediction_0.png')

#print(type(labels_test_npy)) # <class 'torch.Tensor'>
#print(type(labels_prediction_0)) # <class 'torch.Tensor'>

#print(labels_test_npy.shape) # torch.Size([1, 1, 60, 60])
# https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
fig = plt.figure(frameon=False) # (TODO) -> RGB vs. BGR!?
fig.set_size_inches(1,1) # make the foto (60px, 60px)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(labels_test_npy[0,0], aspect='auto') # ground truth # plt.imshow(img_ground_truth_map_npy)
fig.savefig('../data_test/labels_test.png', dpi=60, bbox_inches='tight', pad_inches=0)

# https://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
labels_test = cv2.imread('../data_test/labels_test.png')
labels_test_cv2 = cv2.cvtColor(labels_test, cv2.COLOR_BGR2RGB)
cv2.imwrite('../data_test/labels_test_cv.png', labels_test_cv2)
#fig = plt.figure(frameon=False)
#fig.set_size_inches(1,1) # make the foto (60px, 60px)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#plt.imshow(labels_test_cv2, aspect='auto')
labels_test_cv2_grey = cv2.cvtColor(labels_test, cv2.COLOR_BGR2GRAY)
cv2.imwrite('../data_test/labels_test_cv2_grey.png', labels_test_cv2_grey)

fig = plt.figure(frameon=False)
fig.set_size_inches(1,1) # make the foto (60px, 60px)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(lidar_test_npy[0,0], aspect='auto') # observation # plt.imshow(img_costmap_npy)
fig.savefig('../data_test/lidar_test.png', dpi=60, bbox_inches='tight', pad_inches=0)

lidar_test = cv2.imread('../data_test/lidar_test.png')
lidar_test_cv2_grey = cv2.cvtColor(lidar_test, cv2.COLOR_BGR2GRAY)
cv2.imwrite('../data_test/lidar_test_cv2_grey.png', lidar_test_cv2_grey)

# variant 1
fig = plt.figure(frameon=False)
fig.set_size_inches(1,1) # make the foto (60px, 60px)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(labels_prediction_npy_0["occ_estimate"].detach()[0,0], aspect='auto') # prediction
fig.savefig('../data_test/labels_prediction_0.png', dpi=60, bbox_inches='tight', pad_inches=0)
# variant 2
plt.imsave('../data_test/labels_prediction_0_test.png', labels_prediction_npy_0["occ_estimate"].detach()[0,0])

labels_prediction_0 = cv2.imread('../data_test/labels_prediction_0.png')
labels_prediction_0_cv2_grey = cv2.cvtColor(labels_prediction_0, cv2.COLOR_BGR2GRAY)
cv2.imwrite('../data_test/labels_prediction_0_cv2_grey.png', labels_prediction_0_cv2_grey)

fig = plt.figure(frameon=False)
fig.set_size_inches(1,1) # make the foto (60px, 60px)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(labels_prediction_npy_100["occ_estimate"].detach()[0,0], aspect='auto') # prediction
fig.savefig('../data_test/labels_prediction_100.png', dpi=60, bbox_inches='tight', pad_inches=0)

labels_prediction_100 = cv2.imread('../data_test/labels_prediction_100.png')
labels_prediction_100_cv2_grey = cv2.cvtColor(labels_prediction_100, cv2.COLOR_BGR2GRAY)
cv2.imwrite('../data_test/labels_prediction_100_cv2_grey.png', labels_prediction_100_cv2_grey)

fig = plt.figure(frameon=False)
fig.set_size_inches(1,1) # make the foto (60px, 60px)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(output_prediction_npy_2760["occ_estimate"].detach()[0,0], aspect='auto') # prediction
fig.savefig('../data_test/labels_prediction_2760.png', dpi=60, bbox_inches='tight', pad_inches=0)

labels_prediction_2760 = cv2.imread('../data_test/labels_prediction_2760.png')
labels_prediction_2760_cv2_grey = cv2.cvtColor(labels_prediction_2760, cv2.COLOR_BGR2GRAY)
cv2.imwrite('../data_test/labels_prediction_2760_cv2_grey.png', labels_prediction_2760_cv2_grey)

fig = plt.figure(frameon=False)

# %%
# Test the model with test data (2): a whole test dataset (scenario 6) of ground truth and observation (costmap) pairs
# The npz files should contain an id information!

# load the datasets
#img_ground_truth_map_npz = np.load('../data_test/test_ground_truth_id.npz')
#img_costmap_npz = np.load('../data_test/test_costmap_id.npz')
img_ground_truth_map_npz = np.load('../data_test/6/2_2_container_ground_truth_id.npz')
img_costmap_npz = np.load('../data_test/6/2_1_container_costmap_id.npz')

# bring the datasets into a right format
MapDatasetTestNPZ = CustomDataset(img_ground_truth_map_npz, img_costmap_npz, num_catagories, catagories)
test_dataloader_npz = DataLoader(MapDatasetTestNPZ, batch_size, shuffle=False)

# get an example set of ground truth data and observation data (costmap; labels)
lidar_test_npz, labels_test_npz = next(iter(test_dataloader_npz))

# correct the format
lidar_test_npz = lidar_test_npz.type(torch.float32).to(device)
labels_test_npz = labels_test_npz.type(torch.float32).to(device)

# using different models get the prediction (labels) depending on the observation (lidar)
labels_prediction_npz_0 = model_0(lidar_test_npz)
labels_prediction_npz_100 = model_100(lidar_test_npz)
output_prediction_npz_2760 = anticipator(lidar_test_npz)

#print(labels_test_npz.shape) # torch.Size([8, 1, 60, 60])
# visualization
plt.imshow(labels_test_npz[0,0]) # ground truth
plt.figure()
plt.imshow(lidar_test_npz[0,0]) # observation
plt.figure()
plt.imshow(labels_prediction_npz_0["occ_estimate"].detach()[0,0]) # prediction
plt.figure()
plt.imshow(labels_prediction_npz_100["occ_estimate"].detach()[0,0]) # prediction
plt.figure()
plt.imshow(output_prediction_npz_2760["occ_estimate"].detach()[0,0]) # prediction

# %%
# TODO: do the imagination in real time (move_to_goal.py erweitern!?)
# get current laser scan data (see laser_scan_data.py) -> costmap data 100x100 & robot's position -> put in the model -> get imagination costmap
# from the local information get global information
# TODO: publish the imagination information (the costmap) to rviz

# %%
