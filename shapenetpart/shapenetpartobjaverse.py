import os
import glob
import h5py
import json
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps, furthest_point_sample
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import open3d as o3d

def rotate_pts(pts, angles, device=None): # list of points as a tensor, N*3

    roll = angles[0].reshape(1)
    yaw = angles[1].reshape(1)
    pitch = angles[2].reshape(1)

    tensor_0 = torch.zeros(1).to(device)
    tensor_1 = torch.ones(1).to(device)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

    RY = torch.stack([
                    torch.stack([torch.cos(yaw), tensor_0, torch.sin(yaw)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(yaw), tensor_0, torch.cos(yaw)])]).reshape(3,3)

    RZ = torch.stack([
                    torch.stack([torch.cos(pitch), -torch.sin(pitch), tensor_0]),
                    torch.stack([torch.sin(pitch), torch.cos(pitch), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    if device == "cuda":
        R = R.cuda()
    pts_new = torch.mm(pts, R.T)
    return pts_new

@DATASETS.register_module()
class ShapeNetPartObjaverse(Dataset):
    classes = ['airplane', 'bag', 'cap', 'car', 'chair',
               'earphone', 'guitar', 'knife', 'lamp', 'laptop',
               'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
                   'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
                   'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
                   'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}
    cls2parts = []
    cls2partembed = torch.zeros(16, 50)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat

    def __init__(self, num_points=2048,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=16,
                 presample=False,
                 sampler='fps', 
                 transform=None,
                 multihead=False,
                 **kwargs):
        class_uids = sorted(os.listdir(f"/data/ziqi/objaverse/holdout/shapenetpart"))
        self.obj_path_list = [f"/data/ziqi/objaverse/holdout/shapenetpart/{class_uid}" for class_uid in class_uids]
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.eye = np.eye(16)
        self.transform = transform
        self.name_map={"airplane":"airplane",
                       "handbag":"bag",
                       "cap_headwear":"cap",
                       "baseball_cap":"cap",
                       "car_automobile":"car",
                       "motor_vehicle":"car",
                       "armchair":"chair",
                       "chair":"chair",
                       "earphone":"earphone",
                       "guitar":"guitar",
                       "knife":"knife",
                       "lamp":"lamp",
                       "laptop_computer":"laptop",
                       "motorcycle":"motorbike",
                       "mug":"mug",
                       "pistol":"pistol",
                       "gun":"pistol",
                       "skateboard":"skateboard",
                       "table":"table",
                       "rocket":"rocket"}
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    def __getitem__(self, item):
        file_path = self.obj_path_list[item]
        classname = "_".join(file_path.split("/")[-1].split("_")[:-1])
        label = self.cat2id[self.name_map[classname]]
        pcd = o3d.io.read_point_cloud(f"{file_path}/points5000.pcd")
        normal = torch.tensor(np.asarray(pcd.normals)).float()
        pts_xyz = torch.tensor(np.asarray(pcd.points)).float()
        seg = torch.tensor(np.load(f"{file_path}/labels.npy")) - 1+self.index_start[label]  # so that start-1 is unlabeled

        # subsample 2048 pts
        random_indices = torch.randint(0, pts_xyz.shape[0], (2048,))
        pts_xyz_subsample = pts_xyz[random_indices]
        normal_subsample = normal[random_indices]
        seg_subsample = seg[random_indices]

        # this is model-wise one-hot enocoding for 16 categories of shapes
        cls = np.array([label]).astype(np.int64)
        data = {'pos': pts_xyz_subsample,
                'x': normal_subsample,
                'cls': cls,
                'y': seg_subsample,
                'xyz_sub': pts_xyz_subsample, # keep a copy so that this does not pass through augmentation
                'xyz_full': pts_xyz,
                'y_full': seg}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.obj_path_list)
