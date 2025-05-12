import hydra
from omegaconf import DictConfig
import open3d as o3d
import trimesh
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets.ModelNet40Ply2048 import ModelNet40Ply2048
from model import Adapt_classf_pl
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import LearningRateMonitor
#from point_transformer_cls import PCT_PL
from fvcore.nn import FlopCountAnalysis
import json
from datasets.inside_mesh import check_mesh_contains



cfg = DictConfig
gt_dir = "../ground_truth"
dir_train = "datasets/modelnet40_ply_hdf5_2048/train/"
dir_test = "datasets/modelnet40_ply_hdf5_2048/test/"
modelnet_train = ModelNet40Ply2048("datasets/modelnet40_ply_hdf5_2048", split="train")  # type: ignore
modelnet_test = ModelNet40Ply2048("datasets/modelnet40_ply_hdf5_2048", split="test")  # type: ignore


len_train = modelnet_train.__len__()
len_test = modelnet_test.__len__()

for i in range(len_test):
    points, _, label, filename = modelnet_test.__getitem__(i)
    split = modelnet_test.split
    filename = os.path.splitext(filename)[0]+"_id2file.json"
    print(filename)
    f = open(filename, 'r')
    name = json.loads(f.read())[label]
    print(label, name)
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    #pcd = o3d.geometry.PointCloud()
    # pcd = o3d.t.geometry.PointCloud(
    #     np.array(points, dtype=np.float32))
    #pcd.points = o3d.utility.Vector3dVector(points)
    #print(label)
    folder = os.path.join("datasets/dataset", os.path.split(name)[0], split, os.path.splitext(os.path.split(name)[1])[0])
    if not os.path.exists(folder): os.makedirs(folder)
    #folder = os.path.join(folder, split)
    #if not os.path.exists(folder): os.mkdir(folder)
    gt_path = os.path.join(gt_dir, os.path.split(name)[0], split, os.path.splitext(os.path.split(name)[1])[0]+ ".off")
    #print(gt_path)
    mesh = trimesh.load(gt_path)
    points_size = 100000
    points_uniform_ratio = 1.
    points_padding = 0.1
    points_sigma = 0.01
    n_points_uniform = int(points_size * points_uniform_ratio)
    n_points_surface = points_size - n_points_uniform

    boxsize = 1 + points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    points_surface = mesh.sample(n_points_surface)
    points_surface += points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)
    points = points.astype(np.float32)


    occupancies = check_mesh_contains(mesh, points)
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    filename = os.path.join(folder, "points.npz")

    np.savez(filename, points=points, occupancies=occupancies,
             loc=loc, scale=scale)

    pcd_points, face_idx = mesh.sample(100000, return_index=True)
    pcd_points = pcd_points.astype(np.float32)
    normals = mesh.face_normals[face_idx]
    normals = normals.astype(np.float32)
    filename = os.path.join(folder, "pointcloud.npz")
    np.savez(filename, points=pcd_points, normals=normals, loc=loc, scale=scale)

    #o3d.io.write_point_cloud(os.path.join(folder,os.path.split(name)[1]), pcd)

    # o3d.visualization.draw_plotly([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

