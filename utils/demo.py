import os
import numpy as np
from PIL import Image
import rembg
import PIL
from typing import Any
import torch
import cv2
from tqdm import tqdm
import torchvision


class NormalTransfer:
    def __init__(self):
        self.identity_w2c = torch.tensor([
                                        [0.0,  0.0,  1.0,  0.0],
                                        [ 0.0,  1.0,  0.0,  0.0],
                                        [-1.0, 0.0,  0.0,  4.5]]).float()

    def look_at(self,camera_position, target_position, up_vector=np.array([0, 0, 1])):
        forward = camera_position - target_position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up_vector, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        rotation_matrix = np.array([right, up, forward]).T

        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -camera_position

        rotation_homogeneous = np.eye(4)
        rotation_homogeneous[:3, :3] = rotation_matrix

        w2c = rotation_homogeneous @ translation_matrix
        return w2c
    
    def generate_target_pose(self, azimuths_deg, elevations_deg, radius=4.5):
        azimuths = np.deg2rad(azimuths_deg)
        elevations = np.deg2rad(elevations_deg)

        x = radius * np.cos(azimuths) * np.cos(elevations)
        y = radius * np.sin(azimuths) * np.cos(elevations)
        z = radius * np.sin(elevations)
        camera_positions = np.stack([x, y, z], axis=-1)

        target_position = np.array([0, 0, 0])  # 目标点位置

        # 为每个相机位置生成 w2c 矩阵
        w2c_matrices = [self.look_at(cam_pos, target_position) for cam_pos in camera_positions]
        w2c_matrices = np.stack(w2c_matrices, axis=0)
        return w2c_matrices
    
    def convert_to_blender(self, pose):
        # Swap the y and z axes
        w2c_opengl = pose
        w2c_opengl[[1, 2], :] = w2c_opengl[[2, 1], :]
        
        # Invert the y axis
        w2c_opengl[1] *= -1
        R = w2c_opengl[:3, :3]
        t = w2c_opengl[:3, 3]

        cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
        R = R.T
        t = -R @ t
        R_world2cv = cam_rec @ R
        t_world2cv = cam_rec @ t

        RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
        return RT

    def worldNormal2camNormal(self, rot_w2c, normal_map_world):
        H,W,_ = normal_map_world.shape
        # normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])
        normal_map_world = normal_map_world[...,:3]
        # faster version
        normal_map_flat = normal_map_world.contiguous().view(-1, 3)

        normal_map_camera_flat = torch.matmul(normal_map_flat.float(), rot_w2c.T.float())

        # Reshape the transformed normal map back to its original shape
        normal_map_camera = normal_map_camera_flat.view(normal_map_world.shape)

        return normal_map_camera

    def trans_normal(self, normal, RT_w2c, RT_w2c_target):
        """
        :param normal: (H,W,3), torch tensor, range [-1,1]
        :param RT_w2c: (4,4), torch tensor, world to camera
        :param RT_w2c_target: (4,4), torch tensor, world to camera
        :return: normal_target_cam: (H,W,3), torch tensor, range [-1,1]
        """
        relative_RT = torch.matmul(RT_w2c_target[:3,:3], torch.linalg.inv(RT_w2c[:3,:3]))
        normal_target_cam = self.worldNormal2camNormal(relative_RT[:3,:3], normal)

        return normal_target_cam

    def trans_local_2_global(self, normal_local, azimuths_deg, elevations_deg, radius=4.5, for_lotus=True):
        """
        :param normal_local: (B,H,W,3), torch tensor, range [-1,1]
        :param azimuths_deg: (B,), numpy array, range [0,360]
        :param elevations_deg: (B,), numpy array, range [-90,90]
        :param radius: float, default 4.5
        :return: global_normal: (B,H,W,3), torch tensor, range [-1,1]

        """
        # print(f"normal_local.shape:{normal_local.shape}")
        # print(f"azimuths_deg.shape:{azimuths_deg.shape}")
        # print(f"elevations_deg.shape:{elevations_deg.shape}")
        assert normal_local.shape[0] == azimuths_deg.shape[0] == elevations_deg.shape[0]
        identity_w2c = self.identity_w2c

        # generate target pose
        target_w2c = self.generate_target_pose(azimuths_deg, elevations_deg, radius)
        target_w2c = torch.from_numpy(np.stack([self.convert_to_blender(w2c) for w2c in target_w2c])).float()
        global_normal = []

        # transform normal
        for i in range(normal_local.shape[0]):
            normal_local_i = normal_local[i]
            normal_zero123 = self.trans_normal(normal_local_i, target_w2c[i], identity_w2c)
            global_normal.append(normal_zero123)

        global_normal = torch.stack(global_normal, dim=0)
        if for_lotus:
            global_normal[...,0] *= -1
        global_normal = global_normal / torch.norm(global_normal, dim=-1, keepdim=True)
        return global_normal

    def trans_global_2_local(self, normal_local, azimuths_deg, elevations_deg, radius=4.5):
        """
        :param normal_global: (B,H,W,3), torch tensor, range [-1,1]
        :param azimuths_deg: (B,), numpy array, range [0,360]
        :param elevations_deg: (B,), numpy array, range [-90,90]
        :param radius: float, default 4.5
        :return: local_normal: (B,H,W,3), torch tensor, range [-1,1]

        """
        print(f"normal_local.shape:{normal_local.shape}")
        print(f"azimuths_deg.shape:{azimuths_deg.shape}")
        print(f"elevations_deg.shape:{elevations_deg.shape}")
        assert normal_local.shape[0] == azimuths_deg.shape[0] == elevations_deg.shape[0]
        identity_w2c = self.identity_w2c

        # generate target pose
        target_w2c = self.generate_target_pose(azimuths_deg, elevations_deg, radius)
        target_w2c = torch.from_numpy(np.stack([self.convert_to_blender(w2c) for w2c in target_w2c])).float()
        local_normal = []

        # transform normal
        for i in range(normal_local.shape[0]):
            normal_local_i = normal_local[i]
            normal = self.trans_normal(normal_local_i, identity_w2c, target_w2c[i])
            local_normal.append(normal)

        local_normal = torch.stack(local_normal, dim=0)
        # global_normal[...,0] *= -1
        local_normal = local_normal / torch.norm(local_normal, dim=-1, keepdim=True)
        return local_normal