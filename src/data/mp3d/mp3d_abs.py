import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import PIL
import cv2
import random
import glob
import pickle
import quaternion

import torchvision.transforms as tfs
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from src.data.realestate.realestate_cview import ToTensorVideo, NormalizeVideo

def resize(clip, target_size, interpolation_mode = "bilinear"):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )

# set global parameter
# init for 256 * 256
K = np.array([[128.0, 0.0, 127.0],
              [0.0, 128.0, 127.0],
              [0.0, 0.0, 1.0]], dtype=np.float32)

K_inv = np.linalg.inv(K)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_path = "/MP3D/train", image_size = 256, length = 3, gap = 3, is_validation = False):
        super(VideoDataset, self).__init__()
        
        self.gap = gap
        self.image_size = image_size
        self.length = length
        
        self.clip_length = self.length + (self.length - 1) * (self.gap - 1)
        
        clip_paths = []

        scene_paths = glob.glob(os.path.join(root_path, "*"))
        
        print("----------------Loading the MP3D dataset----------------")
        for scene_path in tqdm(scene_paths):
            clip_list = sorted(glob.glob(os.path.join(scene_path, "*")))

            for clip_path in clip_list:
                file = open(clip_path, 'rb')
                video_clip = pickle.load(file)
                file.close()
                
                if video_clip['rgb'].shape[0] >= self.clip_length:
                    # filter the data which is too short
                    clip_paths.append(clip_path)
                
        print("----------------Finish loading MP3D dataset----------------")

        self.clip_paths = clip_paths
        
        self.size = len(self.clip_paths) # num of scene
        
        self.is_validation = is_validation
        self.transform = transforms.Compose(
        [
            ToTensorVideo(),
            NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

    def __getitem__(self, index):
        file = open(self.clip_paths[index], 'rb')
        video_clip = pickle.load(file)
        file.close()
        
        # first sample a rgb_clip
        rgb_clip = video_clip['rgb']
        action_clip = video_clip['action']
        pos_clip = video_clip['pos']
        rot_clip = video_clip['rot']
        
        # then sample a inter-index
        inter_index = random.randint(0, rgb_clip.shape[0] - self.clip_length)
        
        rgbs = []
        R_s = [] # the relative are all based on the init frame
        t_s = []

        for img_idx in range(inter_index, (inter_index+self.clip_length), self.gap):
            rgbs.append(rgb_clip[img_idx])
            R_dst = quaternion.as_rotation_matrix(quaternion.from_float_array(rot_clip[img_idx]))
            t_dst = pos_clip[img_idx]
            
            R_s.append(R_dst)
            t_s.append(t_dst)

        rgbs = torch.from_numpy(np.stack(rgbs))
        rgbs = self.transform(rgbs)
        rgbs = resize(rgbs, (self.image_size, self.image_size))

        example = {
            "rgbs": rgbs,
            "src_points": np.zeros((1,3), dtype=np.float32),
            "K": K,
            "K_inv": K_inv,
            "R_s": np.stack(R_s).astype(np.float32),
            "t_s": np.stack(t_s).astype(np.float32)
        }

        return example

    def __len__(self):
        return self.size

    def name(self):
        return 'VideoDataset'