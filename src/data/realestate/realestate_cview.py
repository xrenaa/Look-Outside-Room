# naive dataloader for real estate dataset
# this dataloader is used to build a canonical view
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

from src.data.read_write_model import read_model
from src.data import _functional_video as F

class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return F.to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__
    
class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"
    

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_dir = "/RealEstate10K_Downloader/sparse/", image_dir = "/RealEstate10K_Downloader/dataset/", 
                 size = [256,256], length = 3, low = 3, high = 20, split = "train"):
        super(VideoDataset, self).__init__()
        # config
        self.size = size
        self.length = length
        self.low = low
        self.high = high
        
        self.clip_length = self.length + (self.length - 1) * (self.high - 1)
        # path
        self.sequence_dir = os.path.join(sparse_dir, split)
        self.image_dir = os.path.join(image_dir, split)
        
        # total scene
        scene_paths = sorted(glob.glob(os.path.join(self.sequence_dir, "*")))
        
        clip_paths = []
        
        print("----------------Loading the Real Estate dataset----------------")
        for scene_path in tqdm(scene_paths):
            seq = scene_path.split("/")[-1]
                
            im_root = os.path.join(self.image_dir, seq)

            frames = [fname for fname in os.listdir(im_root) if fname.endswith(".png")]
            n = len(frames)
            
            if n > self.clip_length:
                clip_paths.append(scene_path)
                
        print("----------------Finish loading Real Estate dataset----------------")
        
        self.clip_paths = clip_paths
        self.total_length = len(self.clip_paths) # num of scene
        
        self.transform = transforms.Compose(
        [
            ToTensorVideo(),
            NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        
    def __getitem__(self, index):
        seq = self.clip_paths[index].split("/")[-1]
        
        root = os.path.join(self.sequence_dir, seq)
        im_root = os.path.join(self.image_dir, seq)
        
        frames = sorted([fname for fname in os.listdir(im_root) if fname.endswith(".png")])
        n = len(frames)
        
        # load sparse model
        model = os.path.join(root, "sparse")
        try:
            cameras, images, points3D = read_model(path=model, ext=".bin")
        except Exception as e:
            raise Exception(f"Failed to load sparse model {model}.") from e
        
        # load camera parameters and image size
        cam = cameras[1]
        h = cam.height
        w = cam.width
        params = cam.params
        K = np.array([[params[0], 0.0, params[2]],
                      [0.0, params[1], params[3]],
                      [0.0,       0.0,       1.0]])
        
        # then sample a inter-index
        inter_index = random.randint(0, n - self.clip_length)
        
        rgbs = []
        R_rels = [] # the relative are all based on the init frame
        t_rels = []
        
        R_0 = None
        t_0 = None
        
        R_1 = None
        t_1 = None
        
        R_2 = None
        t_2 = None
        
        img_idx = inter_index
        previous_key = None
        
        for idx in range(self.length):
            if idx != 0:
                gap = random.randint(self.low, self.high)
                # print(gap)
            else:
                gap = 0
            
            img_idx += gap
            # get the name
            name = frames[img_idx]
            # get the key of desired name
            key = [k for k in images.keys() if images[k].name==name]
            
            # hand-craft design!
            if len(key) == 0:
                if previous_key is not None:
                    key = previous_key
                else:
                    key = [list(images.keys())[0]]
            else:
                previous_key = key
                
            key_dst = key[0]
            
            R_dst = images[key_dst].qvec2rotmat()
            t_dst = images[key_dst].tvec
            
            if idx == 0:
                K_inv = np.linalg.inv(K)
                R_0 = R_dst
                t_0 = t_dst
            elif idx == 1:
                R_1 = R_dst
                t_1 = t_dst
            elif idx == 2:
                R_2 = R_dst
                t_2 = t_dst

            # get image
            im_path = os.path.join(im_root, images[key_dst].name)
            im = Image.open(im_path).resize((self.size[1],self.size[0]), resample=Image.LANCZOS)
            # im = np.array(im)/127.5-1.0
            im = np.array(im)
            rgbs.append(im)
          
        # compute 
        R_0_inv = R_0.transpose(-1,-2)
        R_01 = R_1@R_0_inv
        t_01 = t_1-R_01@t_0

        R_02 = R_2@R_0_inv
        t_02 = t_2-R_02@t_0
        
        R_1_inv = R_1.transpose(-1,-2)
        R_12 = R_2@R_1_inv
        t_12 = t_2-R_12@t_1
        
        # post-process using size
        if self.size is not None and (self.size[0] != h or self.size[1] != w):
            K[0,:] = K[0,:]*self.size[1]/w
            K[1,:] = K[1,:]*self.size[0]/h
        
        rgbs = torch.from_numpy(np.stack(rgbs))
        rgbs = self.transform(rgbs)
        
        example = {
            "rgbs": rgbs,
            "src_points": np.zeros((1,3), dtype=np.float32),
            "K": K.astype(np.float32),
            "K_inv": K_inv.astype(np.float32),
            "R_01": R_01.astype(np.float32),
            "t_01": t_01.astype(np.float32),
            "R_02": R_02.astype(np.float32),
            "t_02": t_02.astype(np.float32),
            "R_12": R_12.astype(np.float32),
            "t_12": t_12.astype(np.float32)
        }

        return example
        
    def __len__(self):
        return self.total_length

    def name(self):
        return 'VideoDataset'