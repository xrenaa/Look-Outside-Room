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
from src.data.realestate.realestate_cview import ToTensorVideo, NormalizeVideo

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_dir = "/RealEstate10K_Downloader/sparse/", image_dir = "/RealEstate10K_Downloader/dataset/", 
                 size = [256,256], length = 3, low = 3, high = 20, split = "test"):
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
        error = ["31ee8cabc96b9a62","31fb7a80b21fc505","320c8fa9d4c8ecef","321911ae4a16f038","321b9f345813cbea",
                 "322261824c4a3003","32294ad73efca3db","322d03d487fc0f01","3232f7457b27dbbc","323f971c7581f56e",
                 "325ff82707386438","3260a42ccfba8973","32615afea87b52dd","3280ab6917df576f","32895c8add3bc2b4",
                 "3289ad7a811d2348","3290731e5f908b92","329177dabfe2951d","32991f419c96ea0e","329aba411f341398",
                 "329abf340d23b0b9","32a233e3093f09ef","32a2c04fd8321bbb","32b5f8207d08653c","32cccd10c84b4529",
                 "32ce9b303717a29d","32d2163aa65c0e8a","32d28b7513f873be","32db375ab51d77a4","32dc25ef78b564a1",
                 "32dfef9109202812","32e0b9e54e7dc9eb","32eba3e4cfb61f93","32f926c1cb9ce67d","330b925cef643b3f",
                 "3317c40fd3e0a7b7","332294213fa15c56","33288d55dde83e72","333bf1aebc3963b1","402495688372685c",
                 "43c71f2c4e438e48","573c702580c77fb0","67d2d83b0a571e50","81efce32d8a173df","97f84416b29317d3",
                 "9f11dc868fa242d4","a10ef42e7af13974","c1e6ac4f872b9cf8","c1f2a61d61bd7e48","c203bfdc3d2715f9",
                 "c211e68f8e67eccf","c2153789693cbe62","c21df1e3085cbac8","c2347e4668c6b481","c23b29fd5039577d", 
                 "c6f8fddaf7782828"]
        scene_paths = sorted(glob.glob(os.path.join(self.sequence_dir, "*")))[:100]
        clip_paths = []
        
        print("----------------Loading the Real Estate dataset----------------")
        for scene_path in tqdm(scene_paths):
            seq = scene_path.split("/")[-1]
            if seq in error:
                continue
                
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
        
    def __getitem__(self, index, inter_index):
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
        
        rgbs = []
        R_s = []
        t_s = []
        
        R_canon = None
        t_canon = None
        
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
            
            if idx == 0:
                K_inv = np.linalg.inv(K)
            
            R_dst = images[key_dst].qvec2rotmat()
            t_dst = images[key_dst].tvec
            
            R_s.append(R_dst)
            t_s.append(t_dst)

            # get image
            im_path = os.path.join(im_root, images[key_dst].name)
            im = Image.open(im_path).resize((self.size[1],self.size[0]), resample=Image.LANCZOS)
            im = np.array(im)
            rgbs.append(im)
            
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
            "R_s": np.stack(R_s).astype(np.float32),
            "t_s": np.stack(t_s).astype(np.float32)
        }

        return example
        
    def __len__(self):
        return self.total_length

    def name(self):
        return 'VideoDataset'