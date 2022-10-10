import random
import os
import argparse

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys, datetime, glob, importlib
sys.path.insert(0, ".")

import torch
import torchvision

from omegaconf import OmegaConf

# args
parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--base", type=str, default="mp3d_16x16_adaptive",
                    help="experiments name")
parser.add_argument("--exp", type=str, default="exp_1_error",
                    help="experiments name")
parser.add_argument("--ckpt", type=str, default="last",
                    help="checkpoint name")
parser.add_argument("--data-path", type=str, default="/MP3D/test/",
                    help="data path")
parser.add_argument("--len", type=int, default=4, help="len of prediction")
parser.add_argument('--gpu', default= '0', type=str)
parser.add_argument("--video_limit", type=int, default=20, help="# of video to test")
parser.add_argument("--gap", type=int, default=3, help="")
parser.add_argument("--seed", type=int, default=2333, help="")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# fix the seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# config
config_path = "./configs/mp3d/%s.yaml" % args.base
cpt_path = "./experiments/mp3d/%s/model/%s.ckpt" % (args.exp, args.ckpt)

video_limit = args.video_limit
frame_limit = args.len

target_save_path = "./experiments/mp3d/%s/evaluate_frame_%d_video_%d_gap_%d/" % (args.exp, frame_limit, video_limit, args.gap)
os.makedirs(target_save_path, exist_ok=True)

# metircs
from src.metric.metrics import perceptual_sim, psnr, ssim_metric
from src.metric.pretrained_networks import PNet

vgg16 = PNet().to("cuda")
vgg16.eval()
vgg16.cuda()

# load model
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# define relative pose
def compute_camera_pose(R_dst, t_dst, R_src, t_src):
    # first compute R_src_inv
    R_src_inv = R_src.transpose(-1,-2)
    
    R_rel = R_dst@R_src_inv
    t_rel = t_dst-R_rel@t_src
    
    return R_rel, t_rel

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
model.cuda()
model.load_state_dict(torch.load(cpt_path))
model.eval()

# load dataloader
from src.data.mp3d.mp3d_abs import VideoDataset
dataset_abs = VideoDataset(root_path = args.data_path, length = args.len, gap = args.gap)
test_loader_abs = torch.utils.data.DataLoader(
        dataset_abs,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last = True
    )

def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x+1.0)*127.5
    x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def evaluate_per_batch(temp_model, batch, total_time_len = 20, time_len = 1, show = False):
    video_clips = []
    video_clips.append(batch["rgbs"][:, :, 0, ...])

    # first generate one frame
    with torch.no_grad():
        conditions = []
        R_src = batch["R_s"][0, 0, ...]
        R_src_inv = R_src.transpose(-1,-2)
        t_src = batch["t_s"][0, 0, ...]

        # create dict
        example = dict()
        example["K"] = batch["K"]
        example["K_inv"] = batch["K_inv"]

        example["src_img"] = video_clips[-1]
        _, c_indices = temp_model.encode_to_c(example["src_img"])
        c_emb = temp_model.transformer.tok_emb(c_indices)
        conditions.append(c_emb)

        R_dst = batch["R_s"][0, 1, ...]
        t_dst = batch["t_s"][0, 1, ...]

        R_rel = R_dst@R_src_inv
        t_rel = t_dst-R_rel@t_src

        example["R_rel"] = R_rel.unsqueeze(0)
        example["t_rel"] = t_rel.unsqueeze(0)

        embeddings_warp = temp_model.encode_to_e(example)
        conditions.append(embeddings_warp)
        # p1
        p1 = temp_model.encode_to_p(example)

        prototype = torch.cat(conditions, 1)
        z_start_indices = c_indices[:, :0]
        index_sample = temp_model.sample_latent(z_start_indices, prototype, [p1, None, None],
                                       steps=c_indices.shape[1],
                                       temperature=1.0,
                                       sample=False,
                                       top_k=100,
                                       callback=lambda k: None,
                                       embeddings=None)

        sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
        video_clips.append(sample_dec) # update video_clips list
        current_im = as_png(sample_dec.permute(0,2,3,1)[0])

        if show:
            plt.imshow(current_im)
            plt.show()

    # then generate second
    with torch.no_grad():
        for i in range(0, total_time_len-2, time_len):
            conditions = []

            R_src = batch["R_s"][0, i, ...]
            R_src_inv = R_src.transpose(-1,-2)
            t_src = batch["t_s"][0, i, ...]

            # create dict
            example = dict()
            example["K"] = batch["K"]
            example["K_inv"] = batch["K_inv"]

            for t in range(time_len):
                example["src_img"] = video_clips[-2]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i+t+1, ...]
                t_dst = batch["t_s"][0, i+t+1, ...]

                R_rel = R_dst@R_src_inv
                t_rel = t_dst-R_rel@t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p1
                p1 = temp_model.encode_to_p(example)

                example["src_img"] = video_clips[-1]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i+t+2, ...]
                t_dst = batch["t_s"][0, i+t+2, ...]

                R_rel = R_dst@R_src_inv
                t_rel = t_dst-R_rel@t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p2
                p2 = temp_model.encode_to_p(example)
                # p3
                R_rel, t_rel = compute_camera_pose(batch["R_s"][0, i+t+2, ...], batch["t_s"][0, i+t+2, ...], 
                                                   batch["R_s"][0, i+t+1, ...], batch["t_s"][0, i+t+1, ...])
                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                p3 = temp_model.encode_to_p(example)

                prototype = torch.cat(conditions, 1)

                z_start_indices = c_indices[:, :0]
                index_sample = temp_model.sample_latent(z_start_indices, prototype, [p1, p2, p3],
                                               steps=c_indices.shape[1],
                                               temperature=1.0,
                                               sample=False,
                                               top_k=100,
                                               callback=lambda k: None,
                                               embeddings=None)

                sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
                current_im = as_png(sample_dec.permute(0,2,3,1)[0])
                video_clips.append(sample_dec) # update video_clips list

                if show:
                    plt.imshow(current_im)
                    plt.show()
                
    return video_clips

# first save the frame and then evaluate the saved frame 
n_values_percsim = []
n_values_ssim = []
n_values_psnr = []

pbar = tqdm(total=video_limit)
b_i = 0    

while b_i < video_limit:
    
    try:
        batch = next(iter(test_loader_abs))
    except:
        continue
        
    pbar.update(1)
        
    values_percsim = []
    values_ssim = []
    values_psnr = []
        
    sub_dir = os.path.join(target_save_path, "%03d" % b_i)
    os.makedirs(sub_dir, exist_ok=True)

    for key in batch.keys():
        batch[key] = batch[key].cuda()
    
    generate_video = evaluate_per_batch(model, batch, total_time_len = frame_limit, time_len = 1)
        
    for i in range(1, len(generate_video)):
        gt_img = np.array(as_png(batch["rgbs"][0, :, i, ...].permute(1,2,0)))
        forecast_img = np.array(as_png(generate_video[i][0].permute(1,2,0)))
        
        cv2.imwrite(os.path.join(sub_dir, "predict_%02d.png" % i), forecast_img[:, :, [2,1,0]])
        cv2.imwrite(os.path.join(sub_dir, "gt_%02d.png" % i), gt_img[:, :, [2,1,0]])
        
        t_img = (batch["rgbs"][:, :, i, ...] + 1)/2
        p_img = (generate_video[i] + 1)/2
        t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()

        perc_sim = t_perc_sim
        ssim_sim = ssim_metric(p_img, t_img).item()
        psnr_sim = psnr(p_img, t_img).item()
        
        values_percsim.append(perc_sim)
        values_ssim.append(ssim_sim)
        values_psnr.append(psnr_sim)
    
    n_values_percsim.append(values_percsim)
    n_values_ssim.append(values_ssim)
    n_values_psnr.append(values_psnr)
    
    b_i += 1
    
pbar.close()
    
total_percsim = []
total_ssim = []
total_psnr = []

with open(os.path.join(target_save_path, "eval.txt"), 'w') as f:
    for i in range(len(n_values_percsim)):

        f.write("#%d, percsim: %.04f, ssim: %.02f, psnr: %.02f" % (i, np.mean(n_values_percsim[i]), 
                                                                   np.mean(n_values_ssim[i]), np.mean(n_values_psnr[i])))
        f.write('\n')
        
        for j in range(len(n_values_percsim[i])):
            percsim = n_values_percsim[i][j]
            ssim = n_values_ssim[i][j]
            psnr = n_values_psnr[i][j]

            total_percsim.append(percsim)
            total_ssim.append(ssim)
            total_psnr.append(psnr)
            
    f.write("Total, percsim: (%.03f, %.02f), ssim: (%.02f, %.02f), psnr: (%.03f, %.02f)" 
            % (np.mean(total_percsim), np.std(total_percsim), 
               np.mean(total_ssim), np.std(total_ssim),
               np.mean(total_psnr), np.std(total_psnr)))
    f.write('\n')