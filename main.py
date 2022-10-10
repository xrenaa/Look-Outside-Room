import argparse, os, sys, datetime, glob, importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from omegaconf import OmegaConf
# DDP
from torch.utils import data
import torch.distributed as dist
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
    
def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)
    print(worker_id)

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

def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x+1.0)*127.5
    x = x.clip(0, 255).astype(np.uint8)

    return Image.fromarray(x)

# set args
parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--dataset", type=str, default="realestate",
                    help="dataset type")
parser.add_argument("--name", type=str, default="exp_1",
                    help="experiments name")
parser.add_argument("--base", type=str, default="./configs/realestate/realestate_16x16_sine_cview_adaptive.yaml",
                    help="experiments name")
parser.add_argument("--data-path", type=str, default="/latent_opt_test/RealEstate10K_Downloader/",
                    help="data path")
parser.add_argument("--batch-size", type=int, default=4, help="")
parser.add_argument("--ckpt-iter", type=int, default=10000,
                    help="interval for visual the result")
parser.add_argument("--visual-iter", type=int, default=500,
                    help="interval for visual the result")
parser.add_argument("--max_iter", type=int, default=200001,
                    help="epochs for training")
parser.add_argument("--len", type=int, default=3, help="")
parser.add_argument("--gap", type=int, default=3, help="") # disable in realestate
parser.add_argument('--gpu', default= '0,1,2,3', type=str)
parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# set the config
n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = n_gpu > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
    
max_iter = args.max_iter
ngpu = n_gpu
accumulate_grad_batches = 2
time_len = args.len

# first make the dir
model_dir = os.path.join("./experiments/%s" % args.dataset, args.name)
visual_dir = os.path.join("./experiments/%s" % args.dataset, args.name, "visual")
save_dir = os.path.join("./experiments/%s" % args.dataset, args.name, "model")

os.makedirs(model_dir, exist_ok = True)
os.makedirs(visual_dir, exist_ok = True)
os.makedirs(save_dir, exist_ok = True)

summary = SummaryWriter(log_dir=visual_dir)

# get config
config = OmegaConf.load(args.base)
# init model
model = instantiate_from_config(config.model)
# init optim
base_lr = config.model.base_learning_rate
bs = args.batch_size
model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

optimizer, scheduler = model.configure_optimizers()

# set to DDP
model.cuda()
model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

# load data
if args.dataset == "realestate":
    from src.data.realestate.realestate_cview import VideoDataset
    sparse_dir = "%s/sparse/" % args.data_path
    image_dir = "%s/dataset/" % args.data_path
    dataset = VideoDataset(sparse_dir = sparse_dir, image_dir = image_dir, length = time_len, low = 3, high = 20)
elif args.dataset == "mp3d":
    from src.data.mp3d.mp3d_cview import VideoDataset
    dataset = VideoDataset(root_path = args.data_path, length = time_len, gap = args.gap)
else:
    raise ValueError("the dataset must be realestate or mp3d")
    
train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True
)

# trainer
pbar = range(max_iter)

if get_rank() == 0:
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

if args.distributed:
    module = model.module
else:
    module = model

train_loader = sample_data(train_loader)
dist.barrier()

for idx in pbar:
    batch = next(train_loader)

    for key in batch.keys():
        batch[key] = batch[key].cuda()
    
    forecasts, gts, loss, log_dict = module(batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # update tensorboard
    summary.add_scalar(tag='loss', scalar_value=loss.mean().item(), global_step=idx)
    
    if get_rank() == 0:
        pbar.set_description((f"loss: {loss:.4f};"))

        if idx % args.ckpt_iter == 0:
            torch.save(module.state_dict(), os.path.join(save_dir, "last.ckpt"))

    if idx % args.visual_iter == 0:
        if get_rank() == 0:
            gt_clip = batch["rgbs"][0].permute(1,0,2,3)
            gt_clip = vutils.make_grid(gt_clip)
            gt_clip = (gt_clip + 1)/2

            # recon
            predict_1 = batch["rgbs"][0:1, :, 0].cuda()
            recon_clip = [predict_1]
            with torch.no_grad():
                for i in range(time_len - 1):
                    predict = gts[i][0]
                    predict = module.decode_to_img(predict, [1, 256, 16,16])
                    recon_clip.append(predict)

            recon_clip = vutils.make_grid(torch.cat(recon_clip, 0))
            recon_clip = (recon_clip + 1)/2

            # predict
            predict_1 = batch["rgbs"][0:1, :, 0].cuda()
            pred_clip = [predict_1]
            
            with torch.no_grad():
                for i in range(time_len - 1):
                    predict = torch.argmax(forecasts[i], 2)[0]
                    predict = module.decode_to_img(predict, [1, 256, 16,16])
                    pred_clip.append(predict)

            pred_clip = vutils.make_grid(torch.cat(pred_clip, 0))
            pred_clip = (pred_clip + 1)/2

            merge = torch.cat([gt_clip.cpu(), recon_clip.cpu(), pred_clip.cpu()], 1).clamp(0,1)
            plt.imsave(os.path.join(visual_dir, "%06d.png" % idx), merge.permute(1,2,0).numpy())

            dist.barrier()
        else:
            dist.barrier()
        

            