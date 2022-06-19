import hashlib
import requests
from tqdm import tqdm
import torch
import os
from omegaconf import OmegaConf

import os
import ffmpeg
import numpy as np

def ffmpeg_video_write(data, video_path, fps=1):
    """Video writer based on FFMPEG.
    Args:
    data: A `np.array` with the shape of [seq_len, height, width, 3]
    video_path: A video file.
    fps: Use specific fps for video writing. (optional)
    """
    assert len(data.shape) == 4, f'input shape is not valid! Got {data.shape}!'
    _, height, width, _ = data.shape
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = (
        ffmpeg
        .input('pipe:', framerate=fps, format='rawvideo',
             pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(video_path, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in data:
        writer.stdin.write(frame.astype(np.uint8).tobytes())
    writer.stdin.close()


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1",
    "re_first_stage": "https://heibox.uni-heidelberg.de/f/6db3e2beebe34c1e8d06/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "geofree/lpips/vgg.pth",
    "re_first_stage": "geofree/re_first_stage/last.ckpt",
}
CACHE = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
CKPT_MAP = dict((k, os.path.join(CACHE, CKPT_MAP[k])) for k in CKPT_MAP)

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a",
    "re_first_stage": "b8b999aba6b618757329c1f114e9f5a5"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_local_path(name, root=None, check=False):
    path = CKPT_MAP[name]
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        assert name in URL_MAP, name
        print("Downloading {} from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path
