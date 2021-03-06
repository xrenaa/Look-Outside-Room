# convenience function to download pretrained vqmodels and run with provided training configs
import os
import sys
sys.path.append(".")
from src.util import get_local_path

SYMLINK_MAP = {
    "re_first_stage": "pretrained_models/realestate_first_stage/last.ckpt"
}

def create_symlink(name, path):
    print(f"Creating symlink from {path} to {SYMLINK_MAP[name]}")
    os.makedirs("/".join(SYMLINK_MAP[name].split(os.sep)[:-1]))
    os.symlink(src=path, dst=SYMLINK_MAP[name], target_is_directory=False)


if __name__ == "__main__":
    for model in SYMLINK_MAP:
        path = get_local_path(model)
        create_symlink(model, path)
    print("done.")