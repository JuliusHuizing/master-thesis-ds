import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rembg
import tqdm
from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam
from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.device = torch.device("cuda")
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.bg_remover = None
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            self.renderer.initialize(num_pts=self.opt.num_pts)

    def load_input(self, file):
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        self.input_mask = img[..., 3:]
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        self.input_img = self.input_img[..., ::-1].copy()
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for _ in tqdm.trange(iters):
                self.train_step()
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')

    # Add other necessary methods here...

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    gui = GUI(opt)
    gui.train(opt.iters)
