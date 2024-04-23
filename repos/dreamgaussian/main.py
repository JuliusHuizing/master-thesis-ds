
import os
import cv2
import time
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import rembg

import argparse
import os
from omegaconf import OmegaConf


from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize


class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda")
        self.renderer = Renderer(opt, self.device)  # Assume renderer initialization with options and device
        self.bg_remover = None
        self.input_img = None
        self.input_mask = None

    def load_input(self, file):
        print(f'[INFO] Loading image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 4:  # Check for alpha channel
            if self.bg_remover is None:
                self.bg_remover = rembg_new_session()
            img = rembg_remove(img, session=self.bg_remover)
        img = cv2.resize(img, (self.opt.W, self.opt.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        self.input_img = torch.from_numpy(img[..., :3][..., ::-1]).permute(2, 0, 1).unsqueeze(0).to(self.device)  # BGR to RGB
        self.input_mask = torch.from_numpy(img[..., 3:]).to(self.device)

    def train(self, iters):
        for _ in tqdm(range(iters), desc="Training"):
            self.train_step()

    def train_step(self):
        self.renderer.optimizer.zero_grad()
        loss = 0

        # Render the image using the current state of the renderer
        output = self.renderer.render()
        image_loss = F.mse_loss(output["image"], self.input_img)
        mask_loss = F.mse_loss(output["alpha"], self.input_mask)
        total_loss = image_loss + mask_loss

        # Example loss components, may need additional losses as per original requirements
        loss += total_loss
        loss.backward()
        self.renderer.optimizer.step()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output_image = self.input_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        cv2.imwrite(path, output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args = parser.parse_args()

    # Load configuration
    opt = OmegaConf.load(args.config)

    gui = GUI(opt)
    if opt.input:
        gui.load_input(opt.input)

    if hasattr(opt, 'iters'):
        gui.train(opt.iters)  # Make sure 'iters' is specified in your YAML config

    if opt.save_path:
        gui.save_model(opt.save_path)
