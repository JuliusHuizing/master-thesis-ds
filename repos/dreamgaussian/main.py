import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from cam_utils import orbit_camera
from gs_renderer import Renderer, MiniCam
from rembg import remove as rembg_remove
import sys

import argparse
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensures logs are output to standard output
    ]
)
class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info("Initializing the training environment")
        
        # Load and prepare image
        self.input_img, self.input_mask = self.load_input(opt.input)
        
        # Renderer setup
        self.renderer = Renderer(sh_degree=opt.sh_degree).to(self.device)
        if opt.load:
            logging.info(f"Loading model from {opt.load}")
            self.renderer.initialize(opt.load)
        else:
            logging.info("Initializing renderer with new parameters")
            self.renderer.initialize(num_pts=opt.num_pts)

    def load_input(self, file_path):
        logging.info(f'Loading image from {file_path}...')
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.opt.W, self.opt.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        
        if img.shape[-1] == 4:
            mask = img[..., 3:4]
            img = img[..., :3] * mask + (1 - mask)
        else:
            mask = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        
        img = img[..., ::-1]  # BGR to RGB
        return img, mask

    def train(self, iters=500):
        logging.info(f"Starting training for {iters} iterations")
        for _ in tqdm(range(iters), desc="Training"):
            self.train_step()

    def train_step(self):
        logging.debug("Starting a new training step")
        self.renderer.optimizer.zero_grad()
        loss = self.compute_loss()
        logging.debug(f"Computed loss: {loss.item()}")
        loss.backward()
        self.renderer.optimizer.step()

    def compute_loss(self):
        # Define loss computation from current renderer state and input image
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        cam = MiniCam(pose, self.opt.W, self.opt.H, self.opt.fovy, self.opt.fovx, self.opt.near, self.opt.far).to(self.device)
        
        output = self.renderer.render(cam)
        image_loss = F.mse_loss(output['image'], torch.from_numpy(self.input_img).to(self.device))
        mask_loss = F.mse_loss(output['alpha'], torch.from_numpy(self.input_mask).to(self.device))
        
        total_loss = image_loss + mask_loss
        logging.debug(f"Image loss: {image_loss.item()}, Mask loss: {mask_loss.item()}")
        return total_loss

    def save_model(self):
        model_path = os.path.join(self.opt.outdir, f"{self.opt.save_path}_model.ply")
        logging.info(f"Saving model to {model_path}")
        self.renderer.save_ply(model_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model from an image to 3D")
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml config file")
    parser.add_argument("--input", type=str, required=True, help="Input image file path")
    parser.add_argument("--save_path", type=str, required=True, help="Base path for saving output files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    opt = OmegaConf.load(args.config)
    trainer = Trainer(opt)
    trainer.train(opt.iters)
    trainer.save_model()
