import os
import cv2
import time
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import argparse
from omegaconf import OmegaConf
import os

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

      

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
       
        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed
        
  

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():
            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)
                
    def elongation_regularizer(self, scaling_factors, lambda_reg=0.01):
        # Calculate the variance along the scaling dimensions
        variance = torch.var(scaling_factors, dim=1)  # Assume scaling_factors is [N, 3] where N is the number of Gaussians
         # Invert the variance to penalize low variance (which corresponds to less elongation)
        inverted_variance_penalty = 1 / (variance + 1e-6)  # Adding a small epsilon to avoid division by zero
        
        # Calculate the regularization loss as the mean of the inverted variance penalties
        reg_loss = lambda_reg * torch.mean(inverted_variance_penalty)
        return reg_loss
    
    def opacity_regularizer(self, opacity, lambda_reg=0.05):
        # Penalize values far from 0 and 1 using a logistic loss
        reg_loss = lambda_reg * torch.mean(opacity * (1 - opacity))  # Simple quadratic around 0.5
        return reg_loss
    def compactness_regularizer(self, scaling_factors, lambda_compact=0.01):
        # Sum of scaling factors for each Gaussian
        sum_scaling = torch.sum(scaling_factors, dim=1)  # Sum of scaling factors across x, y, and z
        
        # Use the sum as a penalty to encourage smaller Gaussians
        compactness_penalty = sum_scaling
        
        # Calculate the regularization loss as the mean of the sum penalties
        reg_loss = lambda_compact * torch.mean(compactness_penalty)
        return reg_loss
   
    
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)
            

            loss = 0

            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(image, self.input_img_torch)

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                loss = loss + 1000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(mask, self.input_mask_torch)
                
                ## own
                scaling_factors = self.renderer.gaussians.get_scaling  # Assuming this method exists and provides the scaling factors
                reg_loss = self.elongation_regularizer(scaling_factors, lambda_reg=0.1)
                loss += reg_loss
                
                size_loss = self.compactness_regularizer(scaling_factors, lambda_compact=0.1)
                loss += size_loss
                
                # Retrieve opacity values from GaussianModel
                opacities = self.renderer.gaussians.get_opacity  # Assuming this returns a tensor of opacity values

                # Calculate opacity regularization loss
                opacity_reg_loss = self.opacity_regularizer(opacities, lambda_reg=0.1)
                loss += opacity_reg_loss
                    
                


            ### novel view (manual batch)
            render_resolution = 256 if step_ratio < 0.3 else (512 if step_ratio < 0.6 else 1028)

            # render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                
                

           
                        
            images = torch.cat(images, dim=0)
             # Save images
             
              # Inside the train_step method
            save_dir = "image_progressions"
            os.makedirs(save_dir, exist_ok=True) 
            for i in range(images.shape[0]):
                save_path = os.path.join(save_dir, f'rendered_image_{self.step}_{i}.jpg')
                image_np = (images[i].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
                cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

        

            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()
                    
                    
                    

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True


    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()


    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")


    def render(self):
       return
   
    def visualize_gaussian_distribution(self, gaussians, image_size=512):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # Directory for saving plots
        plot_dir = "visualization_plots_with_reasonable_reg"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        scaling_factors = self.renderer.gaussians.get_scaling.detach().cpu().numpy()  # [N, 3] where N is the number of Gaussians
        opacities = self.renderer.gaussians.get_opacity.detach().cpu().numpy()  # [N, 1]
        elongation_ratios = np.max(scaling_factors, axis=1) / np.min(scaling_factors, axis=1)
    
        # Plotting Elongation Ratios
        plt.figure(figsize=(10, 5))
        plt.hist(elongation_ratios, bins=30, color='skyblue', alpha=0.7)
        plt.title('Distribution of Elongation Ratios')
        plt.xlabel('Elongation Ratio (Max scaling factor / Min scaling factor)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plot_path = os.path.join(plot_dir, 'elongation_ratios_histogram.png')
        plt.savefig(plot_path)  # Save the figure
        plt.close()  # Close the plot to free up memory
        print(f"Elongation ratios histogram saved to {plot_path}")
        
        
        # Plotting Opacity Distributions
        plt.figure(figsize=(10, 5))
        plt.hist(opacities, bins=30, color='salmon', alpha=0.7)
        plt.title('Distribution of Opacity Values')
        plt.xlabel('Opacity')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'opacity_values_histogram.png'))  # Save the figure
        plt.close()  # Close the plot
        
        
        # Assuming 'scaling_factors' is a tensor from your model
        variance = torch.var(scaling_factors, dim=1).detach().cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(variance, bins=30, alpha=0.7, color='blue')
        plt.title('Distribution of Variance among Scaling Factors')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'variance_distribution_histogram.png'))  # Save the figure
        plt.show()
    
    # no gui mode
    def train(self, iters=1000):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # visualize gaussian distribution
            self.visualize_gaussian_distribution(self.renderer.gaussians) 
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        
        # self.save_model(mode='model')
        # self.save_model(mode='geo+tex')
        
        # print(f"[INFO] saving random images!")
        # save_dir = "rendered_images"
        # os.makedirs(save_dir, exist_ok=True) 
        
            
        # Example function to generate camera positions with overlap and varied viewpoints
        def generate_camera_positions(num_positions):
            positions = []
            step_angle = 360 / num_positions
            for i in range(num_positions):
                ver = np.random.randint(-30, 30)  # Smaller range to maintain elevation consistency
                hor = step_angle * i  # Ensures complete rotation with overlap
                for radius_variation in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    positions.append((self.opt.elevation + ver, hor, self.opt.radius * radius_variation))
            return positions

        # # Main rendering loop
        # camera_positions = generate_camera_positions(200)  # Generate 200 well-planned positions
        # for idx, (ver, hor, rad) in enumerate(camera_positions):
        #     pose = orbit_camera(ver, hor, rad)
        #     cur_cam = MiniCam(pose, self.opt.ref_size, self.opt.ref_size, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
        #     out = self.renderer.render(cur_cam)
        #     image = out["image"].unsqueeze(0)

        #     image_np = image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        #     image_np = (image_np * 255).astype(np.uint8)
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(save_dir, f'rendered_image_{self.step}_{idx}.jpg'), image_np)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")

    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
   
    
    gui = GUI(opt)
    gui.train(opt.iters)



