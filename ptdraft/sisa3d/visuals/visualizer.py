import os
import numpy as np
import matplotlib.pyplot as plt
import torch

class Visualizer:
    """
    A class to handle visualization of Gaussian distribution parameters such as scaling factors,
    opacities, and elongation ratios of Gaussians.
    """

    @staticmethod
    def ensure_dir_exists(directory):
        """
        Ensures that the specified directory exists.

        Parameters:
            directory (str): The directory path to ensure existence.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def visualize_gaussian_distribution(gaussians, plot_dir, image_size=512):
        """
        Visualizes and saves the distribution plots for the scaling factors, opacities, and 
        elongation ratios of Gaussians into the specified directory.

        Parameters:
            gaussians: An object that must have the following methods:
                - get_scaling: Must return a tensor of scaling factors with dimensions [N, 3],
                  where N is the number of Gaussians and 3 corresponds to the scaling factors along
                  each axis (x, y, z).
                - get_opacity: Must return a tensor of opacity values with dimensions [N, 1],
                  where N is the number of Gaussians.
            plot_dir (str): The directory where plots will be saved.
            image_size (int, optional): The size of the images for the plots (currently unused in plotting).
        """
        Visualizer.ensure_dir_exists(plot_dir)

        scaling_factors = gaussians.get_scaling.detach().cpu().numpy()  # [N, 3] where N is the number of Gaussians
        opacities = gaussians.get_opacity.detach().cpu().numpy()  # [N, 1]
        elongation_ratios = np.max(scaling_factors, axis=1) / np.min(scaling_factors, axis=1)

        # Plotting Elongation Ratios
        plt.figure(figsize=(10, 5))
        plt.hist(elongation_ratios, bins=30, color='skyblue', alpha=0.7)
        plt.title('Distribution of Elongation Ratios')
        plt.xlabel('Elongation Ratio (Max scaling factor / Min scaling factor)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'elongation_ratios_histogram.png'))
        plt.close()

        # Plotting Opacity Distributions
        plt.figure(figsize=(10, 5))
        plt.hist(opacities, bins=30, color='salmon', alpha=0.7)
        plt.title('Distribution of Opacity Values')
        plt.xlabel('Opacity')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'opacity_values_histogram.png'))
        plt.close()

        # Plotting Variance Distribution among Scaling Factors
        variance = torch.var(torch.tensor(scaling_factors), dim=1).numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(variance, bins=30, alpha=0.7, color='blue')
        plt.title('Distribution of Variance among Scaling Factors')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'variance_distribution_histogram.png'))
        plt.close()
