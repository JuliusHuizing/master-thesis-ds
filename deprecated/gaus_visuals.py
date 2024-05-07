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