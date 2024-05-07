import torch

def elongation_regularizer(scaling_factors, lambda_reg=0.01):
    """
    Computes a regularization loss that penalizes low variance in scaling factors,
    promoting more elongation in Gaussian distributions.

    Args:
        scaling_factors (torch.Tensor): A tensor of shape [N, 3], where N is the number of Gaussians
            and each row contains scaling factors along x, y, and z axes.
        lambda_reg (float): Regularization strength.

    Returns:
        torch.Tensor: The computed regularization loss.
    """
    variance = torch.var(scaling_factors, dim=1)  # Calculate the variance along the scaling dimensions
    inverted_variance_penalty = 1 / (variance + 1e-6)  # Invert the variance to penalize low variance (avoid division by zero)
    reg_loss = lambda_reg * torch.mean(inverted_variance_penalty)  # Mean of inverted variance penalties as loss
    return reg_loss

def opacity_regularizer(opacity, lambda_reg=0.05):
    """
    Computes a regularization loss for opacity values to penalize values that are far from 0 and 1,
    effectively encouraging values to be close to 0 or 1.

    Args:
        opacity (torch.Tensor): A tensor containing opacity values of the Gaussians.
        lambda_reg (float): Regularization strength.

    Returns:
        torch.Tensor: The computed regularization loss.
    """
    reg_loss = lambda_reg * torch.mean(opacity * (1 - opacity))  # Logistic loss, simple quadratic around 0.5
    return reg_loss

def compactness_regularizer(scaling_factors, lambda_compact=0.01):
    """
    Computes a regularization loss to encourage smaller Gaussians by penalizing the sum
    of scaling factors.

    Args:
        scaling_factors (torch.Tensor): A tensor of shape [N, 3], where N is the number of Gaussians
            and each row contains scaling factors along x, y, and z axes.
        lambda_compact (float): Compactness regularization strength.

    Returns:
        torch.Tensor: The computed regularization loss.
    """
    sum_scaling = torch.sum(scaling_factors, dim=1)  # Sum of scaling factors across x, y, and z
    reg_loss = lambda_compact * torch.mean(sum_scaling)  # Mean of sum penalties as loss
    return reg_loss
