from typing import Tuple
import logging
from math import floor

import numpy as np
import scipy  # type: ignore
import torch


def compute_output_shape(input_shape, kernel_size, stride):
    return (
        floor((input_shape[0] - kernel_size[0]) / stride[0] + 1),
        floor((input_shape[1] - kernel_size[1]) / stride[1] + 1),
    )


def check_shape_validity(input_shape, target_shape):
    if target_shape[0] > input_shape[0] or target_shape[1] > input_shape[1]:
        raise ValueError(
            f"Input shape {input_shape} cannot be made larger to meet target shape {target_shape}"
        )


def check_shape_compatibility(input_shape, target_shape, kernel_size, stride):
    out_shape = compute_output_shape(input_shape, kernel_size, stride)
    if out_shape[0] != target_shape[0] or out_shape[1] != target_shape[1]:
        raise ValueError(
            f"Incompatible set of parameters: input_shape {input_shape}, target_shape {target_shape}, kernel_size {kernel_size}, stride {stride}"
        )


def compute_kernel_size(input_shape, target_shape, stride):
    """
    Computes what kernel size the conv2d or pooling layer should have given an input shape (H, W), a target shape (nH, nW), and the stride.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        (height, width) of the input tensor
    target_shape : Tuple[int, int]
        (height, width) of the target tensor
    stride : int or Tuple[int, int]
        Distance the kernel window travels at each iteration

    Returns
    -------
    kernel_size : Tuple[int, int]
        Kernel size to use to achieve target shape
    """
    check_shape_validity(input_shape, target_shape)
    if isinstance(stride, int):
        stride = (stride, stride)
    kernel_size = (
        input_shape[0] - stride[0] * (target_shape[0] - 1),
        input_shape[1] - stride[1] * (target_shape[1] - 1),
    )
    check_shape_compatibility(input_shape, target_shape, kernel_size, stride)
    return kernel_size


def compute_stride(input_shape, target_shape, kernel_size):
    """
    Computes what stride the conv2d or pooling layer should have given an input shape (H, W), a target shape (nH, nW), and a kernel_size.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        (height, width) of the input tensor
    target_shape : Tuple[int, int]
        (height, width) of the target tensor
    kernel_size : int or Tuple[int, int]
        Size of the kernel window

    Returns
    -------
    stride : Tuple[int, int]
        Stride to use to achieve target shape
    """
    check_shape_validity(input_shape, target_shape)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    stride = (
        (input_shape[0] - kernel_size[0]) / (target_shape[0] - 1),
        (input_shape[1] - kernel_size[1]) / (target_shape[1] - 1),
    )
    if stride[0] > kernel_size[0] or stride[1] > kernel_size[1]:
        logging.warn(
            f"Stride {stride} is larger than kernel size {kernel_size}. This means you are skipping pixels in the image."
        )
    check_shape_compatibility(input_shape, target_shape, kernel_size, stride)
    return stride


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Taken from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L29"""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def statistics(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.mean(x), torch.std(x)


def sample_multinomial(
    dist: torch.Tensor, sample_dtype=torch.long
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a sample and its log probability for a multinomial distribution"""
    sample = torch.multinomial(dist, 1)  # Shape: (batch_size, 1)
    batch_indices = torch.arange(dist.size(0)).unsqueeze(1)
    return sample.to(sample_dtype), dist[batch_indices, sample].log()


def sample_guassian(
    mean: torch.Tensor, std: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a sample and its log probability for a Guassian distribution"""
    dist = torch.distributions.Normal(mean, std)
    sample = dist.rsample()  # Use rsample() to maintain gradients
    return sample.unsqueeze(1), dist.log_prob(sample).unsqueeze(1)


def sample_action(
    action_dists: Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
) -> torch.Tensor:
    """
    Samples actions from the various distributions and combines them into an action tensor.
    Outputs the action tensor and a logp tensor showing the log probability of taking that action.

    Parameters
    ----------
    action_dists : Tuple
        List of distributions to sample from

    Returns
    -------
    torch.Tensor
        Action tensor representing the items sampled from the various distributions
    torch.Tensor
        Log probabilities of sampling the corresponding action. To get the joint log-probability of the
        action, you can `.sum()` this tensor.
    """
    assert len(action_dists[0].shape) == 2
    # Initialize action and log buffer
    batch_size = action_dists[0].size(0)
    batch_indices = torch.arange(batch_size).unsqueeze(1)
    action = torch.zeros((batch_size, 10), dtype=torch.float)
    logp_action = torch.zeros((batch_size, 10), dtype=torch.float)

    action[batch_indices, 0], _ = sample_multinomial(
        action_dists[0][:], torch.float
    )
    action[batch_indices, 1], _ = sample_multinomial(
        action_dists[1][:], torch.float
    )
    action[batch_indices, 2], _ = sample_multinomial(
        action_dists[2][:], torch.float
    )
    action[batch_indices, 3], _ = sample_multinomial(
        action_dists[3][:], torch.float
    )
    action[batch_indices, 4], _ = sample_multinomial(
        action_dists[4][:], torch.float
    )
    action[batch_indices, 5], _ = sample_multinomial(
        action_dists[5][:], torch.float
    )
    action[batch_indices, 6], _ = sample_multinomial(
        action_dists[6][:], torch.float
    )
    action[batch_indices, 7], _ = sample_multinomial(
        action_dists[7][:], torch.float
    )
    action[batch_indices, 8], _ = sample_guassian(
        action_dists[8][:, 0], action_dists[9][:, 0]
    )
    action[batch_indices, 9], _ = sample_guassian(
        action_dists[8][:, 1], action_dists[9][:, 1]
    )

    return action


def joint_logp_action(
    action_dists: Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    actions_taken: torch.Tensor,
) -> torch.Tensor:
    """
    Outputs the log probability of a sample as if the sample was taken
    from the distribution already.

    Parameters
    ----------
    action_dists : Tuple
        List of distributions to sample from
    actions_taken : torch.Tensor
        Samples produced already

    Returns
    -------
    torch.Tensor
        Join log-probability of taking the action given the distributions of each component
    """
    long_actions_taken = actions_taken.long()
    joint_logp = (  # longitudinal movement
        action_dists[0]
        .gather(1, long_actions_taken[:, 0].unsqueeze(-1))
        .squeeze()
        .log()
    )
    # Avoid += here as it is an in-place operation (which is bad for autograd)
    joint_logp = joint_logp + (  # lateral movement
        action_dists[1]
        .gather(1, long_actions_taken[:, 1].unsqueeze(-1))
        .squeeze()
        .log()
    )
    joint_logp = joint_logp + (  # vertical movement
        action_dists[2]
        .gather(1, long_actions_taken[:, 2].unsqueeze(-1))
        .squeeze()
        .log()
    )
    joint_logp = joint_logp + (  # pitch movement
        action_dists[3]
        .gather(1, long_actions_taken[:, 3].unsqueeze(-1))
        .squeeze()
        .log()
    )
    joint_logp = joint_logp + (  # yaw movement
        action_dists[4]
        .gather(1, long_actions_taken[:, 4].unsqueeze(-1))
        .squeeze()
        .log()
    )
    joint_logp = joint_logp + (  # functional actions
        action_dists[5]
        .gather(1, long_actions_taken[:, 5].unsqueeze(-1))
        .squeeze()
        .log()
    )
    joint_logp = joint_logp + (  # crafting actions
        action_dists[6]
        .gather(1, long_actions_taken[:, 6].unsqueeze(-1))
        .squeeze()
        .log()
    )
    joint_logp = joint_logp + (  # inventory actions
        action_dists[7]
        .gather(1, long_actions_taken[:, 7].unsqueeze(-1))
        .squeeze()
        .log()
    )
    # Focus actions
    x_roi_dist = torch.distributions.Normal(
        action_dists[8][:, 0], action_dists[9][:, 0]
    )
    joint_logp = joint_logp + x_roi_dist.log_prob(actions_taken[:, 8])
    y_roi_dist = torch.distributions.Normal(
        action_dists[8][:, 1], action_dists[9][:, 1]
    )
    joint_logp = joint_logp + y_roi_dist.log_prob(actions_taken[:, 9])

    return joint_logp
