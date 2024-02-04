"""
Various utilities for the project.
"""
from typing import Union, Float
import logging
from math import floor

import torch
import scipy


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


def discount_cumsum(x : torch.Tensor, discount : Float) -> torch.Tensor:
    ''' Taken from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L29 '''
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def statistics(x : torch.Tensor) -> Union[Float, Float]:
    return torch.mean(x), torch.std(x)
