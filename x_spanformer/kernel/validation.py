#!/usr/bin/env python3
"""
validation.py

Kernel parameter validation utilities for X-Spanformer neural network components.
Provides centralized validation logic for convolution parameters following
Section 3.2.3 specifications.
"""
from typing import List


def validate_convolution_kernels(kernels: List[int]) -> None:
    """
    Validate kernel sizes for multi-scale convolution following Section 3.2.3.
    
    Args:
        kernels: List of kernel sizes K for multi-scale convolution
        
    Raises:
        ValueError: If kernels are invalid
        
    Requirements:
        - Must be a non-empty list
        - All values must be positive odd integers
        - Typical values: [3, 5, 7] for hierarchical pattern capture
    """
    if kernels is None:
        raise ValueError(
            "kernels parameter is required - must specify kernel sizes K for multi-scale convolution"
        )
    
    if not isinstance(kernels, list) or len(kernels) == 0:
        raise ValueError(
            "kernels must be a non-empty list of positive odd integers"
        )
    
    if not all(isinstance(k, int) and k > 0 and k % 2 == 1 for k in kernels):
        raise ValueError(
            "All kernel sizes must be positive odd integers (e.g., [3, 5, 7])"
        )


def validate_convolution_dilations(dilations: List[int]) -> None:
    """
    Validate dilation rates for multi-scale convolution following Section 3.2.3.
    
    Args:
        dilations: List of dilation rates D for receptive field patterns
        
    Raises:
        ValueError: If dilations are invalid
        
    Requirements:
        - Must be a non-empty list
        - All values must be positive integers
        - Typical values: [1, 2, 4] for exponential receptive field growth
    """
    if dilations is None:
        raise ValueError(
            "dilations parameter is required - must specify dilation rates D for receptive field patterns"
        )
    
    if not isinstance(dilations, list) or len(dilations) == 0:
        raise ValueError(
            "dilations must be a non-empty list of positive integers"
        )
    
    if not all(isinstance(d, int) and d > 0 for d in dilations):
        raise ValueError(
            "All dilation rates must be positive integers (e.g., [1, 2, 4])"
        )


def validate_convolution_parameters(kernels: List[int], dilations: List[int]) -> None:
    """
    Validate both kernel sizes and dilation rates for Section 3.2.3 compliance.
    
    Args:
        kernels: List of kernel sizes K
        dilations: List of dilation rates D
        
    Raises:
        ValueError: If any parameters are invalid
        
    Note:
        Creates |K| × |D| distinct receptive field patterns as specified in paper.
    """
    validate_convolution_kernels(kernels)
    validate_convolution_dilations(dilations)


def calculate_receptive_field(kernel_size: int, dilation: int) -> int:
    """
    Calculate receptive field for a dilated convolution following Equation (9).
    
    Args:
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        
    Returns:
        Receptive field size in positions
        
    Formula:
        RF_{k,d} = 1 + (k-1) × d
    """
    return 1 + (kernel_size - 1) * dilation


def calculate_convolution_padding(kernel_size: int, dilation: int) -> int:
    """
    Calculate padding needed to maintain sequence length.
    
    Args:
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        
    Returns:
        Padding size to maintain input sequence length
    """
    return ((kernel_size - 1) * dilation) // 2
