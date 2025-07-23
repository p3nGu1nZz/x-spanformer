#!/usr/bin/env python3
"""
X-Spanformer Kernel Package

Centralized kernel implementations and validation logic for multi-scale
convolutional encoders following Section 3.2.3 specifications.

This package provides reusable kernel components that can be shared
across different pipelines while maintaining mathematical correctness
and architectural consistency.
"""

from .conv_encoder_kernel import ConvEncoderKernel
from .validation import (
    validate_convolution_kernels,
    validate_convolution_dilations,
    validate_convolution_parameters,
    calculate_receptive_field,
    calculate_convolution_padding
)

__all__ = [
    'ConvEncoderKernel',
    'validate_convolution_kernels',
    'validate_convolution_dilations', 
    'validate_convolution_parameters',
    'calculate_receptive_field',
    'calculate_convolution_padding'
]
