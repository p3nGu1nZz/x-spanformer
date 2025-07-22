#!/usr/bin/env python3
"""
test_kernel_validation.py

Unit tests for the kernel package validation functions.
Tests centralized validation logic for convolution parameters.
"""

import unittest
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_spanformer.kernel.validation import (
    validate_convolution_kernels,
    validate_convolution_dilations, 
    validate_convolution_parameters,
    calculate_receptive_field,
    calculate_convolution_padding
)
from x_spanformer.kernel import ConvEncoderKernel


class TestKernelValidation(unittest.TestCase):
    """Test kernel parameter validation functions."""
    
    def test_validate_convolution_kernels_valid(self):
        """Test valid kernel configurations."""
        # Standard configuration
        validate_convolution_kernels([3, 5, 7])
        
        # Single kernel
        validate_convolution_kernels([3])
        
        # Odd kernels only
        validate_convolution_kernels([1, 9, 11])
    
    def test_validate_convolution_kernels_invalid(self):
        """Test invalid kernel configurations."""
        # None kernels
        with self.assertRaises(ValueError) as cm:
            validate_convolution_kernels(None)  # type: ignore
        self.assertIn("kernels parameter is required", str(cm.exception))
        
        # Empty list
        with self.assertRaises(ValueError) as cm:
            validate_convolution_kernels([])
        self.assertIn("non-empty list", str(cm.exception))
        
        # Even kernels
        with self.assertRaises(ValueError) as cm:
            validate_convolution_kernels([2, 4, 6])
        self.assertIn("positive odd integers", str(cm.exception))
        
        # Zero kernel
        with self.assertRaises(ValueError) as cm:
            validate_convolution_kernels([0, 3, 5])
        self.assertIn("positive odd integers", str(cm.exception))
        
        # Negative kernel
        with self.assertRaises(ValueError) as cm:
            validate_convolution_kernels([-1, 3, 5])
        self.assertIn("positive odd integers", str(cm.exception))
        
        # Wrong type
        with self.assertRaises(ValueError) as cm:
            validate_convolution_kernels("not a list")  # type: ignore
        self.assertIn("non-empty list", str(cm.exception))
    
    def test_validate_convolution_dilations_valid(self):
        """Test valid dilation configurations."""
        # Standard configuration
        validate_convolution_dilations([1, 2, 4])
        
        # Single dilation
        validate_convolution_dilations([1])
        
        # Various positive integers
        validate_convolution_dilations([1, 3, 8, 16])
    
    def test_validate_convolution_dilations_invalid(self):
        """Test invalid dilation configurations."""
        # None dilations
        with self.assertRaises(ValueError) as cm:
            validate_convolution_dilations(None)  # type: ignore
        self.assertIn("dilations parameter is required", str(cm.exception))
        
        # Empty list
        with self.assertRaises(ValueError) as cm:
            validate_convolution_dilations([])
        self.assertIn("non-empty list", str(cm.exception))
        
        # Zero dilation
        with self.assertRaises(ValueError) as cm:
            validate_convolution_dilations([0, 1, 2])
        self.assertIn("positive integers", str(cm.exception))
        
        # Negative dilation
        with self.assertRaises(ValueError) as cm:
            validate_convolution_dilations([-1, 1, 2])
        self.assertIn("positive integers", str(cm.exception))
        
        # Wrong type
        with self.assertRaises(ValueError) as cm:
            validate_convolution_dilations("not a list")  # type: ignore
        self.assertIn("non-empty list", str(cm.exception))
    
    def test_validate_convolution_parameters_combined(self):
        """Test combined parameter validation."""
        # Valid combination
        validate_convolution_parameters([3, 5, 7], [1, 2, 4])
        
        # Invalid kernels
        with self.assertRaises(ValueError):
            validate_convolution_parameters([2, 4, 6], [1, 2, 4])
        
        # Invalid dilations  
        with self.assertRaises(ValueError):
            validate_convolution_parameters([3, 5, 7], [0, 1, 2])
    
    def test_calculate_receptive_field(self):
        """Test receptive field calculation."""
        # Basic cases from Section 3.2.3
        self.assertEqual(calculate_receptive_field(3, 1), 3)   # 1 + (3-1)*1 = 3
        self.assertEqual(calculate_receptive_field(5, 2), 9)   # 1 + (5-1)*2 = 9
        self.assertEqual(calculate_receptive_field(7, 4), 25)  # 1 + (7-1)*4 = 25
        
        # Edge cases
        self.assertEqual(calculate_receptive_field(1, 1), 1)   # 1 + (1-1)*1 = 1
        self.assertEqual(calculate_receptive_field(3, 8), 17)  # 1 + (3-1)*8 = 17
    
    def test_calculate_convolution_padding(self):
        """Test padding calculation."""
        # Standard cases
        self.assertEqual(calculate_convolution_padding(3, 1), 1)  # ((3-1)*1)//2 = 1
        self.assertEqual(calculate_convolution_padding(5, 1), 2)  # ((5-1)*1)//2 = 2
        self.assertEqual(calculate_convolution_padding(7, 1), 3)  # ((7-1)*1)//2 = 3
        
        # With dilation
        self.assertEqual(calculate_convolution_padding(3, 2), 2)  # ((3-1)*2)//2 = 2
        self.assertEqual(calculate_convolution_padding(5, 2), 4)  # ((5-1)*2)//2 = 4
        self.assertEqual(calculate_convolution_padding(7, 4), 12) # ((7-1)*4)//2 = 12


class TestConvEncoderKernelValidation(unittest.TestCase):
    """Test ConvEncoderKernel validation integration."""
    
    def test_encoder_validation_integration(self):
        """Test that ConvEncoderKernel properly uses validation."""
        embed_dim = 64
        
        # Valid configuration should work
        encoder = ConvEncoderKernel(embed_dim, [3, 5, 7], [1, 2, 4])
        self.assertEqual(encoder.get_pathway_count(), 9)
        
        # Test that validation catches errors at construction time
        # Note: Using type: ignore to bypass static type checking for testing
        with self.assertRaises(ValueError):
            ConvEncoderKernel(embed_dim, [2, 4], [1, 2])  # type: ignore
        
        with self.assertRaises(ValueError):
            ConvEncoderKernel(embed_dim, [3, 5], [0, 1])  # type: ignore
    
    def test_encoder_dynamic_pathway_calculation(self):
        """Test different pathway configurations."""
        embed_dim = 32
        
        # Small configuration: 2x2 = 4 pathways
        small_encoder = ConvEncoderKernel(embed_dim, [3, 5], [1, 2])
        self.assertEqual(small_encoder.get_pathway_count(), 4)
        self.assertEqual(len(small_encoder.conv_layers), 4)
        
        # Large configuration: 4x3 = 12 pathways
        large_encoder = ConvEncoderKernel(embed_dim, [1, 3, 5, 7], [1, 2, 4])
        self.assertEqual(large_encoder.get_pathway_count(), 12)
        self.assertEqual(len(large_encoder.conv_layers), 12)
    
    def test_encoder_receptive_field_info(self):
        """Test receptive field information retrieval."""
        encoder = ConvEncoderKernel(64, [3, 5], [1, 2])
        rf_info = encoder.get_receptive_field_info()
        
        # Should have 4 entries (2 kernels x 2 dilations)
        self.assertEqual(len(rf_info), 4)
        
        # Check specific combinations
        expected_combinations = [
            (3, 1, 3),   # kernel=3, dilation=1, rf=3
            (3, 2, 5),   # kernel=3, dilation=2, rf=5
            (5, 1, 5),   # kernel=5, dilation=1, rf=5
            (5, 2, 9),   # kernel=5, dilation=2, rf=9
        ]
        
        self.assertEqual(rf_info, expected_combinations)


if __name__ == '__main__':
    unittest.main()
