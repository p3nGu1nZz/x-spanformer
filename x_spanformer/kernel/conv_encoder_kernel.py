#!/usr/bin/env python3
"""
conv_encoder_kernel.py

Multi-scale dilated convolutional encoder kernel implementing Section 3.2.3
of the X-Spanformer paper. Provides contextual embedding computation through
hierarchical pattern capture with configurable receptive fields.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .validation import (
    validate_convolution_parameters,
    calculate_receptive_field,
    calculate_convolution_padding
)


class ConvEncoderKernel(nn.Module):
    """
    Multi-scale dilated convolutional encoder implementing the contextual
    embedding computation described in Section 3.2.3.
    
    This kernel provides hierarchical pattern capture through multiple
    receptive field scales, enabling the model to capture both local
    character-level patterns and broader compositional structures.
    """
    
    def __init__(self, embed_dim: int, kernels: List[int], dilations: List[int],
                 dropout_rate: float = 0.1, device: Optional[str] = None):
        """
        Initialize multi-scale convolutional encoder kernel.
        
        Args:
            embed_dim: Embedding dimension d
            kernels: REQUIRED - Kernel sizes K for multi-scale convolution (e.g., [3, 5, 7])
            dilations: REQUIRED - Dilation rates D for receptive field patterns (e.g., [1, 2, 4])
            dropout_rate: Dropout rate for contextualization
            device: PyTorch device for computation. If None, uses 'cuda' if available, else 'cpu'
        """
        super().__init__()
        
        # Smart device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.embed_dim = embed_dim
        
        # Validate critical parameters using centralized validation
        validate_convolution_parameters(kernels, dilations)
        
        # Store validated parameters
        self.kernels = kernels
        self.dilations = dilations
        
        # Build multi-scale dilated convolutions following Section 3.2.3 specifications
        # Creates |K| × |D| distinct receptive field patterns
        self.conv_layers = nn.ModuleList()
        self.receptive_fields = []
        
        for kernel in self.kernels:
            for dilation in self.dilations:
                padding = calculate_convolution_padding(kernel, dilation)
                rf = calculate_receptive_field(kernel, dilation)
                
                self.conv_layers.append(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel, 
                             dilation=dilation, padding=padding)
                )
                self.receptive_fields.append(rf)
        
        # Calculate total number of pathways dynamically
        num_pathways = len(self.kernels) * len(self.dilations)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dynamic output projection from (num_pathways * d) back to d
        self.output_proj = nn.Linear(num_pathways * embed_dim, embed_dim)
        
        # Move all modules to the specified device
        self.to(device)
    
    def forward(self, seed_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale contextualization to seed embeddings.
        
        Args:
            seed_embeddings: Seed embeddings H^0 ∈ R^{T × d}
            
        Returns:
            Contextual embeddings H ∈ R^{T × d}
            
        Implementation follows Equations (7-8) from Section 3.2.3:
        1. Multi-scale convolutions with different kernel sizes and dilations
        2. Pathway concatenation and projection
        3. Residual connection and layer normalization
        """
        # Transpose for conv1d: (T, d) -> (d, T)
        x = seed_embeddings.transpose(-2, -1)
        
        # Apply multi-scale convolutions following Equation (7)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.gelu(conv(x))
            # Ensure output length matches input length by trimming if necessary
            if conv_out.size(-1) != x.size(-1):
                # Trim to match input length
                diff = conv_out.size(-1) - x.size(-1)
                conv_out = conv_out[..., diff//2:-(diff-diff//2)] if diff > 0 else conv_out
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features: list of (d, T) -> (num_pathways*d, T)
        concatenated = torch.cat(conv_outputs, dim=-2)  # Concatenate along channel dimension
        
        # Transpose back and project: (num_pathways*d, T) -> (T, num_pathways*d) -> (T, d)
        concatenated = concatenated.transpose(-2, -1)
        output = self.output_proj(concatenated)
        
        # Residual connection and layer norm following Equation (8)
        output = self.layer_norm(output + seed_embeddings)
        output = self.dropout(output)
        
        return output
    
    def get_receptive_field_info(self) -> List[tuple]:
        """
        Get information about receptive fields for each pathway.
        
        Returns:
            List of (kernel, dilation, receptive_field) tuples for debugging
        """
        info = []
        idx = 0
        for kernel in self.kernels:
            for dilation in self.dilations:
                rf = self.receptive_fields[idx]
                info.append((kernel, dilation, rf))
                idx += 1
        return info
    
    def get_pathway_count(self) -> int:
        """Get the total number of convolution pathways."""
        return len(self.kernels) * len(self.dilations)
