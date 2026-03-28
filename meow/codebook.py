"""
Meow Codebook - VQ-VAE style discrete representation learning.

This module implements the codebook layer of the Meow protocol,
which learns a fixed set of discrete symbols for agent communication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.
    
    Maps continuous embeddings to discrete codebook symbols.
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 768,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        """
        Initialize the vector quantizer.
        
        Args:
            num_embeddings: Number of discrete symbols in codebook (default: 512)
            embedding_dim: Dimension of each embedding (default: 768)
            commitment_cost: Weight for commitment loss (default: 0.25)
            decay: EMA decay factor (default: 0.99)
            epsilon: Small constant for numerical stability (default: 1e-5)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize input embeddings to nearest codebook symbols.
        
        Args:
            inputs: Continuous embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            quantized: Quantized embeddings (same shape as inputs)
            indices: Codebook symbol indices (batch_size,)
            info: Dictionary with loss and diagnostics
        """
        # Calculate distances to codebook entries
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            - 2 * torch.matmul(inputs, self.embedding.weight.T)
            + torch.sum(self.embedding.weight**2, dim=1)
        )
        
        # Find nearest neighbors
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Calculate losses
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # EMA updates (during training)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * encodings.sum(dim=0)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * torch.matmul(encodings.T, inputs)
            
            # Normalize and update codebook
            n = self.ema_cluster_size.sum()
            normalized_cluster_size = (
                (self.ema_cluster_size + self.epsilon) / 
                (n + self.num_embeddings * self.epsilon) * n
            )
            self.embedding.weight.data = self.ema_w / normalized_cluster_size.unsqueeze(1)
        
        # Pass gradients
        quantized = inputs + (quantized - inputs).detach()
        
        # Diagnostics
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage_rate = (encodings.sum(dim=0) > 0).float().mean()
        
        info = {
            'loss': loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'perplexity': perplexity,
            'usage_rate': usage_rate,
            'encoding_indices': encoding_indices,
        }
        
        return quantized, encoding_indices, info


class MeowCodebook(nn.Module):
    """
    Meow Codebook - Complete VQ-VAE codebook for agent communication.
    
    Architecture:
        Encoder: 8192 → 768 (linear projection)
        Codebook: 512 discrete symbols, each 768-dim
        Decoder: 768 → 8192 (linear projection)
    """
    
    def __init__(
        self,
        input_dim: int = 8192,
        codebook_dim: int = 768,
        num_symbols: int = 512,
        commitment_cost: float = 0.25,
    ):
        """
        Initialize the Meow codebook.
        
        Args:
            input_dim: Input embedding dimension (default: 8192 for LLaMA-3-70B)
            codebook_dim: Codebook embedding dimension (default: 768)
            num_symbols: Number of discrete symbols (default: 512)
            commitment_cost: VQ commitment cost (default: 0.25)
        """
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.num_symbols = num_symbols
        
        # Encoder: project input to codebook dimension
        self.encoder = nn.Linear(input_dim, codebook_dim)
        
        # Vector quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_symbols,
            embedding_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )
        
        # Decoder: project back to input dimension
        self.decoder = nn.Linear(codebook_dim, input_dim)
        
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode input embeddings to codebook symbol indices.
        
        Args:
            inputs: Input embeddings (batch_size, input_dim)
            
        Returns:
            symbol_indices: Codebook symbol indices (batch_size,)
        """
        encoded = self.encoder(inputs)
        _, indices, _ = self.quantizer(encoded)
        return indices
    
    def decode(self, symbol_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook symbol indices back to input space.
        
        Args:
            symbol_indices: Codebook symbol indices (batch_size,)
            
        Returns:
            reconstructed: Reconstructed embeddings (batch_size, input_dim)
        """
        # Get codebook embeddings
        codebook_embeddings = self.quantizer.embedding(symbol_indices)
        # Decode to input space
        reconstructed = self.decoder(codebook_embeddings)
        return reconstructed
    
    def forward(
        self, 
        inputs: torch.Tensor,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Full encode-quantize-decode forward pass.
        
        Args:
            inputs: Input embeddings (batch_size, input_dim)
            return_info: Whether to return diagnostic info
            
        Returns:
            reconstructed: Reconstructed embeddings (batch_size, input_dim)
            info: Diagnostic info (if return_info=True)
        """
        # Encode
        encoded = self.encoder(inputs)
        
        # Quantize
        quantized, indices, vq_info = self.quantizer(encoded)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, inputs)
        
        # Total loss
        total_loss = recon_loss + vq_info['loss']
        
        if return_info:
            info = {
                'total_loss': total_loss,
                'reconstruction_loss': recon_loss,
                **vq_info,
            }
            return reconstructed, info
        
        return reconstructed, None
    
    def get_usage_statistics(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        """
        Evaluate codebook usage statistics on a dataset.
        
        Args:
            dataloader: DataLoader with input embeddings
            
        Returns:
            Dictionary with usage statistics
        """
        self.eval()
        all_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch.to(next(self.parameters()).device)
                indices = self.encode(inputs)
                all_indices.append(indices.cpu())
        
        all_indices = torch.cat(all_indices, dim=0)
        
        # Calculate statistics
        unique_symbols = torch.unique(all_indices)
        usage_counts = torch.bincount(all_indices.flatten(), minlength=self.num_symbols)
        
        stats = {
            'total_samples': len(all_indices),
            'unique_symbols_used': len(unique_symbols),
            'usage_rate': len(unique_symbols) / self.num_symbols,
            'mean_usage': usage_counts.float().mean().item(),
            'std_usage': usage_counts.float().std().item(),
            'min_usage': usage_counts.min().item(),
            'max_usage': usage_counts.max().item(),
        }
        
        return stats
    
    def save(self, path: str):
        """Save codebook to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'codebook_dim': self.codebook_dim,
            'num_symbols': self.num_symbols,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'MeowCodebook':
        """Load codebook from disk."""
        checkpoint = torch.load(path, map_location='cpu')
        codebook = cls(
            input_dim=checkpoint['input_dim'],
            codebook_dim=checkpoint['codebook_dim'],
            num_symbols=checkpoint['num_symbols'],
        )
        codebook.load_state_dict(checkpoint['model_state_dict'])
        return codebook
