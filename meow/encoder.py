"""
Meow Encoder - Encode agent embeddings to discrete Meow symbols.
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional
import numpy as np

from .codebook import MeowCodebook


class MeowEncoder:
    """
    Meow Encoder - Converts agent embeddings to discrete symbol sequences.
    
    Example:
        encoder = MeowEncoder(codebook_path="codebook_v1.0.pt")
        embedding = get_agent_embedding()  # 8192-dim
        symbols = encoder.encode(embedding)  # [42, 108, 256]
    """
    
    def __init__(
        self,
        codebook: Optional[MeowCodebook] = None,
        codebook_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the encoder.
        
        Args:
            codebook: Pre-loaded MeowCodebook instance
            codebook_path: Path to saved codebook checkpoint
            device: Device to run on (default: auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if codebook is not None:
            self.codebook = codebook.to(self.device)
        elif codebook_path is not None:
            self.codebook = MeowCodebook.load(codebook_path).to(self.device)
        else:
            raise ValueError("Must provide either codebook or codebook_path")
        
        self.codebook.eval()
        
    def encode(
        self,
        embeddings: Union[torch.Tensor, np.ndarray, List[float]],
        sequence_length: int = 3,
    ) -> torch.Tensor:
        """
        Encode embeddings to Meow symbol indices.
        
        Args:
            embeddings: Input embeddings, shape (batch_size, input_dim) or (input_dim,)
            sequence_length: Number of symbols per embedding (default: 3)
            
        Returns:
            symbols: Symbol indices, shape (batch_size, sequence_length)
        """
        # Convert to tensor
        if isinstance(embeddings, list):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        elif isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        # Ensure batch dimension
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        embeddings = embeddings.to(self.device)
        
        # Encode
        with torch.no_grad():
            # For now, simple single-symbol encoding
            # TODO: Support sequence encoding with attention
            symbols = self.codebook.encode(embeddings)
            
            # Repeat to sequence length (placeholder)
            if sequence_length > 1:
                symbols = symbols.unsqueeze(-1).repeat(1, sequence_length)
        
        return symbols
    
    def encode_batch(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode a large batch of embeddings.
        
        Args:
            embeddings: Input embeddings (N, input_dim)
            batch_size: Batch size for processing
            
        Returns:
            symbols: Symbol indices (N,)
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        n_samples = len(embeddings)
        all_symbols = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = embeddings[i:i+batch_size].to(self.device)
                symbols = self.codebook.encode(batch)
                all_symbols.append(symbols.cpu())
        
        return torch.cat(all_symbols, dim=0)
    
    def get_codebook_info(self) -> dict:
        """Get codebook metadata."""
        return {
            'input_dim': self.codebook.input_dim,
            'codebook_dim': self.codebook.codebook_dim,
            'num_symbols': self.codebook.num_symbols,
            'device': self.device,
        }
    
    def visualize_codebook(self) -> np.ndarray:
        """
        Get codebook embeddings for visualization.
        
        Returns:
            Codebook embeddings (num_symbols, codebook_dim)
        """
        return self.codebook.quantizer.embedding.weight.cpu().numpy()
