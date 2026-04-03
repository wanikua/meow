"""
Meow Decoder - Decode Meow symbols back to embeddings or human-readable text.
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict
import numpy as np

from .codebook import MeowCodebook


class MeowDecoder:
    """
    Meow Decoder - Converts discrete Meow symbols back to embeddings or text.
    
    Supports multiple decode levels:
    - summary: High-level description
    - medium: Moderate detail
    - detailed: Full reconstruction
    
    Example:
        decoder = MeowDecoder(codebook_path="codebook_v1.0.pt")
        symbols = torch.tensor([42, 108, 256])
        embedding = decoder.decode(symbols)  # Reconstructed embedding
        text = decoder.decode_to_text(symbols, level="detailed")
    """
    
    def __init__(
        self,
        codebook: Optional[MeowCodebook] = None,
        codebook_path: Optional[str] = None,
        text_decoder_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the decoder.
        
        Args:
            codebook: Pre-loaded MeowCodebook instance
            codebook_path: Path to saved codebook checkpoint
            text_decoder_model: Model name for text generation (e.g., "llama-3-8b")
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
        self.text_decoder_model = text_decoder_model
        
        # TODO: Load fine-tuned text decoder when available
        self.text_decoder = None
    
    def decode(
        self,
        symbols: Union[torch.Tensor, np.ndarray, List[int]],
    ) -> torch.Tensor:
        """
        Decode symbol indices back to embedding space.
        
        Args:
            symbols: Symbol indices, shape (batch_size,) or (batch_size, seq_len)
            
        Returns:
            reconstructed: Reconstructed embeddings (batch_size, input_dim)
        """
        # Convert to tensor
        if isinstance(symbols, list):
            symbols = torch.tensor(symbols, dtype=torch.long)
        elif isinstance(symbols, np.ndarray):
            symbols = torch.from_numpy(symbols).long()
        
        # Ensure correct shape
        if symbols.dim() == 1:
            symbols = symbols.unsqueeze(0)
        
        # Use first symbol for now (TODO: support sequence decoding)
        symbols = symbols[:, 0]
        
        symbols = symbols.to(self.device)
        
        with torch.no_grad():
            reconstructed = self.codebook.decode(symbols)
        
        return reconstructed
    
    def decode_to_text(
        self,
        symbols: Union[torch.Tensor, np.ndarray, List[int]],
        level: str = "detailed",
    ) -> Union[str, List[str]]:
        """
        Decode symbols to human-readable text.
        
        Args:
            symbols: Symbol indices
            level: Decode level - "summary", "medium", or "detailed"
            
        Returns:
            Text description(s)
        """
        # Decode to embedding first
        embedding = self.decode(symbols)

        # Normalize symbols to a flat list of ints for text generation
        if isinstance(symbols, list):
            symbol_ids = symbols
        elif isinstance(symbols, np.ndarray):
            symbol_ids = symbols.flatten().tolist()
        else:
            symbol_ids = symbols.flatten().tolist()

        batch_size = embedding.shape[0]

        if level == "summary":
            texts = [f"[Meow message: {symbol_ids[0]}]"]
        elif level == "medium":
            texts = [f"[Meow symbol {symbol_ids[0]}: agent state update]"]
        else:  # detailed
            texts = [f"[Meow symbol {symbol_ids[0]}: reconstructed embedding dim={embedding.shape[1]}]"]

        return texts[0] if batch_size == 1 else texts
    
    def decode_with_confidence(
        self,
        symbols: Union[torch.Tensor, np.ndarray, List[int]],
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Decode with confidence scores.
        
        Args:
            symbols: Symbol indices
            
        Returns:
            Dictionary with reconstruction and confidence metrics
        """
        embedding = self.decode(symbols)
        
        # Calculate reconstruction quality (placeholder)
        # TODO: Implement proper confidence estimation
        confidence = 0.85  # Placeholder
        
        return {
            'embedding': embedding,
            'confidence': confidence,
            'symbols': symbols if isinstance(symbols, torch.Tensor) else torch.tensor(symbols),
        }
    
    def get_symbol_meaning(
        self,
        symbol_index: int,
    ) -> Dict[str, float]:
        """
        Get the embedding vector for a specific symbol.
        
        Args:
            symbol_index: Index of the symbol (0 to num_symbols-1)
            
        Returns:
            Dictionary with symbol information
        """
        with torch.no_grad():
            embedding = self.codebook.quantizer.embedding.weight[symbol_index]
        
        return {
            'symbol_index': symbol_index,
            'embedding_norm': embedding.norm().item(),
            'embedding_mean': embedding.mean().item(),
            'embedding_std': embedding.std().item(),
        }
    
    def batch_decode(
        self,
        symbols: Union[torch.Tensor, np.ndarray],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Decode a large batch of symbols.
        
        Args:
            symbols: Symbol indices (N,) or (N, seq_len)
            batch_size: Batch size for processing
            
        Returns:
            Reconstructed embeddings (N, input_dim)
        """
        if isinstance(symbols, np.ndarray):
            symbols = torch.from_numpy(symbols).long()
        
        if symbols.dim() == 1:
            symbols = symbols.unsqueeze(1)
        
        n_samples = len(symbols)
        all_reconstructions = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = symbols[i:i+batch_size].to(self.device)
                recon = self.codebook.decode(batch[:, 0])
                all_reconstructions.append(recon.cpu())
        
        return torch.cat(all_reconstructions, dim=0)
