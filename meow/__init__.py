"""
Meow: A Native Communication Protocol for Multi-Agent AI Systems

Meow combines learned compression (VQ-VAE) with emergent communication
to create an efficient, auditable protocol for agent-to-agent communication.

Example:
    from meow import MeowEncoder, MeowDecoder
    
    # Initialize
    encoder = MeowEncoder(codebook_path="codebook_v1.0.pt")
    decoder = MeowDecoder(codebook_path="codebook_v1.0.pt")
    
    # Encode agent embedding to Meow symbols
    embedding = get_agent_embedding()  # 768-dim vector
    symbols = encoder.encode(embedding)  # [42, 108, 256]
    
    # Decode back to human-readable text
    text = decoder.decode(symbols, level="detailed")

See https://github.com/wanikua/meow for documentation.
"""

__version__ = "0.1.0"
__author__ = "Meow Contributors"
__license__ = "MIT"

from .codebook import MeowCodebook
from .encoder import MeowEncoder
from .decoder import MeowDecoder
from .audit import MeowAudit
from .data import SyntheticEmbeddingDataset, EmbeddingFileDataset, create_dataloaders

__all__ = [
    "MeowCodebook",
    "MeowEncoder",
    "MeowDecoder",
    "MeowAudit",
    "SyntheticEmbeddingDataset",
    "EmbeddingFileDataset",
    "create_dataloaders",
]
