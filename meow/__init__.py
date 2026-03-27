"""
Meow Protocol - A native communication protocol for AI agents

Not human language compressed. Something new.
"""

__version__ = "0.1.0"
__author__ = "Meow Contributors"

from .encoder import MeowEncoder
from .decoder import MeowDecoder
from .codebook import Codebook

__all__ = ["MeowEncoder", "MeowDecoder", "Codebook"]
