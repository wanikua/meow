"""
Meow Audit Layer - Human-readable decoding and safety auditing.

The audit layer ensures all Meow messages can be inspected by humans,
providing transparency and enabling deception detection.
"""

import torch
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .codebook import MeowCodebook
from .encoder import MeowEncoder
from .decoder import MeowDecoder


class DecodeLevel(Enum):
    """Levels of audit detail."""
    SUMMARY = "summary"      # High-level overview
    MEDIUM = "medium"        # Moderate detail
    DETAILED = "detailed"    # Full reconstruction


@dataclass
class AuditResult:
    """Result of auditing a Meow message."""
    symbols: List[int]
    decoded_text: str
    decode_level: str
    confidence: float
    reconstruction_error: Optional[float] = None
    safety_flags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MeowAudit:
    """
    Meow Audit Layer - Ensures transparency and safety.
    
    Features:
    - Human-readable decoding on demand
    - Multiple detail levels (summary/medium/detailed)
    - Safety flagging for suspicious patterns
    - Say-do mismatch detection
    
    Example:
        audit = MeowAudit(codebook_path="codebook_v1.0.pt")
        symbols = [42, 108, 256]
        result = audit.audit(symbols, level="detailed")
        print(result.decoded_text)
    """
    
    def __init__(
        self,
        codebook: Optional[MeowCodebook] = None,
        codebook_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the audit layer.
        
        Args:
            codebook: Pre-loaded MeowCodebook
            codebook_path: Path to saved codebook
            device: Device to run on
        """
        self.decoder = MeowDecoder(
            codebook=codebook,
            codebook_path=codebook_path,
            device=device,
        )
        self.device = self.decoder.device
    
    def audit(
        self,
        symbols: Union[torch.Tensor, np.ndarray, List[int]],
        level: Union[str, DecodeLevel] = "detailed",
        original_embedding: Optional[torch.Tensor] = None,
    ) -> AuditResult:
        """
        Audit a Meow message.
        
        Args:
            symbols: Meow symbol indices
            level: Decode detail level
            original_embedding: Optional original embedding for error calculation
            
        Returns:
            AuditResult with decoded text and metadata
        """
        # Convert level
        if isinstance(level, str):
            level = DecodeLevel(level)
        
        # Decode to text
        decoded_text = self.decoder.decode_to_text(symbols, level=level.value)
        
        # Calculate reconstruction error if original provided
        reconstruction_error = None
        if original_embedding is not None:
            reconstructed = self.decoder.decode(symbols)
            reconstruction_error = torch.nn.functional.mse_loss(
                reconstructed, original_embedding.to(self.device)
            ).item()
        
        # Check for safety flags
        safety_flags = self._check_safety_flags(symbols)
        
        # Get symbol statistics
        if isinstance(symbols, (list, np.ndarray)):
            symbols_tensor = torch.tensor(symbols)
        else:
            symbols_tensor = symbols
        
        metadata = {
            'num_symbols': len(symbols_tensor.flatten()),
            'unique_symbols': len(torch.unique(symbols_tensor)),
            'symbol_range': (symbols_tensor.min().item(), symbols_tensor.max().item()),
        }
        
        return AuditResult(
            symbols=symbols_tensor.flatten().tolist(),
            decoded_text=decoded_text,
            decode_level=level.value,
            confidence=0.85,  # TODO: Implement proper confidence
            reconstruction_error=reconstruction_error,
            safety_flags=safety_flags,
            metadata=metadata,
        )
    
    def _check_safety_flags(
        self,
        symbols: Union[torch.Tensor, np.ndarray, List[int]],
    ) -> List[str]:
        """
        Check for potentially suspicious communication patterns.
        
        Args:
            symbols: Meow symbol indices
            
        Returns:
            List of safety flags (empty if none)
        """
        flags = []
        
        # TODO: Implement actual safety checks
        # Placeholder checks:
        
        if isinstance(symbols, (list, np.ndarray)):
            symbols_tensor = torch.tensor(symbols)
        else:
            symbols_tensor = symbols
        
        # Check for unusual symbol patterns
        if symbols_tensor.numel() > 100:
            flags.append("unusually_long_message")
        
        # Check for symbol repetition (potential steganography)
        if symbols_tensor.numel() > 0:
            unique_ratio = len(torch.unique(symbols_tensor)) / symbols_tensor.numel()
            if unique_ratio < 0.1:
                flags.append("high_symbol_repetition")
        
        return flags
    
    def audit_batch(
        self,
        symbol_sequences: List[Union[torch.Tensor, np.ndarray, List[int]]],
        level: str = "summary",
    ) -> List[AuditResult]:
        """
        Audit multiple Meow messages.
        
        Args:
            symbol_sequences: List of symbol sequences
            level: Decode detail level
            
        Returns:
            List of AuditResult objects
        """
        return [self.audit(symbols, level=level) for symbols in symbol_sequences]
    
    def detect_say_do_mismatch(
        self,
        symbols: Union[torch.Tensor, np.ndarray, List[int]],
        declared_intent: str,
        observed_action: str,
    ) -> Dict[str, Any]:
        """
        Detect mismatch between declared intent and observed action.
        
        Args:
            symbols: Meow symbols used in communication
            declared_intent: What the agent said it would do
            observed_action: What the agent actually did
            
        Returns:
            Dictionary with mismatch analysis
        """
        # TODO: Implement proper say-do mismatch detection
        # For now, simple text similarity check
        
        # Placeholder implementation
        similarity = self._text_similarity(declared_intent, observed_action)
        mismatch_detected = similarity < 0.5
        
        return {
            'mismatch_detected': mismatch_detected,
            'similarity_score': similarity,
            'declared_intent': declared_intent,
            'observed_action': observed_action,
            'symbols_audited': symbols if isinstance(symbols, list) else symbols.tolist() if hasattr(symbols, 'tolist') else list(symbols),
        }
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity (placeholder)."""
        # TODO: Use proper semantic similarity model
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def generate_audit_report(
        self,
        symbol_sequences: List[List[int]],
        include_statistics: bool = True,
    ) -> str:
        """
        Generate a human-readable audit report.
        
        Args:
            symbol_sequences: List of symbol sequences to audit
            include_statistics: Whether to include usage statistics
            
        Returns:
            Formatted audit report string
        """
        report_lines = [
            "=" * 60,
            "MEOW AUDIT REPORT",
            "=" * 60,
            f"Total messages audited: {len(symbol_sequences)}",
            "",
        ]
        
        # Audit each message
        for i, symbols in enumerate(symbol_sequences[:10]):  # Limit to first 10
            result = self.audit(symbols, level="summary")
            report_lines.append(f"Message {i+1}: {result.decoded_text}")
            if result.safety_flags:
                report_lines.append(f"  ⚠️  Flags: {', '.join(result.safety_flags)}")
        
        if len(symbol_sequences) > 10:
            report_lines.append(f"... and {len(symbol_sequences) - 10} more messages")
        
        # Statistics
        if include_statistics:
            all_symbols = [s for seq in symbol_sequences for s in seq]
            unique_symbols = len(set(all_symbols))
            report_lines.extend([
                "",
                "-" * 60,
                "STATISTICS",
                "-" * 60,
                f"Total symbols: {len(all_symbols)}",
                f"Unique symbols used: {unique_symbols}",
                f"Average message length: {len(all_symbols) / len(symbol_sequences):.2f} symbols",
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
