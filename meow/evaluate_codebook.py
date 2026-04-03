"""
Meow Codebook Evaluation - Detailed codebook quality analysis.

Evaluates reconstruction quality, codebook usage, symbol distribution,
and produces a structured report.

Usage:
    python -m meow.evaluate_codebook --codebook codebooks/codebook_v0.1.pt --data data/embeddings.pt
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .codebook import MeowCodebook
from .data import EmbeddingFileDataset


@torch.no_grad()
def evaluate(args: argparse.Namespace):
    device = "cpu"

    # Load codebook
    codebook = MeowCodebook.load(args.codebook).to(device)
    codebook.eval()
    print(f"Codebook: {args.codebook}")
    print(f"  input_dim={codebook.input_dim}, codebook_dim={codebook.codebook_dim}, num_symbols={codebook.num_symbols}")

    # Load data
    dataset = EmbeddingFileDataset(args.data)
    data = dataset.data.to(device)
    print(f"Data: {args.data} ({len(data)} samples, dim={data.shape[1]})")

    # Deduplicate (find unique embeddings)
    unique_data = torch.unique(data, dim=0)
    print(f"Unique embeddings: {len(unique_data)}")

    # --- Reconstruction quality ---
    print("\n=== Reconstruction Quality ===")
    recon, info = codebook(unique_data, return_info=True)
    recon_loss = torch.nn.functional.mse_loss(recon, unique_data).item()
    per_sample_mse = ((recon - unique_data) ** 2).mean(dim=1)
    cosine_sim = torch.nn.functional.cosine_similarity(recon, unique_data, dim=1)

    print(f"  MSE (mean):      {recon_loss:.6f}")
    print(f"  MSE (median):    {per_sample_mse.median().item():.6f}")
    print(f"  MSE (p95):       {per_sample_mse.quantile(0.95).item():.6f}")
    print(f"  MSE (max):       {per_sample_mse.max().item():.6f}")
    print(f"  Cosine sim (mean): {cosine_sim.mean().item():.4f}")
    print(f"  Cosine sim (min):  {cosine_sim.min().item():.4f}")

    # --- Symbol usage ---
    print("\n=== Symbol Usage ===")
    indices = codebook.encode(unique_data)
    symbol_counts = Counter(indices.cpu().numpy().tolist())
    used_symbols = len(symbol_counts)
    total_symbols = codebook.num_symbols
    usage_rate = used_symbols / total_symbols

    counts_array = np.array([symbol_counts.get(i, 0) for i in range(total_symbols)])
    active_counts = counts_array[counts_array > 0]

    print(f"  Symbols used:    {used_symbols}/{total_symbols} ({usage_rate:.1%})")
    print(f"  Dead symbols:    {total_symbols - used_symbols}")

    if len(active_counts) > 0:
        print(f"  Usage (mean):    {active_counts.mean():.1f} samples/symbol")
        print(f"  Usage (std):     {active_counts.std():.1f}")
        print(f"  Usage (min):     {active_counts.min()}")
        print(f"  Usage (max):     {active_counts.max()}")

    # Perplexity
    probs = counts_array / counts_array.sum() if counts_array.sum() > 0 else counts_array
    probs_nz = probs[probs > 0]
    perplexity = np.exp(-np.sum(probs_nz * np.log(probs_nz)))
    print(f"  Perplexity:      {perplexity:.1f} (ideal range: {total_symbols * 0.5:.0f}-{total_symbols * 0.8:.0f})")

    # Top/bottom symbols
    top_10 = sorted(symbol_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 symbols: {', '.join(f'{s}({c})' for s, c in top_10)}")

    # --- Codebook geometry ---
    print("\n=== Codebook Geometry ===")
    cb_weights = codebook.quantizer.embedding.weight.data
    norms = cb_weights.norm(dim=1)
    print(f"  Embedding norms (mean): {norms.mean().item():.4f}")
    print(f"  Embedding norms (std):  {norms.std().item():.4f}")

    # Pairwise distances between used symbols
    used_indices = list(symbol_counts.keys())
    if len(used_indices) > 1:
        used_embeddings = cb_weights[used_indices]
        dists = torch.cdist(used_embeddings.unsqueeze(0), used_embeddings.unsqueeze(0)).squeeze(0)
        # Exclude diagonal
        mask = ~torch.eye(len(used_indices), dtype=torch.bool)
        pairwise = dists[mask]
        print(f"  Pairwise dist (mean):   {pairwise.mean().item():.4f}")
        print(f"  Pairwise dist (min):    {pairwise.min().item():.4f}")
        print(f"  Pairwise dist (max):    {pairwise.max().item():.4f}")

    # --- Summary ---
    report = {
        "codebook_path": args.codebook,
        "data_path": args.data,
        "num_samples": len(data),
        "num_unique": len(unique_data),
        "input_dim": codebook.input_dim,
        "codebook_dim": codebook.codebook_dim,
        "num_symbols": codebook.num_symbols,
        "reconstruction": {
            "mse_mean": round(recon_loss, 6),
            "mse_median": round(per_sample_mse.median().item(), 6),
            "mse_p95": round(per_sample_mse.quantile(0.95).item(), 6),
            "cosine_sim_mean": round(cosine_sim.mean().item(), 4),
            "cosine_sim_min": round(cosine_sim.min().item(), 4),
        },
        "usage": {
            "symbols_used": used_symbols,
            "usage_rate": round(usage_rate, 4),
            "dead_symbols": total_symbols - used_symbols,
            "perplexity": round(float(perplexity), 1),
        },
        "success_criteria": {
            "recon_loss_lt_0.5": recon_loss < 0.5,
            "usage_gt_80pct": usage_rate > 0.8,
        },
    }

    print("\n=== Success Criteria ===")
    print(f"  Reconstruction loss < 0.5:  {recon_loss:.6f}  {'PASS' if recon_loss < 0.5 else 'FAIL'}")
    print(f"  Codebook usage > 80%:       {usage_rate:.1%}  {'PASS' if usage_rate > 0.8 else 'FAIL'}")

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {output_path}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Meow codebook")
    parser.add_argument("--codebook", type=str, required=True, help="Path to codebook checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to embedding data")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON report")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
