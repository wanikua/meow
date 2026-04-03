"""
Meow Codebook Training Script

Train a VQ-VAE codebook on agent embeddings.

Usage:
    python -m meow.train_codebook                          # synthetic data, defaults
    python -m meow.train_codebook --data path/to/embs.pt   # real embeddings
    python -m meow.train_codebook --epochs 200 --lr 3e-4   # custom hyperparams
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.optim as optim

from .codebook import MeowCodebook
from .data import create_dataloaders


def train_one_epoch(
    model: MeowCodebook,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch. Returns average metrics."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_perplexity = 0.0
    total_usage = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        _, info = model(batch, return_info=True)
        loss = info["total_loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += info["reconstruction_loss"].item()
        total_vq += info["loss"].item()
        total_perplexity += info["perplexity"].item()
        total_usage += info["usage_rate"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "vq_loss": total_vq / n_batches,
        "perplexity": total_perplexity / n_batches,
        "usage_rate": total_usage / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: MeowCodebook,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate on validation set. Returns average metrics."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_perplexity = 0.0
    total_usage = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        _, info = model(batch, return_info=True)

        total_loss += info["total_loss"].item()
        total_recon += info["reconstruction_loss"].item()
        total_perplexity += info["perplexity"].item()
        total_usage += info["usage_rate"].item()
        n_batches += 1

    return {
        "val_loss": total_loss / n_batches,
        "val_recon_loss": total_recon / n_batches,
        "val_perplexity": total_perplexity / n_batches,
        "val_usage_rate": total_usage / n_batches,
    }


def train(args: argparse.Namespace) -> Path:
    """Main training loop. Returns path to saved codebook."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print(f"Preparing data (samples={args.num_samples}, dim={args.input_dim})...")
    train_loader, val_loader = create_dataloaders(
        num_samples=args.num_samples,
        embedding_dim=args.input_dim,
        batch_size=args.batch_size,
        data_path=args.data,
        noise_std=args.noise_std,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = MeowCodebook(
        input_dim=args.input_dim,
        codebook_dim=args.codebook_dim,
        num_symbols=args.num_symbols,
        commitment_cost=args.commitment_cost,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_path = output_dir / "codebook_best.pt"

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        history.append(metrics)

        # Save best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            model.save(str(best_path))
            marker = " *"
        else:
            marker = ""

        # Log
        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss {train_metrics['loss']:.4f} | "
                f"recon {train_metrics['recon_loss']:.4f} | "
                f"ppl {train_metrics['perplexity']:.1f} | "
                f"usage {train_metrics['usage_rate']:.2f} | "
                f"val_loss {val_metrics['val_loss']:.4f} | "
                f"{elapsed:.1f}s{marker}"
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            model.save(str(output_dir / f"codebook_epoch{epoch}.pt"))

    # Save final
    final_path = output_dir / "codebook_final.pt"
    model.save(str(final_path))

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    print("-" * 70)
    print(f"Training complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final usage rate: {history[-1]['usage_rate']:.2f}")
    print(f"  Final perplexity: {history[-1]['perplexity']:.1f}")
    print(f"  Saved: {best_path}")

    # Check success criteria
    final = history[-1]
    print("\nSuccess criteria:")
    print(f"  Reconstruction loss < 0.5: {final['val_recon_loss']:.4f} {'✓' if final['val_recon_loss'] < 0.5 else '✗'}")
    print(f"  Codebook usage > 80%:      {final['val_usage_rate']:.2f} {'✓' if final['val_usage_rate'] > 0.8 else '✗'}")

    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Meow codebook")

    # Data
    parser.add_argument("--data", type=str, default=None, help="Path to embedding file (.pt/.npy)")
    parser.add_argument("--num-samples", type=int, default=100_000, help="Synthetic sample count")
    parser.add_argument("--input-dim", type=int, default=8192, help="Embedding dimension")

    # Model
    parser.add_argument("--codebook-dim", type=int, default=768, help="Codebook embedding dim")
    parser.add_argument("--num-symbols", type=int, default=512, help="Number of codebook symbols")
    parser.add_argument("--commitment-cost", type=float, default=0.25, help="VQ commitment cost")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise augmentation std")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--log-every", type=int, default=5, help="Log every N epochs")
    parser.add_argument("--save-every", type=int, default=25, help="Checkpoint every N epochs")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
