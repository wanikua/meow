"""
Meow Data Module - Training data preparation for codebook training.

Provides synthetic embedding generation for initial development,
and a Dataset interface for loading real model embeddings.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import numpy as np


class SyntheticEmbeddingDataset(Dataset):
    """
    Generate synthetic embeddings that mimic structure of real LLM hidden states.

    Creates clustered embeddings to simulate the fact that real embeddings
    have semantic structure (similar concepts cluster together).
    """

    def __init__(
        self,
        num_samples: int = 100_000,
        embedding_dim: int = 8192,
        num_clusters: int = 64,
        cluster_std: float = 0.3,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            num_samples: Total number of embeddings to generate
            embedding_dim: Dimension of each embedding
            num_clusters: Number of semantic clusters
            cluster_std: Standard deviation within clusters
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim

        rng = np.random.RandomState(seed)

        # Generate cluster centers (spread across the space)
        centers = rng.randn(num_clusters, embedding_dim).astype(np.float32)
        # Normalize centers to unit sphere, then scale
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * np.sqrt(embedding_dim)

        # Assign samples to clusters
        assignments = rng.randint(0, num_clusters, size=num_samples)

        # Generate samples around cluster centers
        data = np.empty((num_samples, embedding_dim), dtype=np.float32)
        for i in range(num_samples):
            data[i] = centers[assignments[i]] + rng.randn(embedding_dim).astype(np.float32) * cluster_std

        self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class EmbeddingFileDataset(Dataset):
    """
    Load pre-computed embeddings from a .pt or .npy file.

    Expected format:
        .pt:  torch.Tensor of shape (N, embedding_dim)
        .npy: numpy array of shape (N, embedding_dim)
    """

    def __init__(self, path: str, noise_std: float = 0.0):
        """
        Args:
            path: Path to .pt or .npy embedding file
            noise_std: Gaussian noise std to add per sample (0 = disabled).
                       Helps prevent codebook collapse when corpus diversity is low.
        """
        if path.endswith(".npy"):
            self.data = torch.from_numpy(np.load(path)).float()
        elif path.endswith(".pt"):
            self.data = torch.load(path, map_location="cpu").float()
        else:
            raise ValueError(f"Unsupported file format: {path}")
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x


def create_dataloaders(
    num_samples: int = 100_000,
    embedding_dim: int = 8192,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    data_path: Optional[str] = None,
    noise_std: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        num_samples: Number of synthetic samples (ignored if data_path set)
        embedding_dim: Embedding dimension (ignored if data_path set)
        batch_size: Batch size
        val_ratio: Fraction of data used for validation
        num_workers: DataLoader workers
        seed: Random seed
        data_path: Path to pre-computed embeddings (overrides synthetic)
        noise_std: Gaussian noise for augmentation (0 = disabled)

    Returns:
        (train_loader, val_loader)
    """
    if data_path is not None:
        dataset = EmbeddingFileDataset(data_path, noise_std=noise_std)
    else:
        dataset = SyntheticEmbeddingDataset(
            num_samples=num_samples,
            embedding_dim=embedding_dim,
            seed=seed,
        )

    # Split
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
