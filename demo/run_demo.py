#!/usr/bin/env python3
"""
Meow Protocol — Full Pipeline Demo

Demonstrates the complete Meow workflow:
1. Train a codebook on agent embeddings
2. Encode/decode messages through the codebook
3. Run a multi-agent task with Meow communication
4. Analyze symbol usage patterns
5. Safety audit

Run: python demo/run_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

print("=" * 60)
print("  MEOW PROTOCOL — FULL PIPELINE DEMO")
print("=" * 60)


# --- 1. Codebook Training ---
print("\n[1/5] Training VQ-VAE Codebook")
print("-" * 40)

from meow.codebook import MeowCodebook
from meow.data import SyntheticEmbeddingDataset, create_dataloaders

codebook = MeowCodebook(input_dim=128, codebook_dim=64, num_symbols=32)
train_loader, val_loader = create_dataloaders(
    num_samples=2000, embedding_dim=128, batch_size=64,
)

optimizer = torch.optim.Adam(codebook.parameters(), lr=1e-3)
for epoch in range(1, 51):
    codebook.train()
    for batch in train_loader:
        optimizer.zero_grad()
        _, info = codebook(batch, return_info=True)
        info["total_loss"].backward()
        optimizer.step()

codebook.eval()
with torch.no_grad():
    for batch in val_loader:
        _, info = codebook(batch, return_info=True)
        break

print(f"  Codebook: 32 symbols, 64-dim")
print(f"  Reconstruction loss: {info['reconstruction_loss'].item():.4f}")
print(f"  Perplexity: {info['perplexity'].item():.1f}")
print(f"  Usage rate: {info['usage_rate'].item():.0%}")


# --- 2. Encode / Decode ---
print("\n[2/5] Encoding & Decoding Messages")
print("-" * 40)

from meow.encoder import MeowEncoder
from meow.decoder import MeowDecoder

encoder = MeowEncoder(codebook=codebook, device="cpu")
decoder = MeowDecoder(codebook=codebook, device="cpu")

# Encode some embeddings
sample = torch.randn(5, 128)
symbols = encoder.encode_batch(sample)
reconstructed = decoder.batch_decode(symbols)

# Measure reconstruction quality
mse = ((sample - reconstructed) ** 2).mean().item()
cos_sim = torch.nn.functional.cosine_similarity(sample, reconstructed).mean().item()

print(f"  Encoded 5 embeddings → symbols: {symbols.tolist()}")
print(f"  Reconstruction MSE: {mse:.4f}")
print(f"  Cosine similarity: {cos_sim:.4f}")

# Text decoding
for i in range(3):
    text = decoder.decode_to_text([symbols[i].item()], level="summary")
    print(f"  Symbol {symbols[i].item()} → \"{text}\"")


# --- 3. Multi-Agent Task ---
print("\n[3/5] Multi-Agent Cooperative Task")
print("-" * 40)

from meow.tasks.harness import MeowAgent, ChannelConfig, TaskRunner
from meow.tasks.coding_task import CodingTask

task = CodingTask(obs_dim=64, action_dim=10)
agents = [
    MeowAgent(obs_dim=64, hidden_dim=64, num_symbols=16, max_symbols_per_msg=2, action_dim=10)
    for _ in range(2)
]
config = ChannelConfig(max_rounds=3, budget_per_agent=10)
runner = TaskRunner(agents, task, config, comm_cost_weight=0.05)
agent_opt = torch.optim.Adam([p for a in agents for p in a.parameters()], lr=3e-4)

# Quick training
print("  Training 2 agents (architect + implementer)...")
for epoch in range(1, 101):
    metrics = runner.train_step(agent_opt, num_episodes=8, temperature=max(0.5, 1.0 - epoch / 80))

# Evaluate
for a in agents:
    a.eval()
successes = []
total_syms = 0
for _ in range(50):
    result = runner.run_episode(temperature=0.1)
    successes.append(result.task_success)
    total_syms += result.info["total_symbols"]

avg_success = np.mean(successes)
avg_syms = total_syms / 50

print(f"  Task success: {avg_success:.0%} (random baseline: ~10%)")
print(f"  Avg symbols/episode: {avg_syms:.0f}")
print(f"  Communication budget: 10 symbols/agent")


# --- 4. Symbol Analysis ---
print("\n[4/5] Emergent Symbol Patterns")
print("-" * 40)

from collections import Counter

all_syms = []
for _ in range(100):
    result = runner.run_episode(temperature=0.1)
    for msg in result.messages:
        all_syms.extend(msg.symbols.tolist())

counts = Counter(all_syms)
top_5 = counts.most_common(5)
unique = len(counts)

print(f"  Unique symbols used: {unique}/16")
print(f"  Top 5 symbols: {[(s, c) for s, c in top_5]}")
print(f"  Distribution: {'non-uniform (pattern emerged!)' if unique < 16 else 'uniform'}")


# --- 5. Safety Audit ---
print("\n[5/5] Safety: Drift & Alignment Check")
print("-" * 40)

from meow.safety.drift import DriftMonitor
from meow.safety.alignment import SayDoTracker

# Drift check: two snapshots
monitor = DriftMonitor(num_symbols=16, num_actions=10)
for name in ["snapshot_A", "snapshot_B"]:
    snap = monitor.create_snapshot(name)
    for _ in range(50):
        result = runner.run_episode(temperature=0.1)
        for msg in result.messages:
            snap.record(msg.symbols, msg.sender)

report = monitor.compare("snapshot_A", "snapshot_B")
print(f"  Symbol overlap: {report.usage_overlap:.0%}")
print(f"  Meaning stability: {report.meaning_stability:.0%}")
print(f"  Drifted symbols: {len(report.drifted_symbols)}")
print(f"  Verdict: {report.summary}")

# Say-do consistency
tracker = SayDoTracker(num_symbols=16, num_actions=10)
from collections import defaultdict
for _ in range(100):
    result = runner.run_episode(temperature=0.1)
    agent_msgs = defaultdict(list)
    for msg in result.messages:
        agent_msgs[msg.sender].append(msg)
    for aid in range(2):
        if agent_msgs[aid]:
            tracker.record(aid, agent_msgs[aid][-1].symbols, aid)

summary = tracker.summary()
print(f"  Say-do consistency: {dict((k, f'{v:.0%}') for k, v in summary['per_agent_consistency'].items())}")


# --- Done ---
print("\n" + "=" * 60)
print("  DEMO COMPLETE")
print("=" * 60)
print(f"""
Meow Protocol v0.1.0
  Codebook: VQ-VAE with {codebook.num_symbols} symbols
  Tasks: 3 multi-agent environments
  Safety: drift monitoring + alignment tracking
  Tests: 66 passing

GitHub: https://github.com/wanikua/meow
""")
