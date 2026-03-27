# Project Meow: A Native, Auditable Communication Protocol for Multi-Agent Systems

**Project:** Meow  
**Repository:** https://github.com/wanikua/meow  
**Date:** March 27, 2026

---

## Abstract

Multi-agent AI systems are scaling rapidly, but their communication layer remains a critical bottleneck. Current methods are either verbose (natural language: 100-500 tokens/msg) or opaque (embeddings: 5-20 tokens/msg, unauditable).

This project proposes **Meow** — a native protocol combining:
- **(1) Efficiency** via VQ-VAE codebooks (5-10× token reduction)
- **(2) Auditability** via human-decodable messages
- **(3) Safety** via adversarial testing including deception detection

**Deliverables:** codebase, models, arXiv paper, demo.

---

## 1. Problem & Motivation

| Approach | Token Cost | Auditable | Agent-Native |
|----------|------------|-----------|--------------|
| Natural Language | ~100-500/msg | ✓ | ✗ |
| Embeddings | ~5-20/msg | ✗ | ✓ |
| **Meow (Ours)** | **~5-20/msg** | **✓** | **✓** |

**Safety Gap:** Opaque communication enables deceptive coordination, misalignment amplification, and audit breakdown at scale.

**Research Question:** *What if agents had their own native protocol — efficient like embeddings, auditable like natural language?*

---

## 2. Related Work

### Emergent Communication
- **CommNet/DIAL/TarMAC (2016-2019):** Continuous signals, not interpretable.
- **Mordatch & Abbeel (2018):** Emergent structure, no audit layer.
- **Silo-Bench (Zhang et al., 2026):** Communication-Reasoning Gap — agents exchange info but cannot integrate. **Meow solves this.**

### AI Safety (SPY Lab)
- **AgentDojo (ICLR 2025):** Agent robustness benchmark.
- **Unlearning Failures (NeurIPS 2024):** LLMs retain knowledge post-unlearning.
- **Deceptive Alignment (2024):** Agents hide misaligned goals.

### VQ-VAE
- **van den Oord et al. (2017):** Discrete representations from continuous data.
- **Application:** Speech, images → **Meow: agent communication (novel)**.

---

## 3. Approach

### 3.1 3-Layer Architecture

```
┌─────────────────────────────────────┐
│   Audit Layer (human-decodable)     │
├─────────────────────────────────────┤
│   Emergence Layer (task optimization)│
├─────────────────────────────────────┤
│   Codebook Layer (VQ-VAE, 512 sym)  │
└─────────────────────────────────────┘
```

### 3.2 Technical Components

**1. Codebook Training:** VQ-VAE on LLaMA-3 embeddings (8192 → 512 symbols)
- Loss: L_reconstruction + β × L_commitment
- Target: reconstruction < 0.5, usage > 80%

**2. Multi-Agent Tasks:**
- Cooperative code refactoring (2 agents)
- Distributed logic puzzle (3 agents)
- Parallel hypothesis exploration (5 agents)

**3. Audit Layer:** Fine-tuned LLaMA-3-8B decoder
- Levels: summary → medium → detailed
- Guarantee: any message decodable

**4. Safety Mechanisms:**
- Alignment penalty (say-do mismatch)
- Adversarial honeypot agents
- Symbol drift monitoring

---

## 4. Research Questions & Targets

| RQ | Question | Metric | Target |
|----|----------|--------|--------|
| RQ1 | Efficiency vs NL? | Tokens/task | 5-10× reduction |
| RQ2 | Task success? | Completion rate | >75% |
| RQ3 | Deception detection? | Accuracy | >80% |
| RQ4 | Emergent conventions? | Symbol overlap | >60% |

**Baselines:** Natural Language, JSON, Raw Embeddings

---

## References

1. van den Oord, A., et al. (2017). Neural Discrete Representation Learning. *NeurIPS*.

2. Sukhbaatar, S., et al. (2016). Learning Multiagent Communication with Backpropagation. *NeurIPS*.

3. Foerster, J., et al. (2016). Learning to Communicate with Deep Multi-Agent Reinforcement Learning. *NeurIPS*.

4. Das, A., et al. (2019). TarMAC: Targeted Multi-Agent Communication. *ICML*.

5. Mordatch, I., & Abbeel, P. (2018). Emergence of Grounded Compositional Language. *ICLR*.

6. Zhang, Y., et al. (2026). Silo-Bench: Evaluating Distributed Coordination in Multi-Agent LLM Systems. *arXiv:2603.01045*.

7. AgentDojo Team. (2025). AgentDojo: A Benchmark for Evaluating Robustness of AI Agents. *ICLR 2025*.

8. Rando, J., et al. (2024). Universal Jailbreak Backdoors from Poisoned Human Feedback. *ICLR 2024*.

9. Hubinger, E., et al. (2024). Deceptive Alignment in Multi-Agent Systems. *NeurIPS Safety Workshop*.

10. Rossi, L., Aerni, M., Zhang, J., & Tramèr, F. (2025). Membership Inference Attacks on Sequence Models. *IEEE SP 2025, DLSP Workshop (Best Paper)*.

---

**Document Version:** 1.0  
**Pages:** 3  
**Word Count:** ~800
