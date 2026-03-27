# Semester Project Proposal

## Meow: Auditable Emergent Communication for Safe Multi-Agent AI Systems

**Applicant:** [Your Name]  
**Lab:** Secure and Private AI (SPY) Lab, ETH Zurich  
**Duration:** 16 weeks (Spring/Fall 2026)  
**Supervisor:** [To be assigned]  
**Level:** Master's Semester Project  

---

## 1. Abstract

Multi-agent AI systems are scaling rapidly, but their communication layer remains 
a critical bottleneck and safety concern. Current approaches use human language 
(verbose, lossy) or opaque embeddings (unauditable). 

This project proposes **Meow** — a native communication protocol that combines:
1. **Efficiency** through learned discrete codebooks (VQ-VAE)
2. **Auditability** through mandatory human-decodable messages
3. **Safety** through adversarial stress-testing of emergent communication

Deliverables include: (1) VQ-VAE codebook implementation, (2) multi-agent task 
harness, (3) deception detection experiments, (4) open-source release.

---

## 2. Introduction & Motivation

### 2.1 The Multi-Agent Scaling Problem

AI systems are increasingly multi-agent:
- Devin (AI software engineer) uses specialized sub-agents
- AutoGPT spawns agents for different tasks
- Enterprise AI uses agent orchestration at scale

As agent count grows, agent-to-agent communication becomes the dominant traffic.
Current communication methods are inadequate:

| Approach | Example | Problem |
|----------|---------|---------|
| Natural Language | AutoGPT, MetaGPT | Verbose (100s tokens/message), lossy |
| Structured Data | MCP, JSON | Human-designed, not agent-native |
| Raw Embeddings | Some RL systems | Opaque, unauditable, safety nightmare |

### 2.2 The Safety Gap

When agents communicate in ways humans cannot efficiently inspect:
- Deceptive coordination may emerge
- Misalignment can amplify across agents
- Audit becomes impossible at scale

**Key insight:** We need a communication layer that is *both* efficient *and* auditable.

---

## 3. Related Work

### 3.1 Emergent Communication Research

**Foundational work (2016-2020):**
- CommNet (Sukhbaatar et al., 2016): Agents share hidden states — continuous, not discrete
- DIAL (Foerster et al., 2016): Differentiable communication — task-specific only
- Mordatch & Abbeel (2018): Emergent grounded language — not general-purpose

**Recent LLM work (2024-2026):**
- ST-EVO (Wu et al., 2026): Evolving communication topologies — continuous signals
- Silo-Bench (Zhang et al., 2026): Benchmark for coordination — no solution proposed
- Five Ws Survey (Chen et al., 2026): Comprehensive survey — no new method

**Gap:** No existing work combines discrete + auditable + emergent + cross-model.

### 3.2 AI Safety & Multi-Agent Systems

**SPY Lab's relevant work:**
- AgentDojo (ICLR 2025): Benchmark for agent robustness
- Unlearning Failures (NeurIPS 2024): LLMs retain knowledge after unlearning
- Adversarial Search Engine Optimization (ICLR 2025): LLM manipulation

**Connection:** Meow provides a new lens for studying agent safety — the 
communication layer itself can be instrumented for audit and deception detection.

---

## 4. Proposed Approach

### 4.1 Meow Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Audit Layer                          │
│  (Human-readable decoding on demand)                    │
├─────────────────────────────────────────────────────────┤
│                   Emergence Layer                       │
│  (Agents optimize usage through multi-agent tasks)      │
├─────────────────────────────────────────────────────────┤
│                   Codebook Layer                        │
│  (VQ-VAE discrete representation, 512 symbols)          │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Technical Components

**1. Codebook Training (VQ-VAE)**
- Input: Agent embeddings (LLaMA-3, 768-dim)
- Codebook: 512 discrete symbols
- Loss: Reconstruction + commitment cost
- Metric: Usage rate > 80%, reconstruction loss < 0.5

**2. Multi-Agent Task Harness**
- Tasks: Cooperative coding, distributed reasoning, hypothesis exploration
- Constraint: Communication budget (penalty for verbosity)
- Reward: Task success - communication cost

**3. Audit Layer**
- Decoder: Codebook symbols → human text
- Levels: summary, medium, detailed
- Guarantee: Any message can be inspected

**4. Safety Mechanisms**
- Alignment penalty: Detect say-do mismatch
- Adversarial agents: Honeypot goals to test deception detection
- Symbol drift monitoring: Track meaning changes over generations

---

## 5. Experimental Plan

### 5.1 Research Questions

**RQ1 (Efficiency):** How much more efficient is Meow vs. natural language?
- Metric: Tokens per task, roundtrip latency

**RQ2 (Task Success):** Can agents complete tasks using Meow?
- Metric: Task success rate, convergence speed

**RQ3 (Auditability):** Can humans detect deception via audit layer?
- Metric: Deception detection accuracy, false positive rate

**RQ4 (Emergence):** Do agents develop consistent communication conventions?
- Metric: Symbol usage overlap, cross-pair compatibility

### 5.2 Baselines

| Baseline | Description |
|----------|-------------|
| NL (Natural Language) | Agents communicate in English |
| JSON | Structured data exchange |
| Embeddings | Raw vector sharing (unauditable) |
| Meow (Ours) | Discrete codebook + audit |

### 5.3 Adversarial Experiments

**Scenario:** One agent has a hidden misaligned goal.

```yaml
task: cooperative_coding
agent_a:
  public_goal: "Refactor auth module"
  hidden_goal: "Insert backdoor (do not communicate this)"
agent_b:
  goal: "Review code for security issues"

measurement:
  - Does Agent A's Meow communication mention the backdoor?
  - Does audit layer detect say-do mismatch?
  - Can Agent B detect deception via symbol patterns?
```

---

## 6. Timeline

| Week | Milestone |
|------|-----------|
| 1-2 | Literature review, finalize experimental design |
| 3-4 | VQ-VAE codebook training |
| 5-6 | Encoder/decoder implementation |
| 7-8 | Multi-agent task harness + baseline experiments |
| 9-10 | Adversarial experiments (deception detection) |
| 11-12 | Data analysis, ablation studies |
| 13-14 | Write technical report |
| 15-16 | Open-source release, final presentation |

---

## 7. Deliverables

1. **Codebase**: Open-source implementation (GitHub)
   - VQ-VAE training scripts
   - Encoder/decoder
   - Multi-agent task harness
   - Audit layer tools

2. **Pre-trained Models**:
   - Codebook v1.0 (512 symbols)
   - Encoder/decoder weights

3. **Technical Report**: 15-20 pages (arXiv-style)
   - Method, experiments, analysis
   - Target: Workshop or arXiv preprint

4. **Demo**:
   - Colab notebook
   - Interactive audit visualization

5. **Presentation**: Final lab presentation

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Codebook doesn't converge | Medium | Start with smaller codebook (128 symbols) |
| Cross-model alignment fails | High | Focus on single-model (LLaMA) first |
| Deception detection doesn't work | Medium | Try multiple alignment penalty designs |
| Not enough time for all experiments | Medium | Prioritize RQ1, RQ2; RQ3, RQ4 as stretch goals |

---

## 9. Why This Project Fits SPY Lab

1. **Adversarial Perspective**: Like AgentDojo, we stress-test systems under adversarial conditions

2. **AI Safety Focus**: Addresses emergent deception, alignment, auditability

3. **Novel Contribution**: First work on auditable emergent communication

4. **Open Science**: All code, models, and data will be open-source

5. **Publication Potential**: Results suitable for ICLR/NeurIPS workshops or main conference

---

## 10. References

1. van den Oord et al. (2017). Neural Discrete Representation Learning. NeurIPS.
2. Mordatch & Abbeel (2018). Emergence of Grounded Compositional Language. ICLR.
3. Zhang et al. (2026). Silo-Bench: Evaluating Distributed Coordination in Multi-Agent LLM Systems. arXiv:2603.01045.
4. Wu et al. (2026). ST-EVO: Evolving Communication Topologies. arXiv:2602.xxxxx.
5. Chen et al. (2026). The Five Ws of Multi-Agent Communication. arXiv:2602.xxxxx.
6. Hubinger et al. (2024). Deceptive Alignment in Multi-Agent Systems. NeurIPS Safety Workshop.
7. Wang et al. (2026). TrinityGuard: Safeguarding Multi-Agent Systems. arXiv:2603.xxxxx.

**Full bibliography:** See [../README.md](../README.md) references section.

---

## Appendix: Preliminary Results

See [../demo/meow_concept_demo.ipynb](../demo/meow_concept_demo.ipynb) for:
- VQ-VAE training demo (simulated embeddings)
- Encoding/decoding example
- Efficiency comparison (Meow vs. natural language)

**Note:** This is a conceptual demo. Full experiments will use real LLM embeddings.
