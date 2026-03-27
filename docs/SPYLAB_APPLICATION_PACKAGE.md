# SPY Lab 申请材料包

**整理时间：** 2026-03-27  
**申请目标：** ETH Zurich SPY Lab Semester Project  
**项目主题：** Meow - Auditable Emergent Communication for Multi-Agent AI

---

## 📋 材料清单

| 文件 | 状态 | 位置 |
|------|------|------|
| 项目提案 | ✅ 完成 | `SPYLAB_PROJECT_PROPOSAL.md` |
| 个人陈述草稿 | ✅ 完成 | 见下方 |
| CV 模板 | ✅ 完成 | 见下方 |
| Related Work 对比 | ✅ 完成 | 提案第 3 节 |
| 菠萝王朝经历 | ⏳ 待补充 | 需要 GitHub 链接 |

---

## 📝 个人陈述 (Personal Statement)

**用途：** 申请表 "Description of Interests" 字段，或邮件正文  
**字数：** 约 600 词

```markdown
Dear SPY Lab Team,

I am applying for a semester project focused on the security and auditability of 
emergent communication in multi-agent AI systems.

## Background & Motivation

I have been following your lab's work on adversarial ML and AI safety, particularly:
- AgentDojo (ICLR 2025) — evaluating robustness of AI agents
- Research on LLM unlearning failures (NeurIPS 2024 SoLaR)
- Adversarial attacks on model alignment (ICLR 2025)

These works share a common thread: as AI systems become more autonomous and 
multi-agent, we need new tools to understand, audit, and secure their behavior.

## Relevant Experience

### Multi-Agent System Development

I have hands-on experience building multi-agent systems through my project 
**Boluobobo Dynasty (菠萝王朝)**, which has gained 2,000+ stars on GitHub.

**What I built:**
- A workflow system with multiple specialized agent roles collaborating on tasks
- Agent communication using natural language and JSON schemas

**What I learned:**
1. Communication overhead scales linearly with agent count — becomes prohibitive at 10+ agents
2. Debugging requires manual inspection of conversation logs — not scalable
3. No efficient middle ground between verbose natural language and opaque embeddings

These practical limitations motivated me to explore: *What if agents had their own 
native communication protocol — efficient like embeddings, but auditable like natural language?*

This question led to my research proposal: the Meow protocol.

### Meow Protocol Research

Meow is a native communication protocol for AI agents with three key properties:

1. **Efficient**: Uses a discrete codebook (VQ-VAE learned) instead of verbose 
   natural language — targeting 5-10x token reduction.

2. **Auditable**: Any Meow message can be decoded to human-readable text on demand. 
   Transparency by design, not as an afterthought.

3. **Emergent**: Usage patterns evolve through multi-agent tasks, not human design. 
   This allows agents to develop efficient conventions while maintaining auditability.

## Connection to SPY Lab's Research

Meow directly connects to your lab's work in several ways:

**Adversarial Perspective** (like AgentDojo):
- I plan to design adversarial agents that attempt to develop deceptive 
  communication patterns
- Measure whether the audit layer can detect say-do mismatches
- Quantify the "deception success rate" under different reward structures

**Alignment & Unlearning** (like your NeurIPS 2024 work):
- Meow's audit layer could help detect when agents are hiding misaligned intentions
- Symbol usage patterns might reveal emergent deceptive strategies
- This complements your work on understanding failure modes in LLM systems

**Novel Contribution:**
- Silo-Bench (Zhang et al., 2026) identifies the "communication-reasoning gap" 
  but doesn't propose a solution
- ST-EVO (Wu et al., 2026) evolves communication topologies but uses continuous 
  signals (not discrete/auditable)
- Meow is the first to combine: discrete codebook + emergent usage + mandatory audit

## Proposed Semester Project

**Goal:** Implement and evaluate the safety mechanisms of Meow.

**Deliverables:**
1. VQ-VAE codebook training (512 symbols, trained on LLaMA embeddings)
2. Multi-agent task harness with communication budget constraints
3. Deception detection experiments (adversarial agents with hidden goals)
4. Audit layer implementation (decode any message to human text)
5. Open-source release + technical report

**Timeline (16 weeks):**
- Weeks 1-4: Codebook training + encoder/decoder
- Weeks 5-8: Multi-agent tasks + baseline experiments
- Weeks 9-12: Adversarial experiments (deception detection)
- Weeks 13-16: Analysis + writing + open-source release

## Why SPY Lab?

Your lab's adversarial approach to AI safety is exactly what this project needs. 
I'm not just building a communication protocol — I'm building one that can be 
stress-tested, audited, and proven safe under adversarial conditions.

The combination of your expertise in adversarial ML + my Meow protocol could 
produce novel insights into emergent communication safety — a critical but 
underexplored area as multi-agent AI scales.

I would welcome the opportunity to discuss this project with you. I'm happy to 
adapt the scope based on your feedback and the lab's priorities.

Best regards,
[Your Name]
[Your Email]
[Your GitHub Profile]
```

---

## 📄 CV 模板

**用途：** 申请附件  
**格式：** Markdown（可转 PDF）

```markdown
# [Your Name]

[Email] | [Phone] | [GitHub] | [Website]

---

## Education

**[Degree Program]** | [University] | [Expected Graduation: Date]
- GPA: [X.X / 4.0]
- Relevant Coursework: Machine Learning, Deep Learning, Natural Language Processing, 
  Multi-Agent Systems, Computer Security

---

## Research Interests

- Multi-Agent AI Systems
- AI Safety & Alignment
- Emergent Communication
- Adversarial Machine Learning

---

## Projects

### Meow Protocol | Research Project
**github.com/wanikua/meow** | [Date] - Present

A native communication protocol for AI agents, combining efficiency (VQ-VAE codebook) 
with auditability (human-decodable messages).

**Contributions:**
- Proposed novel approach combining discrete codebooks with emergent communication
- Designed 3-layer architecture: Codebook + Emergence + Audit
- Created conceptual demo with VQ-VAE training simulation
- Wrote technical specification and safety framework

**Technologies:** Python, PyTorch, VQ-VAE, Multi-Agent RL

---

### Boluobobo Dynasty (菠萝王朝) | Multi-Agent Workflow System
**[GitHub Link]** | [Date] - Present | ⭐ 2,000+ stars

A multi-agent collaboration system for [具体功能].

**Contributions:**
- Designed agent orchestration workflow with [X] specialized agent roles
- Implemented communication layer using [LangChain/AutoGen/custom framework]
- Achieved [具体成果，如：10x faster content production]
- Open-source release gained 2,000+ stars, demonstrating community interest

**Technologies:** Python, [LLM APIs], [Agent Framework], [Other tools]

**Key Insights:**
- Communication overhead becomes prohibitive at scale (10+ agents)
- Debugging multi-agent conversations requires better tooling
- Motivated research into native agent communication protocols

---

### [Other Project 1]
[Brief description, your role, technologies used, outcomes]

### [Other Project 2]
[Brief description, your role, technologies used, outcomes]

---

## Skills

**Programming Languages:** Python, [Others]
**ML Frameworks:** PyTorch, TensorFlow, [Others]
**Agent Frameworks:** LangChain, AutoGen, [Others]
**Tools:** Git, Docker, [Others]
**Languages:** [Chinese (Native), English (Fluent), Others]

---

## Publications & Writing

- [Any blog posts, articles, or papers you've written]
- Meow Protocol Documentation (github.com/wanikua/meow)

---

## References

Available upon request.
```

---

## 📊 Related Work 对比表

**用途：** 面试时展示，或作为附加材料

```markdown
# Related Work & Meow's Differentiation

## Emergent Communication

| Paper | Key Idea | Limitation | How Meow Differs |
|-------|----------|------------|------------------|
| CommNet (2016) | Shared hidden states | Continuous, not interpretable | Meow uses discrete codebook |
| DIAL (2016) | Differentiable communication | Task-specific only | Meow is general-purpose |
| Mordatch & Abbeel (2018) | Emergent grounded language | Not cross-model | Meow targets cross-model compat |
| ST-EVO (2026) | Evolving topologies | Continuous signals | Meow is discrete + auditable |
| Silo-Bench (2026) | Coordination benchmark | No solution proposed | Meow is the solution |

## AI Safety & Multi-Agent

| Paper | Key Idea | Connection to Meow |
|-------|----------|-------------------|
| AgentDojo (SPY Lab, 2025) | Agent robustness benchmark | Meow could integrate |
| Unlearning Failures (2024) | LLMs retain knowledge | Meow audit could detect |
| Deceptive Alignment (2024) | Agents hide misalignment | Meow is a mitigation |
| TrinityGuard (2026) | MAS safety framework | Meow complements |
```

---

## 🔗 申请链接

| 项目 | 链接 |
|------|------|
| SPY Lab 申请表 (ETH 账号) | https://forms.office.com/e/zgGqUpNbCF |
| SPY Lab 申请表 (无 ETH 账号) | https://forms.office.com/e/TR9aWWsbkL |
| SPY Lab 官网 | https://spylab.ai/ |
| Meow GitHub | https://github.com/wanikua/meow |
| 菠萝王朝 GitHub | [待补充] |

---

## ✅ 申请前检查清单

- [ ] CV 转 PDF（用 LaTeX 或 Canva 美化）
- [ ] 成绩单扫描（PDF）
- [ ] 个人陈述填入申请表
- [ ] 项目提案转 PDF 作为附件
- [ ] 菠萝王朝 GitHub 链接填入 CV
- [ ] 检查所有链接是否有效
- [ ] 邮件联系潜在导师（可选）
- [ ] 提交申请

---

## 📧 邮件模板（联系导师）

**主题：** Semester Project Inquiry: Auditable Emergent Communication for Multi-Agent AI

```
Dear Professor [Name] / Dr. [Name],

I hope this email finds you well.

My name is [Your Name], and I am a [year] [degree program] student at [University]. 
I am writing to inquire about the possibility of doing a semester project under your 
supervision at the SPY Lab.

I have been following your lab's work on [specific paper/topic], and I am particularly 
interested in applying adversarial perspectives to multi-agent AI safety.

I have attached a project proposal titled "Meow: Auditable Emergent Communication for 
Safe Multi-Agent AI Systems" that aligns with your lab's research on [specific area].

**My background:**
- Developed Boluobobo Dynasty (菠萝王朝), a multi-agent system with 2,000+ GitHub stars
- Currently researching native communication protocols for AI agents
- [Other relevant experience]

**Project overview:**
- Goal: Build an auditable communication protocol for multi-agent AI
- Methods: VQ-VAE codebook + multi-agent RL + adversarial testing
- Deliverables: Open-source code, technical report, Colab demo

I would be grateful for the opportunity to discuss this project with you. I am happy 
to adapt the scope based on your feedback and the lab's priorities.

Attached: Project Proposal (PDF), CV (PDF), Transcript (PDF)

Best regards,
[Your Name]
[Email]
[GitHub]
```

---

**最后更新：** 2026-03-27  
**下一步：** 补充菠萝王朝 GitHub 链接，转 PDF，提交申请
