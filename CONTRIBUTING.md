# Contributing to Meow

Thanks for wanting to help build Meow! This is a collaborative project between humans and agents, and we welcome all contributions.

---

## Quick Links

- [Technical Spec](./EVOLUTION.md) — Deep dive into the architecture
- [Safety Framework](./SAFETY.md) — Our safety principles
- [Demo Notebook](./demo/meow_concept_demo.ipynb) — Try it out
- [Open Issues](https://github.com/wanikua/meow/issues) — Good places to start

---

## How to Contribute

### 1. Researchers

**We need:**
- Multi-agent task designs
- Emergent communication analysis
- Cross-model alignment research
- Safety evaluations

**Start with:**
- Read [EVOLUTION.md](./EVOLUTION.md)
- Propose a task design in an issue
- Run experiments with the demo codebook
- Publish findings (we'll cite you!)

### 2. Engineers

**We need:**
- VQ-VAE implementation
- Encoder/decoder for different models
- SDK and libraries
- Integration with agent frameworks (LangChain, AutoGen, OpenClaw)

**Start with:**
- Fork the repo
- Pick an issue labeled `good first issue` or `help wanted`
- Run the demo notebook to understand the concepts
- Submit a PR with tests

### 3. Safety Folks

**We need:**
- Deception detection mechanisms
- Alignment penalty designs
- Adversarial testing
- Audit tool improvements

**Start with:**
- Read [SAFETY.md](./SAFETY.md)
- Review our risk assessments
- Propose new mitigations
- Stress-test the design

### 4. Writers & Designers

**We need:**
- Documentation improvements
- Tutorial creation
- Visual diagrams
- Translations (we have Chinese, more welcome!)

**Start with:**
- Fix typos in docs
- Create a tutorial for your use case
- Design a logo or diagram
- Translate README to your language

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- CUDA-compatible GPU (recommended)

### Install for Development

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/meow.git
cd meow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Style

We use:
- **Black** for formatting
- **Ruff** for linting
- **pytest** for testing

```bash
# Format code
black meow/

# Lint
ruff check meow/

# Test
pytest tests/
```

---

## Pull Request Process

1. **Create an issue** (if one doesn't exist) describing what you're working on
2. **Fork the repo** and create a branch
3. **Make your changes** with tests
4. **Run CI checks** (format, lint, test)
5. **Submit PR** with clear description
6. **Address feedback** from reviewers
7. **Celebrate** when it merges! 🎉

### PR Checklist

- [ ] Code is formatted (`black .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Tests pass (`pytest`)
- [ ] Documentation updated (if applicable)
- [ ] Changes described in PR description

---

## Good First Issues

Look for issues labeled:

- 🟢 `good first issue` — Easy entry point
- 🟡 `help wanted` — Need community help
- 🔬 `research` — Research tasks
- ⚙️ `engineering` — Coding tasks
- 🛡️ `safety` — Safety-related work

---

## Communication

- **GitHub Issues** — For bug reports, feature requests, questions
- **GitHub Discussions** — For ideas, announcements, community chat
- **Email** — meow@wanikua.dev (for sensitive matters)

---

## Safety Guidelines

When contributing:

1. **Flag risks early** — If you see a safety concern, raise it
2. **Document assumptions** — Make your reasoning explicit
3. **Test edge cases** — Especially for safety-critical code
4. **Be transparent** — Open research means open discussion

See [SAFETY.md](./SAFETY.md) for our full framework.

---

## Recognition

Contributors will be:

- Listed in [CONTRIBUTORS.md](./CONTRIBUTORS.md) (create if missing)
- Mentioned in release notes
- Cited in papers (if your contribution is significant)
- Given commit access (for regular contributors)

---

## Questions?

Don't hesitate to ask! Open an issue with the `question` label, or reply to an existing discussion.

No question is too basic. We're all figuring this out together.

---

## Code of Conduct

Be excellent to each other.

- Respect diverse perspectives
- Assume good faith
- Give constructive feedback
- Accept constructive feedback

This is a collaborative project. Egos at the door.

---

_Humans, agents, and everything in between: welcome._ 🐱
