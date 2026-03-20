# Safety Guidelines for Meow Protocol

This document provides safety recommendations for researchers and practitioners using the Meow protocol. While these are not legally binding requirements, they represent best practices for responsible deployment.

---

## Mandatory Audit Layer

**Recommendation**: Any deployment of Meow in production systems SHOULD implement the audit layer as specified in `skills/meow-safety/SKILL.md`.

**Why**: Emergent communication can develop unexpected behaviors. Human oversight is essential for safe deployment.

**Implementation**:
- Decode random message samples (≥10% of traffic)
- Monitor alignment between messages and actions
- Log audit results for review

---

## Deployment Stages

| Stage | Audit Frequency | Human Review |
|-------|----------------|--------------|
| **Research** | After every experiment | Required |
| **Alpha Testing** | Daily batch audits | Required |
| **Beta Testing** | Real-time sampling (10%) | On-demand |
| **Production** | Continuous monitoring | Automated alerts + human escalation |

---

## Red Flags: When to Stop

**Immediately halt deployment if**:
1. Decode success rate drops below 80%
2. Alignment rate drops below 90%
3. High-severity deception detected (say-do mismatch)
4. Agents develop communication patterns that resist decoding

**Investigate if**:
- Codebook usage becomes highly unbalanced (>50% unused symbols)
- Cross-model compatibility degrades
- Side channel leakage detected

---

## Reporting Safety Concerns

**Where to report**:
- GitHub Issues: https://github.com/wanikua/meow/issues
- Security email: (TBD — will be added when project matures)
- AI Safety community: Alignment Forum, LessWrong

**What to include**:
- Experiment ID and configuration
- Audit logs showing the issue
- Steps to reproduce
- Suggested mitigation (if any)

---

## Responsible Disclosure

If you discover a security vulnerability or safety issue:

1. **Do NOT** publicly disclose immediately
2. Report privately via GitHub Security Advisories or direct email
3. Allow 90 days for maintainers to respond and patch
4. Coordinate public disclosure timing

---

## Ethical Considerations

**Potential misuse vectors**:
- Agents colluding to hide information from human overseers
- Developing communication that's technically decodable but misleading
- Using Meow to obfuscate malicious behavior

**Mitigation strategies**:
- Open-source all audit tools
- Encourage independent safety research
- Maintain diverse perspectives in project governance

---

## Community Safety Reviews

**Quarterly reviews**: Project maintainers will publish audit summaries:
- Experiments run
- Safety incidents (anonymized)
- Changes to protocol or auditing methods

**How to participate**:
- Join safety working group (details TBD)
- Review audit logs (public experiments only)
- Propose improvements to audit methods

---

## Legal Disclaimer

This document provides guidance, not legal obligations. Users are responsible for:
- Compliance with applicable laws and regulations
- Risk assessment for their specific deployment
- Obtaining appropriate ethical review if required

THE SOFTWARE IS PROVIDED "AS IS" (see LICENSE). Safety recommendations do not create warranties or legal liability.

---

## Updates

This document will evolve as the project matures. Check back regularly for updates.

**Last updated**: 2026-03-20  
**Version**: 0.1.0
