# Security Policy

This is a research codebase (label-free forecast-difficulty probing on a
Time-Series Foundation Model). It is not a production service and ships no
network-facing components, but security reports are still taken seriously.

## Supported versions

Only the latest `main` is supported; fixes land there. There are no tagged
releases or backports.

## Reporting a vulnerability

For anything exploitable, please report **privately** rather than opening a
public issue:

- **Preferred:** GitHub private vulnerability reporting — the *"Report a
  vulnerability"* button under this repository's **Security** tab (Security
  Advisories).
- For non-sensitive, low-risk concerns, a regular issue is fine.

Please include reproduction steps, the affected file/commit, and the impact.
This is a solo-maintained academic project, so acknowledgement and fixes are
best-effort.

## Scope / notes

- Model checkpoints are loaded with `torch.load(..., weights_only=True)`; do not
  disable that when loading untrusted `.pt` files.
- Treat downloaded datasets and checkpoints (the ETTh1 CSV, Chronos weights, and
  trained SAE checkpoints) as untrusted input.
- The pinned dependency stack (`requirements.txt`) exists for bit-reproducibility,
  not security patching — review advisories before bumping in a deployment.
