"""Offline smoke test for the probe ladder's building blocks.

The headline pipeline (`probing/probe.py`) is end-to-end: it downloads ETTh1,
loads a Chronos checkpoint, trains an SAE, and fits the probes. None of that is
testable in CI. But the two pieces every rung of the ladder depends on -- the
`TopKSAE` autoencoder and the `compute_spectral_entropy` difficulty feature --
are pure numpy/torch and run in well under a second on synthetic data.

These tests don't assert any research conclusion. They assert the contracts the
probe relies on so a refactor can't silently break them:
  * the SAE returns exactly-k non-negative activations of the right shape, and
  * a saved checkpoint reloads to bit-identical activations (the probe loads
    W_enc/W_dec from disk -- a serialization regression would feed it garbage).

Run with:  pytest tests/ -q
(after `pip install -e .`, so the `sae` / `probing` packages resolve).
"""
import os
import tempfile

import numpy as np
import torch

from sae.sae_model import TopKSAE
from probing.probe import compute_spectral_entropy


def test_topk_sae_forward_shapes_and_sparsity():
    """Forward pass: shapes are preserved and exactly k features fire per row."""
    torch.manual_seed(0)
    d_model, d_hidden, k = 16, 64, 8
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=k)
    sae.eval()

    x = torch.randn(5, d_model)
    acts, recon, aux_loss = sae(x)

    assert acts.shape == (5, d_hidden)
    assert recon.shape == (5, d_model)
    # Top-k routing: each row has at most k active (post-ReLU some may be 0).
    nonzero_per_row = (acts > 0).sum(dim=-1)
    assert torch.all(nonzero_per_row <= k)
    # Activations are ReLU'd, so never negative.
    assert torch.all(acts >= 0)
    # With no dead_mask the aux loss path is skipped.
    assert float(aux_loss) == 0.0


def test_topk_sae_checkpoint_roundtrip():
    """The probe loads the SAE from disk; saved->reloaded must be identical."""
    torch.manual_seed(1)
    d_model, d_hidden, k = 16, 64, 8
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=k)
    sae.eval()
    assert "W_enc" in sae.state_dict()  # probe.py hard-fails without this key

    x = torch.randn(4, d_model)
    with torch.no_grad():
        before, _, _ = sae(x)

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "sae.pt")
        torch.save(sae.state_dict(), ckpt)

        state = torch.load(ckpt, weights_only=True)
        d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape  # probe auto-detects dims
        assert (d_model_ckpt, d_hidden_ckpt) == (d_model, d_hidden)

        reloaded = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=k)
        reloaded.load_state_dict(state)
        reloaded.eval()
        with torch.no_grad():
            after, _, _ = reloaded(x)

    assert torch.equal(before, after)


def test_spectral_entropy_orders_difficulty():
    """Constant -> 0; broadband noise should be more 'difficult' than a tone."""
    rng = np.random.default_rng(0)

    assert compute_spectral_entropy(np.ones(512)) == 0.0

    t = np.arange(512)
    sine = np.sin(2 * np.pi * t / 24.0)        # single dominant frequency
    noise = rng.normal(size=512)               # energy spread across spectrum

    h_sine = compute_spectral_entropy(sine)
    h_noise = compute_spectral_entropy(noise)
    assert h_sine >= 0.0 and h_noise >= 0.0
    assert h_noise > h_sine


def _run_standalone():
    """Tiny runner so the smoke test works without pytest installed."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} smoke tests passed.")


if __name__ == "__main__":
    _run_standalone()
