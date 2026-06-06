"""TopK SAE — now sourced from the shared fm-difficulty-probe core.

Both legacy repos shipped byte-identical copies of this module; it has been
promoted to `core.sae` in fm-difficulty-probe (installed editable, see the
README "Running" section: `pip install -e ../fm-difficulty-probe --no-deps`).
This thin re-export keeps the historical import path `from sae_model import
TopKSAE` working for train_sae.py / probe.py / eval/* / tests while the single
implementation lives in one place.

The core version is a strict superset of the old local class: same forward
signature `(acts, x_reconstruct, aux_loss)`, same `normalize_decoder`, plus an
`expansion=` constructor knob and a `from_checkpoint` classmethod.
"""
from core.sae import TopKSAE

__all__ = ["TopKSAE"]
