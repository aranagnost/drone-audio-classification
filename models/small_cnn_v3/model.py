# models/small_cnn_v3/model.py
"""
SmallCNNv3 — same architecture as v2 (ResBlocks + SE attention).

The v3 improvement is entirely in the training pipeline:
  - Waveform-level noise mixing (speech, wind, etc. from not_a_drone)
  - Gain jitter
  - SpecAugment (carried over from v2)

The model architecture is re-exported from v2 to avoid duplication.
"""
import importlib.util
from pathlib import Path

# Load SmallCNNv2 from sibling v2 directory without polluting sys.path
_V2_MODEL = Path(__file__).resolve().parent.parent / "small_cnn_v2" / "model.py"
_spec = importlib.util.spec_from_file_location("small_cnn_v2_model", str(_V2_MODEL))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SmallCNNv2 = _mod.SmallCNNv2

# Alias so training scripts and model.json can reference SmallCNNv3
SmallCNNv3 = SmallCNNv2
