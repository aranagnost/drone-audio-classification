# models/3_stages_cnn_v1/model.py
"""
ThreeStagesCNNv1 — same architecture as SmallCNNv2 (ResBlocks + SE attention).

The improvement here is the 3-stage cascade pipeline:
  - Stage 1: drone vs no_drone (binary)
  - Stage 2: coarse motor classification — 2_motors / 4or6_motors / 8_motors (3-class)
  - Stage 3: fine motor classification — 4_motors vs 6_motors (binary, runs only
             when stage 2 is confident about the 4or6_motors class)

This avoids forcing the model to separate acoustically-similar 4/6 motor classes
in stage 2, then dedicates a focused binary classifier to that hard distinction.

The architecture is re-exported from v2 to avoid duplication.
"""
import importlib.util
from pathlib import Path

# Load SmallCNNv2 from sibling v2 directory without polluting sys.path
_V2_MODEL = Path(__file__).resolve().parent.parent / "small_cnn_v2" / "model.py"
_spec = importlib.util.spec_from_file_location("small_cnn_v2_model", str(_V2_MODEL))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SmallCNNv2 = _mod.SmallCNNv2

# Alias so training scripts can reference ThreeStagesCNNv1
ThreeStagesCNNv1 = SmallCNNv2
