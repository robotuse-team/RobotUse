"""Load the unmodified official SAM2 source and adapt its point predictor."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SAM2_SOURCE = Path(__file__).resolve().parent / "third_party" / "sam2"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
DEFAULT_SNAPSHOT = Path(os.environ.get("SAM2_CHECKPOINT",
    Path(os.environ.get("ROBOTUSE_RUNTIME_ROOT", REPOSITORY_ROOT / "runtime")) / "model-cache/sam2"))
DEFAULT_PYTHON = Path(__file__).resolve().parent / "python.sh"
ENVIRONMENT_PYTHON = REPOSITORY_ROOT / "runtime" / "tool-envs" / "sam2" / "bin" / "python"


def validate_python(python: str | Path) -> None:
    """Fail before simulator startup if the selected worker cannot be launched."""
    launcher = Path(python).expanduser()
    required = [launcher]
    if launcher.resolve() == DEFAULT_PYTHON.resolve():
        runtime = os.environ.get("ROBOTUSE_RUNTIME_ROOT")
        default_python = Path(runtime) / "tool-envs/sam2/bin/python" if runtime else ENVIRONMENT_PYTHON
        environment = os.environ.get("SAM2_ENVIRONMENT")
        default_python = Path(environment) / "bin/python" if environment else default_python
        required.append(Path(os.environ.get("ROBOTUSE_SAM_PYTHON",
            os.environ.get("SAM_RUNTIME_PYTHON", default_python))))
    missing = [str(path) for path in required if not path.is_file() or not os.access(path, os.X_OK)]
    if missing:
        raise FileNotFoundError("Official SAM2 worker Python unavailable; run scripts/setup/sam2.sh:\n" + "\n".join(missing))


def resolve_checkpoint(snapshot: str | Path) -> Path:
    """Accept a provisioned checkpoint directory or an explicit official .pt file."""
    path = Path(snapshot).expanduser().resolve()
    checkpoint = path / CHECKPOINT_NAME if path.is_dir() else path
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Official SAM2 checkpoint missing: {checkpoint}; provision {CHECKPOINT_NAME} explicitly"
        )
    if checkpoint.suffix != ".pt":
        raise ValueError("Official SAM2 requires a .pt checkpoint, not Transformers weights")
    return checkpoint


def validate_source() -> Path:
    """Require this tool's upstream checkout and reject another loaded sam2 package."""
    source = SAM2_SOURCE.resolve()
    package = source / "sam2"
    required = [package / name for name in (
        "__init__.py", "build_sam.py", "sam2_image_predictor.py", MODEL_CONFIG,
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Official SAM2 submodule is incomplete:\n" + "\n".join(missing))
    for name, module in tuple(sys.modules.items()):
        if name != "sam2" and not name.startswith("sam2."):
            continue
        location = getattr(module, "__file__", None)
        if location is None and name != "sam2":
            continue  # Upstream configuration packages may be namespace packages.
        if location is None or not Path(location).resolve().is_relative_to(package):
            raise RuntimeError(f"SAM2 must load from {package}; {name} is already loaded from {location}")
    return source


def create_predictor(checkpoint: str | Path, device: str) -> Any:
    """Build the official image predictor without extra mask cleanup or autocast."""
    source = validate_source()
    checkpoint = resolve_checkpoint(checkpoint)
    sys.path[:] = [entry for entry in sys.path if entry != str(source)]
    sys.path.insert(0, str(source))
    importlib.import_module("sam2")
    validate_source()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    validate_source()
    model = build_sam2(
        MODEL_CONFIG, ckpt_path=str(checkpoint), device=device,
        mode="eval", apply_postprocessing=False,
    )
    model = model.to(device=device, dtype=torch.float32).eval()
    return SAM2ImagePredictor(
        model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0,
    )


def predict_masks(predictor: Any, image: Any, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """Pass the RGB image and one positive pixel prompt to the official predictor."""
    import torch

    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray([[x, y]], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
            return_logits=False,
            normalize_coords=True,
        )
    return np.asarray(masks, dtype=bool), np.asarray(scores, dtype=np.float32).reshape(-1)
