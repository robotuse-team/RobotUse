"""Initialize the unchanged CGN HTTP service with this tool's pinned model.

The service owns serialization, retry sampling and GPU request serialization.
This adapter supplies repository-local imports, checkpoint paths and the same
service state that its original ``main`` initializes. No source is patched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys


TOOL_ROOT = Path(__file__).resolve().parent
CGN_SOURCE = TOOL_ROOT / "third_party/contact_graspnet"
SERVICE_SOURCE = TOOL_ROOT / "third_party/capx_service"


def verify_service_source():
    """Reject a modified snapshot before importing any GPU dependencies."""
    manifest = json.loads((TOOL_ROOT / "service_source.json").read_text())
    for record in manifest["files"]:
        path = SERVICE_SOURCE / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"CGN service original source was modified: {path}")
    return manifest


def create_service(*, device="cuda"):
    """Load the original trained model and return the original FastAPI app."""
    manifest = verify_service_source()
    revision = subprocess.check_output(
        ["git", "-C", str(CGN_SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != manifest["original_cgn_revision"]:
        raise RuntimeError("CGN checkout differs from the pinned model source")
    checkpoint_root = CGN_SOURCE / "checkpoints/contact_graspnet"
    checkpoint_dir = checkpoint_root / "checkpoints"
    for path in (checkpoint_root / "config.yaml", checkpoint_dir / "model.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"Pinned CGN asset is missing: {path}; initialize git submodules")
    sys.path[:0] = [str(SERVICE_SOURCE), str(CGN_SOURCE),
                   str(CGN_SOURCE / "Pointnet_Pointnet2_pytorch")]
    service = importlib.import_module("capx.serving.launch_contact_graspnet_server")
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

    estimator_source = Path(sys.modules[GraspEstimator.__module__].__file__).resolve()
    if not estimator_source.is_relative_to(CGN_SOURCE.resolve()):
        raise RuntimeError(f"CGN imported outside the pinned tool checkout: {estimator_source}")

    # These are the service lifecycle assignments performed by original main().
    service._DEVICE = device
    config = service.load_contact_graspnet_config(checkpoint_root)
    service._GRASP_ESTIMATOR = GraspEstimator(config)
    checkpoint = CheckpointIO(checkpoint_dir=str(checkpoint_dir), model=service._GRASP_ESTIMATOR.model)
    try:
        checkpoint.load("model.pt")
    except FileExistsError as exc:
        raise RuntimeError("No model checkpoint found; refusing untrained grasp inference") from exc
    return service.app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", default=8115, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    app = create_service(device=args.device)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
