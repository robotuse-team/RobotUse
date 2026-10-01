"""Official SAM2 integration contracts without model, GPU, or network calls."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.tools.perception import rgbd_adapter, sam2_adapter


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(tmp_path):
    snapshot = tmp_path / "sam2"
    snapshot.mkdir()
    checkpoint = snapshot / sam2_adapter.CHECKPOINT_NAME
    checkpoint.write_bytes(b"file-presence fixture, never loaded")
    return checkpoint


def test_checkpoint_directory_and_file_resolve_to_same_local_weights(checkpoint):
    assert sam2_adapter.resolve_checkpoint(checkpoint.parent) == checkpoint.resolve()
    assert sam2_adapter.resolve_checkpoint(checkpoint) == checkpoint.resolve()


def test_transformers_only_snapshot_fails_without_download_or_fallback(tmp_path):
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(FileNotFoundError):
        sam2_adapter.resolve_checkpoint(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {"model.safetensors"}


def test_default_launcher_requires_executable_environment_python(tmp_path, monkeypatch):
    for name in ('ROBOTUSE_SAM_PYTHON', 'SAM_RUNTIME_PYTHON', 'SAM2_ENVIRONMENT', 'ROBOTUSE_RUNTIME_ROOT'):
        monkeypatch.delenv(name, raising=False)
    launcher = tmp_path / "python.sh"
    launcher.write_text("#!/bin/sh\nexit 97\n")
    launcher.chmod(0o755)
    interpreter = tmp_path / "environment-python"
    monkeypatch.setattr(sam2_adapter, "DEFAULT_PYTHON", launcher)
    monkeypatch.setattr(sam2_adapter, "ENVIRONMENT_PYTHON", interpreter)
    with pytest.raises(FileNotFoundError, match="environment-python"):
        sam2_adapter.validate_python(launcher)
    interpreter.write_text("#!/bin/sh\nexit 98\n")
    interpreter.chmod(0o644)
    with pytest.raises(FileNotFoundError, match="environment-python"):
        sam2_adapter.validate_python(launcher)
    interpreter.chmod(0o755)
    sam2_adapter.validate_python(launcher)


def test_custom_launcher_requires_execute_permission_without_default_environment(tmp_path, monkeypatch):
    launcher = tmp_path / "custom-python"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o644)
    monkeypatch.setattr(sam2_adapter, "ENVIRONMENT_PYTHON", tmp_path / "missing-environment")
    with pytest.raises(FileNotFoundError, match="custom-python"):
        sam2_adapter.validate_python(launcher)
    launcher.chmod(0o755)
    sam2_adapter.validate_python(launcher)


def test_runtime_checks_official_checkpoint_before_starting_models(checkpoint, monkeypatch):
    from src.runtime.configuration import validate_runtime

    monkeypatch.setitem(sys.modules, "torch", None)
    args = SimpleNamespace(sam_python=Path(sys.executable), sam2_snapshot=checkpoint.parent)
    configuration = SimpleNamespace(client=lambda: None)
    validate_runtime(args, configuration)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        validate_runtime(args, configuration)


def test_missing_official_source_fails_before_loading_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(sam2_adapter, "SAM2_SOURCE", tmp_path / "uninitialized-submodule")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(FileNotFoundError):
        sam2_adapter.validate_source()


def test_already_imported_foreign_sam2_is_not_silently_reused(tmp_path, monkeypatch):
    foreign = SimpleNamespace(__file__=str(tmp_path / "sam2" / "__init__.py"))
    monkeypatch.setitem(sys.modules, "sam2", foreign)
    with pytest.raises(RuntimeError, match="SAM2 must load from"):
        sam2_adapter.validate_source()


def test_model_loader_uses_official_large_config_without_extra_mask_cleanup(checkpoint, monkeypatch):
    source = sam2_adapter.SAM2_SOURCE / "sam2"
    calls = {}
    float32 = object()

    class Model:
        def to(self, **kwargs):
            calls["to"] = kwargs
            return self

        def eval(self):
            calls["eval"] = True
            return self

    model = Model()

    def build(config, **kwargs):
        calls["build"] = config, kwargs
        return model

    def predictor(model_arg, **kwargs):
        calls["predictor"] = model_arg, kwargs
        return "official predictor"

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=float32))
    monkeypatch.setitem(sys.modules, "sam2", SimpleNamespace(__file__=str(source / "__init__.py")))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", SimpleNamespace(
        __file__=str(source / "build_sam.py"), build_sam2=build))
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", SimpleNamespace(
        __file__=str(source / "sam2_image_predictor.py"), SAM2ImagePredictor=predictor))
    monkeypatch.setattr(sys, "path", ["/some/other/sam2/environment", *sys.path,
                                      str(sam2_adapter.SAM2_SOURCE.resolve())])
    assert sam2_adapter.create_predictor(checkpoint, "cpu") == "official predictor"
    assert sys.path[0] == str(sam2_adapter.SAM2_SOURCE.resolve())
    assert sys.path.count(str(sam2_adapter.SAM2_SOURCE.resolve())) == 1
    assert calls["build"] == ("configs/sam2.1/sam2.1_hiera_l.yaml", {
        "ckpt_path": str(checkpoint.resolve()), "device": "cpu", "mode": "eval",
        "apply_postprocessing": False,
    })
    assert calls["to"] == {"device": "cpu", "dtype": float32}
    assert calls["eval"] is True
    assert calls["predictor"] == (model, {
        "mask_threshold": 0.0, "max_hole_area": 0.0, "max_sprinkle_area": 0.0,
    })


def test_importing_perception_adapter_does_not_load_model_runtimes():
    code = """
import sys
from src.tools.perception import sam2_adapter, rgbd_adapter
assert not {'torch', 'torchvision', 'transformers', 'sam2', 'groundingdino'} & sys.modules.keys()
assert 'third_party' in sam2_adapter.SAM2_SOURCE.parts
assert sam2_adapter.validate_source().resolve() == sam2_adapter.SAM2_SOURCE.resolve()
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True,
                   capture_output=True, text=True)


class FakePredictor:
    """Records what the model receives and returns distinct candidate masks."""

    def __init__(self):
        self.images = []
        self.prompts = []
        self.fail_next = False
        self.contain_point = True

    def set_image(self, image):
        self.images.append(np.asarray(image).copy())

    def predict(self, **prompt):
        self.prompts.append(prompt)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("deliberate model failure")
        height, width = self.images[-1].shape[:2]
        masks = np.zeros((3, height, width), dtype=bool)
        masks[0, 0, 0] = True  # Highest score does not contain the requested point.
        if self.contain_point:
            masks[1, 1:3, 1:3] = True
            masks[2, 1:4, 1:4] = True
        return masks, np.array([.95, .75, .55], dtype=np.float32), np.empty((3, 256, 256))


@pytest.fixture
def fake_model(monkeypatch):
    predictor = FakePredictor()
    loads = []

    def create(checkpoint, device):
        loads.append((checkpoint, device))
        return predictor

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(sam2_adapter, "create_predictor", create)
    monkeypatch.setattr(rgbd_adapter, "_MODEL_CACHE", {})
    return predictor, loads


def request(tmp_path, snapshot, *, name="first", color=(12, 34, 56)):
    image = tmp_path / f"{name}.png"
    Image.new("RGB", (5, 4), color).save(image)
    return SimpleNamespace(snapshot=str(snapshot), image=str(image),
                           output=str(tmp_path / f"{name}-mask.png"), x=1, y=2, device="cpu")


def assert_positive_pixel_prompt(prompt):
    np.testing.assert_array_equal(prompt["point_coords"], [[1, 2]])
    np.testing.assert_array_equal(prompt["point_labels"], [1])
    assert prompt["multimask_output"] is True
    assert prompt["return_logits"] is False
    assert prompt["normalize_coords"] is True


def test_worker_keeps_best_point_bound_mask_and_png_json_contract(checkpoint, tmp_path, fake_model):
    predictor, loads = fake_model
    args = request(tmp_path, checkpoint.parent)
    rgbd_adapter._worker(args)

    mask = np.asarray(Image.open(args.output))
    expected = np.zeros((4, 5), dtype=np.uint8)
    expected[1:3, 1:3] = 255
    np.testing.assert_array_equal(mask, expected)
    assert Image.open(args.output).mode == "L"
    sidecar = json.loads(Path(args.output).with_suffix(".json").read_text())
    assert sidecar == {"model": str(checkpoint.parent), "prompt": "positive-point", "score": .75}
    assert loads == [(checkpoint.resolve(), "cpu")]
    np.testing.assert_array_equal(predictor.images[0], np.array(Image.open(args.image)))
    assert_positive_pixel_prompt(predictor.prompts[0])


def test_worker_reuses_weights_but_replaces_image_for_every_request(checkpoint, tmp_path, fake_model):
    predictor, loads = fake_model
    first = request(tmp_path, checkpoint.parent)
    second = request(tmp_path, checkpoint, name="second", color=(201, 202, 203))
    rgbd_adapter._worker(first)
    rgbd_adapter._worker(second)

    assert len(loads) == 1
    assert len(predictor.images) == len(predictor.prompts) == 2
    assert predictor.images[0][0, 0].tolist() == [12, 34, 56]
    assert predictor.images[1][0, 0].tolist() == [201, 202, 203]
    assert all(Path(args.output).is_file() for args in (first, second))


def test_no_point_bound_mask_preserves_error_without_writing_output(checkpoint, tmp_path, fake_model):
    predictor, _ = fake_model
    predictor.contain_point = False
    args = request(tmp_path, checkpoint.parent)
    with pytest.raises(RuntimeError, match="^SAM2 returned no mask containing the point$"):
        rgbd_adapter._worker(args)
    assert not Path(args.output).exists()
    assert not Path(args.output).with_suffix(".json").exists()


def test_failed_inference_does_not_poison_cached_predictor(checkpoint, tmp_path, fake_model):
    predictor, loads = fake_model
    predictor.fail_next = True
    first = request(tmp_path, checkpoint.parent)
    with pytest.raises(RuntimeError, match="deliberate model failure"):
        rgbd_adapter._worker(first)
    assert not Path(first.output).exists()
    second = request(tmp_path, checkpoint.parent, name="retry", color=(101, 102, 103))
    rgbd_adapter._worker(second)
    assert len(loads) == 1
    assert predictor.images[-1][0, 0].tolist() == [101, 102, 103]
    assert Path(second.output).is_file()
