# Third-party sources

External code is loaded from `third_party/`; RobotUse setup does not rewrite it.
Git submodules use pinned commits. Snapshot dependencies preserve recorded files
and licenses byte for byte. Their provenance records distinguish upstream
matches from pre-existing changes; a snapshot is not necessarily identical to
an upstream checkout. Snapshots do not install full upstream runtime environments.

| Dependency | Location | Management |
| --- | --- | --- |
| [RoboLab](https://github.com/NVlabs/RoboLab) | `src/simulator/robolab/third_party/robolab` | Pinned submodule |
| GAP connector | `src/simulator/robolab/third_party/graph_as_policy` | Existing connector and core snapshot |
| [Contact-GraspNet](https://github.com/uynitsuj/contact_graspnet_pytorch) | `src/tools/grasp/third_party/contact_graspnet` | Pinned submodule |
| CaP-X CGN service | `src/tools/grasp/third_party/capx_service` | Six preserved service files, recorded in `service_source.json` |
| Open Robot Skills geometry helpers | `src/tools/grasp/third_party/open_robot_skills` | Geometry helpers and LICENSE snapshot |
| [cuRobo](https://github.com/NVlabs/curobo) | `src/tools/curobo/third_party/curobo` | Pinned submodule, disabled by default |
| Open Robot Skills cuRobo wrapper | `src/tools/curobo/third_party/open_robot_skills` | Two preserved files, recorded in `implementation_source.json` |
| [SAM2](https://github.com/facebookresearch/sam2) | `src/tools/perception/third_party/sam2` | Pinned official submodule |
| robosuite Panda assets | `src/tools/pose_editor/third_party/robosuite` | Snapshot of XML, five meshes, and LICENSE |
| [TiPToP](https://github.com/tiptop-robot/tiptop) | `src/tools/pose_editor/third_party/tiptop` | Original depth-conversion adapter source and LICENSE |

Use `git clone --recurse-submodules` and then `scripts/setup/sources.sh`, or run
the script after an ordinary clone. Submodules record exact commits; recursive
initialization downloads those commits and their nested dependencies. The script
checks existing sources before updating, pulls RoboLab LFS assets, and verifies
all submodule revisions and snapshot hashes. It does not update to upstream HEAD.
Use `python3 scripts/check/setup.py --sources-only` for a read-only verification.
Snapshot sources, scope, and SHA-256 hashes are recorded in
[DEPENDENCIES.json](DEPENDENCIES.json). The
[CGN service manifest](src/tools/grasp/service_source.json) and
[cuRobo wrapper manifest](src/tools/curobo/implementation_source.json) also record
public upstream mappings and pre-existing differences. Recorded hashes identify
the shipped files; historical lineage alone does not establish an upstream match.

## Dependency and asset licenses

RobotUse's root Apache-2.0 license does not replace upstream terms for code,
checkpoints, or assets. The following summaries refer to the pinned sources;
their full license texts and notices govern.

- **Contact-GraspNet:** the pinned PyTorch implementation carries the
  [NVIDIA Source Code License](https://github.com/uynitsuj/contact_graspnet_pytorch/blob/da3dcfb2f53e43b186083ee4a9d1e232f73efc98/License.pdf).
  Section 3.3 limits use to noncommercial research or evaluation. Bundled
  Pointnet code has separate licenses. The bundled `model.pt` has no separate
  checkpoint-specific license statement identified in the pinned README or
  checkpoint directory; its inclusion and checksum do not establish a separate
  grant of commercial rights. Checkpoint-specific permission remains unverified.
- **SAM2 checkpoints:** the pinned upstream
  [license statement](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md#license)
  explicitly covers the model checkpoints under Apache-2.0.
- **RoboLab assets:** the framework is Apache-2.0, but its
  [asset notices](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/THIRD_PARTY_NOTICES.md#bundled-assets)
  assign CC BY-NC-SA 4.0 to HANDAL, HOPE, `basic`, `fruits_veggies`, `objaverse`,
  fixtures, materials, scenes, and robots. Asset-local licenses take precedence;
  for example, Kinova/Robotiq robot assets are BSD-3-Clause. HOT3D models have
  [CC BY-SA 4.0 plus the dataset agreement's non-sale restriction](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/assets/objects/hot3d/LICENSE).
  Consult the same notices for YCB, VOMP, and background terms. Exporting or
  converting assets does not remove their conditions: redistributed derived
  meshes need the source attribution, license notices, and modification details,
  with noncommercial and ShareAlike conditions where applicable.

GAP's preserved `NOTICE.md` describes an MIT license, while its `LICENSE` and
the Open Robot Skills `LICENSE` files contain Apache-2.0 text. These originals
are preserved without resolving the discrepancy or granting additional rights.
