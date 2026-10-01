#!/usr/bin/env python3
"""Export Panda visual meshes in the visual-pose annotation frame, without simulation.

Output uses indexed triangles with normals, in metres, ready for WebGL. Geometry
comes exclusively from XML group=1 visual meshes. This export does not establish
contact, collision freedom, reachability, or an executable robot pose.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tools.pose_editor.mesh_assets import DEFAULT_ASSETS, export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jaw-separation", type=float, default=.065)
    args = parser.parse_args()
    export(args.assets_root.resolve(), args.output.resolve(), args.jaw_separation)


if __name__ == "__main__":
    main()
