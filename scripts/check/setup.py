#!/usr/bin/env python3
"""Check source pins, dependency locks and model assets without calling an LLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {'cpu': '3.11.9', 'ui': '3.11.9', 'sam2': '3.11.9', 'cgn': '3.11.9',
            'robolab': '3.11.13', 'curobo': '3.11.13'}
SOURCES = (
    'src/tools/perception/third_party/sam2',
    'src/tools/grasp/third_party/contact_graspnet',
    'src/tools/curobo/third_party/curobo',
    'src/simulator/robolab/third_party/robolab',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=ROOT if cwd is None else cwd,
                                   text=True, stderr=subprocess.PIPE).strip()


def gitlinks(directory):
    entries = command('git', 'ls-tree', '-rz', 'HEAD', cwd=directory)
    result = {}
    for entry in entries.split('\0'):
        if entry:
            metadata, relative = entry.split('\t', 1)
            mode, kind, revision = metadata.split()
            if mode == '160000' and kind == 'commit':
                result[relative] = revision
    return result


def check_submodules(*, before_update=False):
    result = {}
    pins = gitlinks(ROOT)
    for relative in SOURCES:
        if relative not in pins:
            raise ValueError(f'{relative}: missing committed submodule pin')
    if before_update:
        if command('git', 'diff', 'HEAD', '--', '.gitmodules'):
            raise ValueError('.gitmodules has uncommitted changes; sources were left unchanged')
        staged = command('git', 'diff', '--cached', '--name-only', '--ignore-submodules=none',
                         'HEAD', '--', *pins)
        if staged:
            raise ValueError('Submodule pins have staged changes; sources were left unchanged')

    def visit(directory, expected):
        relative = directory.relative_to(ROOT).as_posix()
        if not (directory / '.git').exists():
            if before_update and (not directory.exists() or (directory.is_dir() and not any(directory.iterdir()))):
                return
            raise ValueError(f'{relative}: missing submodule checkout; run scripts/setup/sources.sh')
        actual = command('git', 'rev-parse', 'HEAD', cwd=directory)
        if not before_update and actual != expected:
            raise ValueError(f'{relative}: wrong source revision; run scripts/setup/sources.sh')
        changed_files = command('git', 'status', '--porcelain', '--untracked-files=all',
                                '--ignore-submodules=all', cwd=directory)
        staged_changes = command('git', 'diff', '--cached', '--name-only',
                                 '--ignore-submodules=none', 'HEAD', cwd=directory)
        if changed_files or staged_changes:
            raise ValueError(f'{relative}: upstream checkout is not clean; sources were left unchanged')
        result[relative] = actual
        for child, revision in gitlinks(directory).items():
            visit(directory / child, revision)

    for relative, revision in pins.items():
        visit(ROOT / relative, revision)
    return result


def check_sources(*, before_update=False):
    result = check_submodules(before_update=before_update)
    snapshots = json.loads((ROOT / 'DEPENDENCIES.json').read_text())['sha256']
    for tool, filename, subdirectory in (
        ('grasp', 'service_source.json', 'capx_service'),
        ('curobo', 'implementation_source.json', 'open_robot_skills'),
    ):
        directory = ROOT / 'src/tools' / tool
        for row in json.loads((directory / filename).read_text())['files']:
            snapshots[str((directory / 'third_party' / subdirectory / row['path']).relative_to(ROOT))] = row['sha256']
    for relative, expected in snapshots.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Original snapshot changed: {relative}')
    return {'submodules': result, 'verified_snapshot_files': len(snapshots)}


def check_robolab_assets():
    source = ROOT / SOURCES[-1]
    output = command('git', 'lfs', 'ls-files', '-l', cwd=source)
    rows = [re.fullmatch(r'([0-9a-f]{64}) ([*-]) (.+)', line) for line in output.splitlines()]
    if not rows or not all(rows):
        raise ValueError('Unable to enumerate RoboLab assets; install Git LFS and run scripts/setup/sources.sh')
    for row in rows:
        expected, materialized, relative = row.groups()
        path = source / relative
        if materialized != '*' or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Missing or invalid RoboLab LFS asset: {relative}; run scripts/setup/sources.sh')
    return {'sha256_verified_lfs_files': len(rows)}


def check_models(runtime, profiles):
    manifest = json.loads((ROOT / 'configs/checkpoints.json').read_text())
    result = {}
    if 'sam2' in profiles:
        item = manifest['sam2']
        path = Path(os.environ.get('SAM2_CHECKPOINT', runtime / item['runtime_path']))
        if sha256(path) != item['sha256']:
            raise ValueError(f'SAM2 checkpoint SHA-256 mismatch: {path}')
        result['sam2'] = {'path': str(path), 'sha256': item['sha256']}
    if 'cgn' in profiles:
        for relative, expected in manifest['contact_graspnet']['files'].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f'CGN checkpoint/config SHA-256 mismatch: {relative}')
        result['cgn'] = {'sha256_verified_files': 2}
    return result


def check_environment(runtime, profile, gpu):
    expected = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;\\]+)',
                              (ROOT / 'requirements' / f'{profile}.txt').read_text(), re.MULTILINE))
    if not expected:
        raise ValueError(f'Empty dependency lock for {profile}')
    override = {'cpu': 'CPU_ENVIRONMENT', 'ui': 'UI_ENVIRONMENT', 'sam2': 'SAM2_ENVIRONMENT',
                'curobo': 'CUROBO_ENVIRONMENT'}.get(profile)
    environment = Path(os.environ.get(override, runtime / 'tool-envs' / profile)) if override else runtime / 'tool-envs' / profile
    python = environment / 'bin/python'
    interpreter_overrides = {
        'ui': ('ROBOTUSE_UI_PYTHON',),
        'sam2': ('ROBOTUSE_SAM_PYTHON', 'SAM_RUNTIME_PYTHON'),
        'cgn': ('ROBOTUSE_CGN_PYTHON',),
        'curobo': ('ROBOTUSE_CUROBO_PYTHON',),
        'robolab': ('ROBOLAB_PYTHON',),
    }
    for name in interpreter_overrides.get(profile, ()):
        if os.environ.get(name):
            python = Path(os.environ[name]).expanduser()
            break
    code = '''
import importlib.metadata as metadata
import json, platform, sys
expected, gpu = json.load(sys.stdin)
mismatches = {}
for name, version in expected.items():
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        actual = None
    if actual != version:
        mismatches[name] = {'expected': version, 'actual': actual}
result = {'python': platform.python_version(), 'interpreter': sys.executable,
          'virtual_environment': sys.prefix != sys.base_prefix,
          'packages_checked': len(expected), 'mismatches': mismatches}
if gpu:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this interpreter')
    result['gpu'] = {'name': torch.cuda.get_device_name(0), 'torch_cuda': torch.version.cuda,
                     'capability': list(torch.cuda.get_device_capability(0))}
print(json.dumps(result))
'''
    excluded = {'PYTHONPATH', 'PYTHONHOME', 'PYTHONEXE', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV'}
    env = {key: value for key, value in os.environ.items() if key not in excluded}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    gpu_check = gpu and profile not in {'cpu', 'ui'}
    if gpu_check:
        name = 'ROBOTUSE_CGN_GPU' if profile == 'cgn' else 'ROBOTUSE_GPU'
        physical_gpu = os.environ.get(name) or '0'
        if not physical_gpu.isdigit():
            raise ValueError('Select one physical GPU index with ROBOTUSE_GPU or ROBOTUSE_CGN_GPU')
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=physical_gpu)
    process = subprocess.run([str(python), '-c', code], input=json.dumps([expected, gpu_check]),
                             capture_output=True, text=True, env=env, timeout=120)
    if process.returncode:
        raise RuntimeError(f'{profile} interpreter check failed: {process.stderr.strip()}')
    result = json.loads(process.stdout)
    if gpu_check:
        result['gpu']['physical_index'] = int(physical_gpu)
    if not result['virtual_environment'] or result['python'] != PROFILES[profile] or result['mismatches']:
        raise ValueError(f'{profile} differs from its pinned environment: {result}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=os.environ.get('ROBOTUSE_RUNTIME_ROOT', ROOT / 'runtime'))
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=['cpu', 'sam2', 'cgn', 'robolab', 'ui'])
    parser.add_argument('--gpu', action='store_true', help='Also check CUDA availability in the selected interpreters')
    source_mode = parser.add_mutually_exclusive_group()
    source_mode.add_argument('--sources-only', action='store_true',
                             help='Check pinned sources, snapshots and RoboLab LFS assets without virtual environments')
    source_mode.add_argument('--source-preflight', action='store_true',
                             help='Check existing sources before downloading missing or updated submodules')
    args = parser.parse_args(argv)
    result = {'scope': 'Installation readiness, not task success or deterministic output equivalence',
              'llm_calls': 0, 'checks': {}, 'errors': []}
    if args.source_preflight:
        result['scope'] = 'Existing source integrity before download; missing submodules are allowed'
        checks = {'sources': lambda: check_sources(before_update=True)}
    elif args.sources_only:
        result['scope'] = 'Source and RoboLab asset integrity only; runtime environments are not checked'
        checks = {'sources': check_sources, 'robolab_assets': check_robolab_assets}
    else:
        checks = {'sources': check_sources,
                  'checkpoints': lambda: check_models(args.runtime_root, args.profiles)}
        if 'robolab' in args.profiles:
            checks['robolab_assets'] = check_robolab_assets
        for profile in dict.fromkeys(args.profiles):
            checks[profile] = lambda profile=profile: check_environment(args.runtime_root, profile, args.gpu)
    for name, check in checks.items():
        try:
            result['checks'][name] = check()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result['errors'].append({'check': name, 'error': str(error)})
    result['ready'] = not result['errors']
    print(json.dumps(result, indent=2))
    return 0 if result['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
