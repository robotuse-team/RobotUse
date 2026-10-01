"""Pinned source setup must preserve originals and validate nested dependencies."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = 'tool/third_party/package'
NESTED = f'{MODULE}/nested'


def git(directory, *arguments):
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    return subprocess.check_output(
        ['git', '-c', 'protocol.file.allow=always', '-c', 'user.name=Source tests',
         '-c', 'user.email=source-tests@example.invalid', '-C', str(directory), *arguments],
        text=True, stderr=subprocess.PIPE, env=env,
    ).strip()


def repository(path):
    path.mkdir()
    git(path, 'init', '-b', 'main')
    return path


def commit_file(directory, name, content):
    destination = directory / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    git(directory, 'add', name)
    git(directory, 'commit', '-m', 'Update fixture source')
    return git(directory, 'rev-parse', 'HEAD')


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    child = repository(tmp_path / 'child-origin')
    child_revision = commit_file(child, 'source.txt', 'pinned child\n')
    package = repository(tmp_path / 'package-origin')
    commit_file(package, 'source.txt', 'pinned package\n')
    git(package, 'submodule', 'add', str(child), 'nested')
    git(package, 'commit', '-am', 'Pin nested dependency')
    package_revision = git(package, 'rev-parse', 'HEAD')
    origin = repository(tmp_path / 'origin')
    snapshot = 'src/tools/grasp/third_party/snapshot/source.txt'
    commit_file(origin, snapshot, 'preserved snapshot\n')
    manifest = {'sha256': {snapshot: hashlib.sha256((origin / snapshot).read_bytes()).hexdigest()}}
    (origin / 'DEPENDENCIES.json').write_text(json.dumps(manifest))
    for tool, filename in [('grasp', 'service_source.json'), ('curobo', 'implementation_source.json')]:
        target = origin / 'src/tools' / tool / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({'files': []}))
    git(origin, 'submodule', 'add', str(package), MODULE)
    git(origin, 'add', '.')
    git(origin, 'commit', '-m', 'Pin source dependencies')
    checkout = tmp_path / 'checkout'
    git(tmp_path, 'clone', str(origin), str(checkout))
    child_latest = commit_file(child, 'source.txt', 'newer child\n')
    package_latest = commit_file(package, 'source.txt', 'newer package\n')
    spec = importlib.util.spec_from_file_location('source_setup_check', ROOT / 'scripts/check/setup.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    monkeypatch.setattr(checker, 'ROOT', checkout)
    monkeypatch.setattr(checker, 'SOURCES', (MODULE,))
    return SimpleNamespace(
        root=checkout, checker=checker, snapshot=snapshot, package_origin=package,
        package_revision=package_revision, package_latest=package_latest,
        child_revision=child_revision, child_latest=child_latest,
    )


def initialize(tree):
    git(tree.root, 'submodule', 'update', '--init', '--recursive', '--checkout')


def ignore_submodules(tree):
    git(tree.root, 'config', f'submodule.{MODULE}.ignore', 'all')
    git(tree.root / MODULE, 'config', 'submodule.nested.ignore', 'all')


def test_fresh_clone_initializes_recursive_pins_despite_newer_remote_branches(source_tree):
    tree = source_tree
    tree.checker.check_sources(before_update=True)
    with pytest.raises(ValueError):
        tree.checker.check_sources()
    initialize(tree)
    result = tree.checker.check_sources()
    assert result['submodules'] == {
        MODULE: tree.package_revision,
        NESTED: tree.child_revision,
    }
    assert result['verified_snapshot_files'] == 1
    assert git(tree.root / MODULE, 'rev-parse', 'HEAD') != tree.package_latest
    assert git(tree.root / NESTED, 'rev-parse', 'HEAD') != tree.child_latest
    tree.checker.check_sources(before_update=True)


@pytest.mark.parametrize('relative', [MODULE, NESTED])
def test_clean_wrong_revision_can_be_repaired_to_the_pin(source_tree, relative):
    tree = source_tree
    initialize(tree)
    ignore_submodules(tree)
    latest = tree.package_latest if relative == MODULE else tree.child_latest
    git(tree.root / relative, 'checkout', '--detach', latest)
    tree.checker.check_submodules(before_update=True)
    with pytest.raises(ValueError):
        tree.checker.check_submodules()
    initialize(tree)
    assert tree.checker.check_submodules()[relative] == (
        tree.package_revision if relative == MODULE else tree.child_revision
    )


@pytest.mark.parametrize('relative', [MODULE, NESTED])
@pytest.mark.parametrize('kind', ['tracked', 'staged', 'untracked'])
def test_dirty_original_is_rejected_even_with_ignore_all(source_tree, relative, kind):
    tree = source_tree
    initialize(tree)
    ignore_submodules(tree)
    directory = tree.root / relative
    target = directory / ('local.txt' if kind == 'untracked' else 'source.txt')
    target.write_bytes(b'preserve local changes\n')
    if kind == 'staged':
        git(directory, 'add', target.name)
    original_head = git(directory, 'rev-parse', 'HEAD')
    original_status = git(directory, 'status', '--porcelain', '--ignore-submodules=all')
    for before_update in (True, False):
        with pytest.raises(ValueError):
            tree.checker.check_submodules(before_update=before_update)
    assert target.read_bytes() == b'preserve local changes\n'
    assert git(directory, 'rev-parse', 'HEAD') == original_head
    assert git(directory, 'status', '--porcelain', '--ignore-submodules=all') == original_status


def test_missing_nested_checkout_is_allowed_only_before_update(source_tree):
    tree = source_tree
    initialize(tree)
    ignore_submodules(tree)
    git(tree.root / MODULE, 'submodule', 'deinit', '-f', 'nested')
    tree.checker.check_submodules(before_update=True)
    with pytest.raises(ValueError):
        tree.checker.check_submodules()
    initialize(tree)
    assert tree.checker.check_submodules()[NESTED] == tree.child_revision


@pytest.mark.parametrize('relative', [MODULE, NESTED])
def test_staged_submodule_pin_is_rejected_before_update(source_tree, relative):
    tree = source_tree
    initialize(tree)
    ignore_submodules(tree)
    parent = tree.root if relative == MODULE else tree.root / MODULE
    child = MODULE if relative == MODULE else 'nested'
    latest = tree.package_latest if relative == MODULE else tree.child_latest
    git(tree.root / relative, 'checkout', '--detach', latest)
    git(parent, 'add', child)
    original_index = git(parent, 'ls-files', '--stage', child)
    with pytest.raises(ValueError):
        tree.checker.check_submodules(before_update=True)
    assert git(parent, 'ls-files', '--stage', child) == original_index
    assert git(tree.root / relative, 'rev-parse', 'HEAD') == latest


def test_uncommitted_submodule_url_is_rejected_before_update(source_tree):
    tree = source_tree
    target = tree.root / '.gitmodules'
    target.write_text(target.read_text().replace(str(tree.package_origin), '/untrusted/source'))
    original = target.read_bytes()
    with pytest.raises(ValueError):
        tree.checker.check_submodules(before_update=True)
    assert target.read_bytes() == original


def test_nonempty_uninitialized_path_is_preserved(source_tree):
    tree = source_tree
    target = tree.root / MODULE / 'existing.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'do not replace\n')
    for before_update in (True, False):
        with pytest.raises(ValueError):
            tree.checker.check_submodules(before_update=before_update)
    assert target.read_bytes() == b'do not replace\n'
    assert not (target.parent / '.git').exists()


def test_missing_module_metadata_does_not_fall_back_to_parent_repository(source_tree):
    tree = source_tree
    initialize(tree)
    (tree.root / NESTED / '.git').unlink()
    for before_update in (True, False):
        with pytest.raises(ValueError):
            tree.checker.check_submodules(before_update=before_update)
    assert (tree.root / NESTED / 'source.txt').read_text() == 'pinned child\n'


def test_snapshot_mismatch_fails_before_any_update(source_tree):
    tree = source_tree
    target = tree.root / tree.snapshot
    target.write_bytes(b'preserve modified snapshot\n')
    with pytest.raises(ValueError, match='snapshot'):
        tree.checker.check_sources(before_update=True)
    assert target.read_bytes() == b'preserve modified snapshot\n'


@pytest.mark.parametrize('tool,manifest,directory', [
    ('grasp', 'service_source.json', 'capx_service'),
    ('curobo', 'implementation_source.json', 'open_robot_skills'),
])
def test_tool_snapshot_manifests_are_verified_before_update(source_tree, tool, manifest, directory):
    tree = source_tree
    tool_root = tree.root / 'src/tools' / tool
    target = tool_root / 'third_party' / directory / 'wrapper.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'original wrapper\n')
    (tool_root / manifest).write_text(json.dumps({'files': [{
        'path': 'wrapper.py', 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    }]}))
    assert tree.checker.check_sources(before_update=True)['verified_snapshot_files'] == 2
    target.write_bytes(b'preserve modified wrapper\n')
    with pytest.raises(ValueError, match='snapshot'):
        tree.checker.check_sources(before_update=True)
    assert target.read_bytes() == b'preserve modified wrapper\n'


@pytest.mark.parametrize('flag', ['--source-preflight', '--sources-only'])
def test_source_only_modes_do_not_require_models_or_environments(source_tree, monkeypatch, capsys, flag):
    checker = source_tree.checker
    checks = []

    def sources(*, before_update=False):
        checks.append(('sources', before_update))
        return {'submodules': {}, 'verified_snapshot_files': 1}

    def assets():
        checks.append(('assets', False))
        return {'sha256_verified_lfs_files': 1}

    def unavailable(*args, **kwargs):
        pytest.fail('Source setup must not require model checkpoints or virtual environments')

    monkeypatch.setattr(checker, 'check_sources', sources)
    monkeypatch.setattr(checker, 'check_robolab_assets', assets)
    monkeypatch.setattr(checker, 'check_models', unavailable)
    monkeypatch.setattr(checker, 'check_environment', unavailable)
    assert checker.main([flag]) == 0
    assert checks == ([('sources', True)] if flag == '--source-preflight'
                      else [('sources', False), ('assets', False)])
    result = json.loads(capsys.readouterr().out)
    assert result['ready'] is True
    assert result['llm_calls'] == 0


def test_source_only_modes_are_mutually_exclusive(source_tree):
    with pytest.raises(SystemExit) as error:
        source_tree.checker.main(['--sources-only', '--source-preflight'])
    assert error.value.code == 2


@pytest.mark.parametrize('fail_preflight', [False, True])
def test_source_script_checks_before_writing_and_verifies_afterward(tmp_path, fail_preflight):
    checkout = tmp_path / 'checkout'
    script = checkout / 'scripts/setup/sources.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'scripts/setup/sources.sh', script)
    checker = checkout / 'scripts/check/setup.py'
    checker.parent.mkdir(parents=True)
    checker.write_text(
        'import json, os, sys\n'
        'with open(os.environ["SOURCE_SETUP_EVENTS"], "a") as output:\n'
        '    output.write(json.dumps(["check", *sys.argv[1:]]) + "\\n")\n'
        'if "--source-preflight" in sys.argv and os.environ["SOURCE_SETUP_FAIL"] == "1":\n'
        '    raise SystemExit(1)\n'
    )
    commands = tmp_path / 'bin'
    commands.mkdir()
    git_program = commands / 'git'
    git_program.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'with open(os.environ["SOURCE_SETUP_EVENTS"], "a") as output:\n'
        '    output.write(json.dumps(["git", *sys.argv[1:]]) + "\\n")\n'
    )
    git_program.chmod(0o755)
    event_path = tmp_path / 'events.jsonl'
    env = dict(os.environ, SOURCE_SETUP_EVENTS=str(event_path),
               SOURCE_SETUP_FAIL=str(int(fail_preflight)), PYTHONDONTWRITEBYTECODE='1',
               PATH=str(commands) + os.pathsep + os.environ['PATH'])
    result = subprocess.run(['bash', str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=20)
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    operations = [event for event in events if event != ['git', 'lfs', 'version']]
    preflight = ['check', '--source-preflight']
    if fail_preflight:
        assert result.returncode != 0
        assert operations == [preflight]
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert operations == [
            preflight,
            ['git', '-C', str(checkout), 'submodule', 'sync', '--recursive'],
            ['git', '-C', str(checkout), 'submodule', 'update', '--init', '--recursive', '--checkout'],
            ['git', '-C', str(checkout / 'src/simulator/robolab/third_party/robolab'), 'lfs', 'pull'],
            ['check', '--sources-only'],
        ]
