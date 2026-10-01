"""Arguments shared by isolated RobotUse entrypoint tests."""


def flags(tmp_path):
    return ['--difficulty', 'simple', '--task', 'BananaInBowlTask',
            '--output-dir', str(tmp_path / 'live')]
