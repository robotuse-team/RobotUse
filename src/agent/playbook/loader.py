"""Load a frozen, role-scoped advisory document without simulator dependencies."""
from dataclasses import dataclass
import hashlib
from pathlib import Path


ROLES = ('common', 'prime', 'point', 'grasp', 'place', 'refiner')


@dataclass(frozen=True)
class DecisionPlaybook:
    path: Path
    text: str
    sha256: str
    sections: dict[str, str]

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        if not raw or len(raw) > 48000:
            raise ValueError('decision playbook must contain 1–48000 bytes')
        text = raw.decode('utf-8')
        sections = {}
        current = None
        for line in text.splitlines():
            if line.startswith('## '):
                current = line[3:].strip()
                if current not in ROLES or current in sections:
                    raise ValueError(f'unknown or duplicate playbook section: {current}')
                sections[current] = []
            elif current is not None:
                sections[current].append(line)
        sections = {key: '\n'.join(lines).strip() for key, lines in sections.items()}
        if set(sections) != set(ROLES) or not all(sections.values()):
            raise ValueError('decision playbook requires nonempty common and all five role sections')
        return cls(path, text, hashlib.sha256(raw).hexdigest(), sections)

    def render(self, role):
        if role not in ROLES or role == 'common':
            return ''
        return (f'\n\nDECISION PLAYBOOK sha256={self.sha256} role={role}\n'
                'Advisory decision guidance; current tool schemas, preconditions, and measured '
                'results remain authoritative.\n' + self.sections['common'] + '\n\n' + self.sections[role])

    def metadata(self):
        return {'path': str(self.path), 'sha256': self.sha256,
                'roles': list(self.sections), 'bytes': len(self.text.encode('utf-8'))}
