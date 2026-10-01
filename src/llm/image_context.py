"""Image references and per-role image selection policies."""
from pathlib import Path
from uuid import uuid4


class ImageRegistry:
    def __init__(self):
        self.paths = {}
        self.encoded = {}

    def add(self, path):
        ref = "img_" + uuid4().hex
        self.paths[ref] = Path(path)
        return ref

    def unique_refs_by_path(self, refs):
        """Keep the first reference to each file, including path aliases."""
        selected, seen = [], set()
        for ref in refs:
            path = self.paths[ref].resolve()
            if path not in seen:
                selected.append(ref)
                seen.add(path)
        return selected

    def refs(self, value):
        found = []
        def walk(item):
            if isinstance(item, dict):
                for child in item.values():
                    walk(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    walk(child)
            elif isinstance(item, str) and item in self.paths and item not in found:
                found.append(item)
        walk(value)
        return found

    def request_refs(self, messages, *, role, target_intent_mode=False):
        """Eight relevant images: scene + inspected details, otherwise proposals."""
        if role == 'progress_review':
            current = next((m['content']['current_observation'] for m in reversed(messages)
                if isinstance(m.get('content'), dict) and 'current_observation' in m['content']), {})
            # A condition string can itself equal a registered historical ref.
            # Only the explicit current views may supply reviewer image pixels.
            return {v['image_ref'] for v in current.get('views', [])[:2]
                    if isinstance(v, dict) and v.get('image_ref') in self.paths}
        all_refs = self.refs(messages)
        if role == 'place' and target_intent_mode:
            current = next((m['content']['current_observation'] for m in reversed(messages)
                if isinstance(m.get('content'), dict) and 'current_observation' in m['content']), {})
            bundle = next((m['content'] for m in reversed(messages)
                if m.get('tool') == 'place_candidates' and isinstance(m.get('content'), dict)
                and 'candidates' in m['content']), {})
            inspection = next((m['content'] for m in reversed(messages)
                if m.get('tool') == 'inspect_place_candidate' and isinstance(m.get('content'), dict)), {})
            selected, sources = [], set()
            for ref in [*self.refs(current)[:2], *self.refs(bundle.get('image_refs', [])),
                        *self.refs(inspection)]:
                source = str(self.paths[ref])
                if source not in sources:
                    selected.append(ref)
                    sources.add(source)
                if len(selected) == 8:
                    break
            return set(selected)
        if not target_intent_mode:
            return set(all_refs[-8:])
        current = next((m['content']['current_observation'] for m in reversed(messages)
                        if isinstance(m.get('content'), dict) and 'current_observation' in m['content']), {})
        task = next((m['content'] for m in messages if isinstance(m.get('content'), dict)
                     and any(key in m['content'] for key in
                             ('candidate_bundle', 'diagnostic_candidate', 'target_reference'))), {})
        scene = self.refs(current or task.get('current_scene', {}))[:2]
        reference = self.refs(task.get('target_reference', {}))
        bundle = task.get('candidate_bundle', {})
        ensemble = (role == 'grasp' and bundle.get('generator') == 'ensemble'
                    and not any(m.get('role') == 'tool' for m in messages))
        # Rejected poses are review inputs too; unrelated history must not crowd
        # their previews out before Selector can request a correction.
        candidates = self.refs((bundle.get('candidates', []),
                                bundle.get('diagnostic_candidates', []),
                                task.get('diagnostic_candidate', {})))
        if ensemble:
            candidates = self.refs([c.get('image_refs', [])[:1] for c in
                [*bundle.get('candidates', []), *bundle.get('diagnostic_candidates', [])]]) + candidates
        inspection = next((m['content'] for m in reversed(messages)
                           if m.get('role') == 'tool' and m.get('tool') in
                           ('inspect_candidate', 'preview_candidate', 'refine_candidate', 'adjust_grasp',
                            'select_region', 'view_waypoint', 'propose_waypoint', 'propose_downward_waypoint', 'shift_waypoint',
                            'preview_view', 'refine_view') and 'image_refs' in m.get('content', {})), {})
        feedback = next((m['content'] for m in reversed(messages) if role == 'prime'
                         and m.get('role') == 'tool' and m.get('tool', '').startswith('delegate_')), {})
        recalled = next((m['content'] for m in reversed(messages) if role == 'prime'
                         and m.get('role') == 'tool' and m.get('tool') == 'review_observation'
                         and isinstance(m.get('content'), dict)
                         and m['content'].get('reference_only') is True), {})
        selected, sources = [], set()
        for ref in [*scene, *(candidates if ensemble else []), *reference, *self.refs(recalled), *self.refs(feedback),
                    *self.refs(inspection), *candidates, *reversed(all_refs)]:
            source = str(self.paths[ref])
            if source not in sources:
                sources.add(source)
                selected.append(ref)
            if len(selected) == 8:
                break
        return set(selected)

    def current_turn_refs(self, messages, *, role):
        """Use current RGB and this turn's result without filling from image history."""
        current = next((m['content']['current_observation'] for m in reversed(messages)
            if isinstance(m.get('content'), dict) and 'current_observation' in m['content']), {})
        scene = [v['image_ref'] for v in current.get('views', [])[:2]
                 if isinstance(v, dict) and v.get('image_ref') in self.paths]
        if role == 'progress_review':
            return scene
        task = next((m['content'] for m in messages if m.get('role') == 'user'
                     and isinstance(m.get('content'), dict)), {})
        latest = next((m for m in reversed(messages) if m.get('role') == 'tool'), None)
        bundle = task.get('candidate_bundle', {})
        ensemble = role == 'grasp' and latest is None and bundle.get('generator') == 'ensemble'
        refs = list(scene)
        if ensemble:
            refs += self.refs([c.get('image_refs', [])[:1] for c in
                [*bundle.get('candidates', []), *bundle.get('diagnostic_candidates', [])]])
        if latest is None:
            # A new delegation is a new visual task. Its explicit inputs can
            # include historical identity context, but not previous_feedback.
            bundle = task.get('candidate_bundle', {})
            refs += self.refs([task.get('image_refs', []) if task.get('inflight_refinement') else [],
                               task.get('target_reference', {}),
                               task.get('prior_candidate_inspection', {}),
                               bundle.get('candidates', []), bundle.get('diagnostic_candidates', []),
                               task.get('diagnostic_candidate', {}), task.get('image_refs', [])])
        else:
            refs += self.refs(latest.get('content', {}))
        if role == 'refiner' and task.get('inflight_refinement'):
            # A rejected nudge leaves the pending pose unchanged, and this
            # session has no inspect tool. Keep only that pose's last preview.
            pending = task.get('image_refs', [])
            for message in messages:
                result = message.get('content')
                if (message.get('role') == 'tool' and message.get('tool') in ('nudge_grasp', 'nudge_place')
                        and isinstance(result, dict) and result.get('accepted') is True):
                    pending = result.get('image_refs', [])
            refs += self.refs(pending)
        selected, sources = [], set()
        for ref in refs:
            source = str(self.paths[ref])
            if source not in sources:
                selected.append(ref)
                sources.add(source)
            if len(selected) == 8:
                break
        return selected
