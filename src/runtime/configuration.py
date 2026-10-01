"""RobotUse startup configuration, CGN service and calibrated native robot assets."""
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from src.tools.grasp.adapter import ContactGraspNetClient
from src.utils.gpu import gpu_visibility


from src.agent.playbook import DEFAULT_PATH as DEFAULT_PLAYBOOK_PATH, DEFAULT_VERSION as DEFAULT_PLAYBOOK_VERSION


@dataclass(frozen=True)
class RuntimeConfiguration:
    cgn_service_url: str | None = None
    cgn_timeout_s: float = 120.
    waypoint_path_collision_checks: bool = False
    grasp_path_collision_checks: bool = False
    mandatory_observation: bool = False
    pose_preview_editor: bool = True
    playbook_path: Path = DEFAULT_PLAYBOOK_PATH
    playbook_version: str = DEFAULT_PLAYBOOK_VERSION
    observed_transit_planner: Callable | None = None

    def load_playbook(self):
        from src.agent.playbook.loader import DecisionPlaybook
        return DecisionPlaybook.load(self.playbook_path)

    def client(self):
        return ContactGraspNetClient(self.cgn_service_url, timeout_s=self.cgn_timeout_s)

    def metadata(self):
        from src.tools.grasp.arguments import HEIGHT_REFERENCES
        client = self.client()
        playbook = self.load_playbook()
        return dict(name='RobotUse',
            cgn_service_url=client.service_url, cgn_timeout_s=self.cgn_timeout_s,
            waypoint_path_collision_checks=self.waypoint_path_collision_checks,
            grasp_path_collision_checks=self.grasp_path_collision_checks,
            mandatory_observation=self.mandatory_observation,
            pose_preview_editor=self.pose_preview_editor,
            grasp_generators=['contact_graspnet', 'observed_median', 'observed_mean'] + (
                ['observed_clicked'] if self.pose_preview_editor else []),
            grasp_selection='exclusive_direction_or_center', geometric_yaws_deg=[-45, 0, 45, 90],
            pick_lift_guidance_min_m=.20,
            playbook={**playbook.metadata(), 'version': self.playbook_version},
            placement_generator='observed_clicked_median_mean', anyplace=False,
            height_references=list(HEIGHT_REFERENCES), height_fields=['reference', 'value_m'],
            place_transit_height='independent_agent_choice', place_projection_view='agentview',
            position_reference='jaw_contact_center',
            coordinate_frame='connector_base', length_units='metres', angle_units='degrees',
            geometric_rotation=('generated_top_down_edits_unrestricted' if self.pose_preview_editor else 'top_down_yaw_only'), direction_policy='agent_angular_threshold',
            transit_policy='independent_pre_post', release_policy='separate_explicit_agent_call',
            gpu=f"CUDA_VISIBLE_DEVICES={gpu_visibility()}; logical cuda:0")


def configure_args(args):
    if args.environment != 'robolab' or not args.intent_driven or args.task_runner:
        raise ValueError('RobotUse requires RoboLab intent execution without task runner')
    if not args.pause_refine or not args.world_planning:
        raise ValueError('RobotUse requires the paused refiner and observed-world planning')
    if args.decision_playbook:
        raise ValueError('RobotUse selects role policy with --playbook-version')
    if args.max_restarts != 0:
        raise ValueError('Automatic restarts are disabled: release requires an agent decision')
    args.anyplace = False
    args.grasp_scene_filter = False
    return args


def validate_runtime(args, configuration):
    configuration.client()  # validate URL/timeout before any simulator or model call
    from src.tools.perception.sam2_adapter import resolve_checkpoint, validate_source, validate_python
    validate_source()
    validate_python(args.sam_python)
    resolve_checkpoint(args.sam2_snapshot)


def make_assets(args, connector, output):
    from src.simulator.robolab.calibration import RoboLabGripperAssets
    # Native robot USD provides the meshes; the path interface uses this checkout.
    root = Path(__file__).resolve().parents[2]
    assets = RoboLabGripperAssets(connector.env.robot.cfg.spawn.usd_path, root)
    return SimpleNamespace(mesh_assets=assets, checkout=assets, libero_adapter=False,
        official_clearance_m=args.grasp_clearance_m, calls=0)


def classes(*, pose_preview_editor=True):
    """Build the one supported RobotUse backend and its agent orchestrator."""
    if not pose_preview_editor:
        raise ValueError('RobotUse requires its pose preview editor')
    from src.tools.grasp.backend import GraspBackend
    from src.tools.place.execution import PlacementMotionMixin
    from src.tools.pose_editor.adapter import PoseEditorMixin
    from src.backend.orchestrator import AgentOrchestrator

    class RobotBackend(PoseEditorMixin, PlacementMotionMixin, GraspBackend):
        pass

    return RobotBackend, AgentOrchestrator


ACTOR_CONTEXT = (
    'RobotUse uses calibrated observed RGB-D and Contact-GraspNet plus observed mean/median grasps. '
    'Agent numeric positions are jaw contact-center coordinates in robot base metres, not flange positions. '
    'Specify heights as {reference,value_m}: absolute Z or a signed offset from current_tcp, grasp, '
    'clicked_point, segment_median, segment_mean, observed_min or observed_max. '
    'Place also requires an independent transit_height above or equal to final height. '
    'Choose independent pre-pick transit and post-pick lift heights. '
    'Direction preference filters existing CGN poses by an explicit angular threshold; it never rotates them. '
    'Request CGN by direction or explicitly request mean/median as separate modes. '
    'Mean/median each expose four downward poses at fixed base yaw -45, 0, 45, 90 degrees. '
    'For transport picks choose an outgoing height at least 0.20 m above the grasp height. '
    'Placement movement retains the object; a later explicit release call is required. '
    'A completed command does not establish physical retention or task success.'
)
