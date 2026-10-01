"""Robot-only kinematics captured from PhysX; planning never moves the simulator.

Space screws are reconstructed from the measured body Jacobian at one joint
configuration. Product-of-exponentials FK uses the same calibrated chain and
flange as the native controller, without a Panda-hand TCP assumption.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from .robolab_control import rigid_pose, validate_joint_target


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def screw_transform(omega, velocity, theta):
    out = np.eye(4)
    if np.linalg.norm(omega) < 1e-8:
        out[:3, 3] = velocity * theta
        return out
    out[:3, :3] = Rotation.from_rotvec(omega * theta).as_matrix()
    out[:3, 3] = (np.eye(3) - out[:3, :3]) @ np.cross(omega, velocity) + omega * (omega @ velocity) * theta
    return out


class RobotKinematics:
    def __init__(self, reference_joints, reference_pose, jacobian, limits):
        self.reference_joints = validate_joint_target(reference_joints)
        self.reference_pose = rigid_pose(reference_pose)
        jacobian = np.asarray(jacobian, dtype=float)
        self.limits = np.asarray(limits, dtype=float)
        if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
            raise ValueError('expected finite 6x7 robot Jacobian')
        self.omega = jacobian[3:].T.copy()
        norms = np.linalg.norm(self.omega, axis=1)
        if not np.allclose(norms, 1, atol=1e-4):
            raise ValueError('DROID arm must have seven revolute joints')
        self.velocity = jacobian[:3].T - np.cross(self.omega, self.reference_pose[:3, 3])
        self._hat = np.array([_skew(w) for w in self.omega])
        self._hat2 = self._hat @ self._hat
        self._cross = np.cross(self.omega, self.velocity)
        self._along = np.sum(self.omega * self.velocity, axis=1)
        validate_joint_target(self.reference_joints, self.limits)

    def _forward(self, joints, *, jacobian=False):
        q = validate_joint_target(joints)
        pose = np.eye(4)
        omega, velocity = [], []
        for i, delta in enumerate(q-self.reference_joints):
            if jacobian:
                w = pose[:3, :3] @ self.omega[i]
                omega.append(w)
                velocity.append(pose[:3, :3] @ self.velocity[i] + np.cross(pose[:3, 3], w))
            step = np.eye(4)
            step[:3, :3] += np.sin(delta)*self._hat[i] + (1-np.cos(delta))*self._hat2[i]
            step[:3, 3] = (np.eye(3)-step[:3, :3]) @ self._cross[i] + self.omega[i]*self._along[i]*delta
            pose = pose @ step
        pose = pose @ self.reference_pose
        if jacobian:
            omega = np.asarray(omega)
            return pose, np.vstack(((np.cross(omega, pose[:3, 3]) + velocity).T, omega.T))
        return pose

    def fk(self, joints):
        return self._forward(joints)

    def residual_jacobian(self, target, joints):
        actual, geometric = self._forward(joints, jacobian=True)
        error = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
        theta = np.linalg.norm(error)
        hat = _skew(error)
        coefficient = (1/12 + theta*theta/720 if theta < 1e-4 else
                       1/(theta*theta) - (1+np.cos(theta))/(2*theta*np.sin(theta)))
        right_inverse = np.eye(3) + .5*hat + coefficient*(hat @ hat)
        residual = np.r_[(actual[:3, 3]-target[:3, 3])*5, error]
        derivative = np.vstack((5*geometric[:3], -right_inverse @ geometric[3:]))
        return residual, derivative

    def solve(self, target, seed):
        target = rigid_pose(target)
        seed = validate_joint_target(seed, self.limits)
        class Converged(Exception):
            def __init__(self, joints):
                self.joints = joints.copy()
        cached_q, cached = None, None
        def evaluate(q):
            nonlocal cached_q, cached
            if cached_q is None or not np.array_equal(q, cached_q):
                cached_q, cached = q.copy(), self.residual_jacobian(target, q)
                # Stop at 0.1 mm / 0.001 rad, tighter than the public acceptance
                # tolerance. Redundant-arm least squares otherwise spends most
                # of its time polishing an already usable waypoint to 1e-9.
                error = cached[0]
                if np.linalg.norm(error[:3]) < .0005 and np.linalg.norm(error[3:]) < .001:
                    raise Converged(q)
            return cached
        try:
            solved = least_squares(lambda q: evaluate(q)[0], seed, jac=lambda q: evaluate(q)[1],
                                   bounds=(self.limits[:, 0], self.limits[:, 1]),
                                   max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            solution = solved.x
        except Converged as done:
            solution = done.joints
        actual = self.fk(solution)
        # Match the straight-waypoint corridor: 3 mm and 0.05 rad.
        # The tighter convergence target above still guides the optimizer.
        if (np.linalg.norm(actual[:3, 3]-target[:3, 3]) > .003 or
                Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude() > .05):
            return None
        return solution.tolist()


class RoboLabIK:
    """Common connector IK/linear planning API; no implicit Panda model fallback."""
    trajectory_needs_joint_reverse = False
    collision_checks_enabled = False
    planning_failure_code = None

    def __init__(self, env):
        self.env = env
        robot = env.robot
        jac = env._numpy(robot.root_physx_view.get_jacobians()[0, env._eef_id-1])[:, env._arm_ids]
        base = env._base_matrix()
        root_rotation = base[:3, :3].T
        jac[:3] = root_rotation @ jac[:3]
        jac[3:] = root_rotation @ jac[3:]
        flange = env.ee_matrix()
        center_w = env._numpy(robot.data.body_com_pos_w[0, env._eef_id])
        center = root_rotation @ (center_w - base[:3, 3])
        # PhysX gives linear velocity at the body's center of mass, while our
        # control pose is the base_link origin. Shift the Jacobian reference
        # point before extracting screw axes (v_link = v_com + w x offset).
        jac[:3] += np.cross(jac[3:].T, flange[:3, 3] - center).T
        self.model = RobotKinematics(env.joints(), flange, jac,
            env._numpy(robot.data.joint_pos_limits[0, env._arm_ids]))

    @staticmethod
    def _matrix(pose):
        if isinstance(pose, dict):
            from gap_core.types import pose_to_matrix
            return pose_to_matrix(pose)
        return rigid_pose(pose)

    def supports_world_aware_plan(self):
        return False

    def check_release_self_collision(self, joint_positions, *, jaw_width_m):
        from types import SimpleNamespace
        checker = self.env.make_path_collision(SimpleNamespace(env=self.env, ik=self), clearance_m=0.)
        return [checker.check_joint_self_collision(q, jaw_width_m=jaw_width_m) for q in joint_positions]

    def release_goal_ik(self, poses, *, jaw_width_m):
        results = []
        for pose in poses:
            try:
                target = self._matrix(pose)
            except (ValueError, TypeError):
                results.append(dict(accepted=False, stage='release_goal_ik',
                    reason_code='invalid_release_pose', path_planned=False))
                continue
            q = self.solve_ik(target)
            if q is None:
                results.append(dict(accepted=False, stage='release_goal_ik', reason_code='unreachable', path_planned=False))
                continue
            actual = self.model.fk(q)
            collision = self.check_release_self_collision([q], jaw_width_m=jaw_width_m)[0]
            results.append(dict(accepted=collision['accepted'], stage='release_goal_ik',
                joint_positions=q, position_error_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])),
                orientation_error_rad=float(Rotation.from_matrix(actual[:3,:3] @ target[:3,:3].T).magnitude()),
                solver='native PhysX screw-chain bounded least-squares IK', num_seeds=1,
                within_joint_limits=True, self_collision_check=True, self_collision=collision, path_planned=False))
        return results

    def solve_ik(self, target_world_pose, *, arm_id=0, seed_joints=None, tcp_offset=None):
        if arm_id != 0:
            raise ValueError('DROID has one arm')
        if tcp_offset is not None and not np.allclose(tcp_offset, 0):
            raise ValueError('RoboLab targets the measured flange; arbitrary Panda TCP offsets are invalid')
        seed = self.env.joints() if seed_joints is None else seed_joints
        return self.model.solve(self._matrix(target_world_pose), seed)

    def plan_linear(self, start_pose, target_world_pose, *, seed_joints=None, arm_id=0, tcp_offset=None, **kwargs):
        target = self._matrix(target_world_pose)
        start_joints = self.env.joints() if seed_joints is None else seed_joints
        start = self._matrix(start_pose)
        angle = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).magnitude()
        steps = max(2, int(np.ceil(np.linalg.norm(target[:3, 3]-start[:3, 3])/.003)), int(np.ceil(angle/.025)))
        rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
        q = validate_joint_target(start_joints)
        waypoints = []
        for fraction in np.linspace(0, 1, steps+1):
            pose = np.eye(4)
            pose[:3, 3] = start[:3, 3]*(1-fraction) + target[:3, 3]*fraction
            pose[:3, :3] = rotations(fraction).as_matrix()
            result = self.solve_ik(pose, arm_id=arm_id, seed_joints=q, tcp_offset=tcp_offset)
            if result is None or np.max(np.abs(np.asarray(result)-q)) > .2:
                return None
            q = np.asarray(result)
            waypoints.append({'positions': q.tolist()})
        return {'waypoints': waypoints}
