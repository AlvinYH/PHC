"""Studio PHC host for the fresh ``ours`` residual policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from isaacgym import gymtorch
from rl_games.algos_torch import torch_ext

from phc.env.tasks.humanoid import compute_humanoid_observations_max
from phc.env.tasks.humanoid_im import compute_imitation_observations_v6
from phc.env.tasks.humanoid_im_passive_object import HumanoidImPassiveObject
from phc.learning.network_loader import load_pnn
from phc.utils.flags import flags
from phc.utils.isaacgym_torch_utils import quat_apply, slerp
from pipeline.physics.contact import (
    finger_part_aware_distances,
    joint_region_and_part_surface_distances_chunked,
    joint_region_surface_distances_chunked,
)
from pipeline.physics.finger_segments import (
    FINGER_SEGMENT_BODY_NAMES,
    canonical_finger_segment_order,
    load_finger_contact_part_link_names,
)
from pipeline.physics.ours.observation import OBSERVATION_DIM, build_observation
from pipeline.physics.ours.policy import (
    compose_residual_action,
    normalize_teacher_observation,
    teacher_action,
)
from pipeline.physics.ours.reward import (
    body_contact_indicator,
    build_phase_predecessor_table,
    build_phase_progress_tables,
    compute_reward,
    finger_object_contact_indicator,
    phase_relative_progress,
    phase_prerequisite_transition,
    reference_paced_progress,
    reward_body_indices,
    reward_reset_signals,
    whole_hand_object_distance,
)


class HumanoidImStudioResidual(HumanoidImPassiveObject):
    """Feed Studio tensors through the residual formulas and existing PD path."""

    def get_obs_size(self):
        return OBSERVATION_DIM

    def reset(self, env_ids=None):
        """Reset once; do not warm the contact solver before policy step zero."""

        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
        self._reset_envs(env_ids)
        # 复用统一的 Isaac Gym 运行时快照，只在启动后的首次 reset 写一次。
        from scripts.physics.capture_isaac_runtime import write_requested_task_physics
        write_requested_task_physics(self)

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env = cfg["env"]
        # PHC's Isaac Gym binding corrupts heap state at teardown after actor
        # body-property queries. Shapes, DOFs, and root poses are still read
        # from every actor by the runtime acceptance snapshot.
        self._runtime_skip_body_property_readback = True
        self._ours_hybrid_init_max_fraction = float(env["hybridInitMaxFraction"])
        if not 0.0 <= self._ours_hybrid_init_max_fraction <= 1.0:
            raise ValueError("ours Hybrid max fraction must be in [0,1]")
        self._ours_hybrid_random_init_mode = str(env["hybridRandomInitMode"])
        if self._ours_hybrid_random_init_mode not in {
            "default", "random", "pre_contact",
        }:
            raise ValueError(
                "ours Hybrid random init mode must be default, random, "
                "or pre_contact"
            )
        self._ours_hybrid_start_mask = None
        self._ours_reference_path = Path(env["oursReferencePath"]).expanduser()
        self._ours_reward_weights = dict(env["oursReward"])
        self._ours_progress_mode = str(
            self._ours_reward_weights.pop("progress_mode", "phase_relative_completion")
        )
        self._ours_progress_prerequisite_mode = str(
            self._ours_reward_weights.pop("progress_prerequisite_mode", "previous")
        )
        self._ours_progress_prerequisite_threshold = float(
            self._ours_reward_weights.pop("progress_prerequisite_threshold", 0.5)
        )
        self._ours_progress_target_cap_scale = float(
            self._ours_reward_weights.pop("progress_target_cap_scale", 1.0)
        )
        self._ours_progress_lead_tolerance = float(
            self._ours_reward_weights.pop("progress_lead_tolerance", 0.0)
        )
        self._ours_part_knn_k = self._ours_reward_weights.pop(
            "part_aware_part_knn_k", 2,
        )
        self._ours_active_joint_torque_enabled = self._ours_reward_weights[
            "active_joint_torque"
        ]["enabled"]
        self._ours_part_aware_contact_reward_enabled = self._ours_reward_weights[
            "part_aware_contact_reward_enabled"
        ]
        self._ours_needs_part_attribution = (
            bool(self._ours_part_aware_contact_reward_enabled)
            or self._ours_active_joint_torque_enabled
        )
        self._ours_reset = dict(env["oursReset"])
        self._ours_residual_scale = float(env["oursResidualScale"])
        self._ours_teacher_frame_offset = int(env.get("oursTeacherFrameOffset", 0))
        self._ours_tracker_path = Path(env["studioTrackerCheckpoint"]).expanduser()
        self._ours_tracker_activation = env["studioTrackerActivation"]
        self._ours_trace_path = Path(env["oursTracePath"]).expanduser() if env.get("oursTracePath") else None
        self._ours_eval_path = Path(env["oursEvalPath"]).expanduser()
        self._ours_closed_loop_trace_path = Path(env["oursClosedLoopTracePath"]).expanduser() if env.get("oursClosedLoopTracePath") else None
        self._ours_closed_loop_trace_steps = int(env.get("oursClosedLoopTraceSteps", 0))
        self._ours_batch_replay = bool(env.get("oursBatchReplay", False))
        self._ours_action_replay_path = Path(env["oursActionReplayPath"]).expanduser() if env.get("oursActionReplayPath") else None
        self._ours_final_action_replay_path = Path(env["oursFinalActionReplayPath"]).expanduser() if env.get("oursFinalActionReplayPath") else None
        self._ours_reset_replay_path = Path(env["oursResetReplayPath"]).expanduser() if env.get("oursResetReplayPath") else None
        self._ours_diagnostic_zero_start_human_dof_velocity = bool(
            env.get("oursDiagnosticZeroStartHumanDofVelocity", False)
        )
        self._ours_diagnostic_zero_residual = bool(
            env.get("oursDiagnosticZeroResidual", False)
        )
        self._ours_batch_step_trace_path = (
            Path(env["oursBatchStepTracePath"]).expanduser()
            if env.get("oursBatchStepTracePath") else None
        )
        if self._ours_batch_step_trace_path is not None and not self._ours_diagnostic_zero_residual:
            raise ValueError("batch-step trace is restricted to the zero-residual audit")
        if self._ours_closed_loop_trace_path is not None and not 0 < self._ours_closed_loop_trace_steps <= 64:
            raise ValueError("ours closed-loop trace length must be in [1,64]")
        if self._ours_batch_replay and self._ours_batch_step_trace_path is None:
            raise ValueError("bounded batch replay requires a batch-step trace")
        if self._ours_action_replay_path is not None and self._ours_closed_loop_trace_path is None and not self._ours_batch_replay:
            raise ValueError("ours action replay is only valid for a bounded closed-loop trace")
        if self._ours_reset_replay_path is not None and self._ours_closed_loop_trace_path is None and not self._ours_batch_replay:
            raise ValueError("ours reset replay is only valid for a bounded closed-loop trace")
        if self._ours_final_action_replay_path is not None and self._ours_closed_loop_trace_path is None and not self._ours_batch_replay:
            raise ValueError("ours final-action replay is only valid for a bounded closed-loop trace")

        self._ours_trace_rows = []
        self._ours_closed_loop_rows = []
        self._ours_closed_loop_pending = None
        self._ours_last_reward_terms = None
        self._ours_reward_sum = {}
        self._ours_reward_min = {}
        self._ours_reward_max = {}
        self._ours_peak_progress = None
        self._ours_hinge_q_at_peak_progress = None
        self._ours_reward_count = 0
        self._ours_last_residual = None
        self._ours_outputs_finalized = False
        self._ours_action_replay_index = 0
        self._ours_final_action_replay_index = 0
        self._ours_batch_step_pre = None
        self._ours_batch_substeps = None
        self._ours_prepare_files()
        reference_contact = np.asarray(self._ours_reference_np["contact_labels"])
        contact_frames = np.flatnonzero(np.any(reference_contact, axis=1))
        self._ours_first_contact_frame = (
            None if len(contact_frames) == 0 else int(contact_frames[0])
        )
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._ours_opening_q_sum = torch.zeros(
            (), dtype=torch.float64, device=self.device,
        )
        self._ours_opening_reference_q_sum = torch.zeros(
            (), dtype=torch.float64, device=self.device,
        )
        self._ours_opening_q_error_sum = torch.zeros(
            (), dtype=torch.float64, device=self.device,
        )
        self._ours_opening_q_count = torch.zeros(
            (), dtype=torch.int64, device=self.device,
        )
        self._ours_bind_studio_tensors()
        self._compute_observations()
        self._ours_capture_trace()
        from scripts.physics.capture_isaac_runtime import request_task_physics_dump
        request_task_physics_dump(self)

    def arm_ours_closed_loop_replay(self):
        """Start bounded replay after the residual player restores its policy."""

        if self._ours_closed_loop_trace_path is None or self._ours_batch_replay:
            return
        if self._ours_closed_loop_pending is not None:
            raise RuntimeError("cannot arm closed-loop replay with a pending step")
        self._ours_closed_loop_rows.clear()
        self._ours_action_replay_index = 0
        self._ours_final_action_replay_index = 0
        self._ours_outputs_finalized = False

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel,
                       root_ang_vel, dof_vel, **kwargs):
        if self._ours_diagnostic_zero_start_human_dof_velocity:
            dof_vel = torch.zeros_like(dof_vel)
        return super()._set_env_state(
            env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel,
            dof_vel, **kwargs,
        )

    def _object_reference_state_at_times(self, motion_times):
        """Look up the object reference at the same continuous time as the human."""

        frame = (
            motion_times * float(self._target_qpos_fps)
        ).clamp(0.0, float(self._ours_ref_root_state.shape[0] - 1))
        lower = torch.floor(frame).long()
        upper = (lower + 1).clamp(max=self._ours_ref_root_state.shape[0] - 1)
        blend = (frame - lower.to(frame)).unsqueeze(-1)

        root_lower = self._ours_ref_root_state[lower]
        root_upper = self._ours_ref_root_state[upper]
        root_state = torch.cat((
            torch.lerp(root_lower[:, :3], root_upper[:, :3], blend),
            slerp(root_lower[:, 3:7], root_upper[:, 3:7], blend),
            torch.lerp(root_lower[:, 7:], root_upper[:, 7:], blend),
        ), dim=-1)
        qpos = torch.lerp(
            self._target_joint_qpos[lower], self._target_joint_qpos[upper], blend,
        )
        qvel = torch.lerp(
            self._target_joint_qvel[lower], self._target_joint_qvel[upper], blend,
        )
        return root_state, qpos, qvel

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        motion_times = (
            self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )
        root_state, qpos, qvel = self._object_reference_state_at_times(
            motion_times,
        )
        self._target_states[env_ids] = root_state
        self._target_dof_pos[env_ids] = qpos
        self._target_dof_vel[env_ids] = qvel
        self._target_last_reset_joint_qpos[env_ids] = qpos
        self._target_last_reset_reference_frame[env_ids] = torch.floor(
            motion_times * float(self._target_qpos_fps),
        ).long().clamp(0, self._ours_ref_root_state.shape[0] - 1)
        if self._ours_reset_replay_np is None or len(env_ids) == 0:
            return
        if not self._ours_batch_replay and (
            self.num_envs != 1 or len(env_ids) != 1 or int(env_ids[0]) != 0
        ):
            raise RuntimeError("bounded reset replay requires the sole environment")
        replay = self._ours_reset_replay_np
        replay_ids = env_ids if self._ours_batch_replay else None

        def replay_tensor(name, like):
            value = torch.as_tensor(
                replay[name], dtype=like.dtype, device=self.device,
            )
            return value.index_select(0, replay_ids) if replay_ids is not None else value.unsqueeze(0)

        self._humanoid_root_states[env_ids] = replay_tensor(
            "human_root", self._humanoid_root_states,
        )
        self._dof_pos[env_ids] = replay_tensor("human_dof_pos", self._dof_pos)
        self._dof_vel[env_ids] = replay_tensor("human_dof_vel", self._dof_vel)
        body = replay_tensor("human_body", self._rigid_body_pos)
        self._rigid_body_pos[env_ids] = body[..., 0:3]
        self._rigid_body_rot[env_ids] = body[..., 3:7]
        self._rigid_body_vel[env_ids] = body[..., 7:10]
        self._rigid_body_ang_vel[env_ids] = body[..., 10:13]
        self._reset_rb_pos = self._rigid_body_pos.clone()
        self._reset_rb_rot = self._rigid_body_rot.clone()
        self._reset_rb_vel = self._rigid_body_vel.clone()
        self._reset_rb_ang_vel = self._rigid_body_ang_vel.clone()
        self._target_states[env_ids] = replay_tensor(
            "object_root", self._target_states,
        )
        object_dof_pos = replay_tensor("object_dof_pos", self._target_dof_pos)
        object_dof_vel = replay_tensor("object_dof_vel", self._target_dof_vel)
        if object_dof_pos.shape[1] == self._target_max_dof:
            self._target_dof_pos[env_ids] = object_dof_pos
            self._target_dof_vel[env_ids] = object_dof_vel
        else:
            # Selected source traces store the sole active joint only.  Replay
            # that named DOF and retain Studio's reset state for passive DOFs.
            self._target_dof_pos[env_ids, self._ours_active_dof] = object_dof_pos[:, 0]
            self._target_dof_vel[env_ids, self._ours_active_dof] = object_dof_vel[:, 0]

    def _sample_time(self, motion_ids):
        """Bound Studio Hybrid reference starts without changing its materializer."""

        sampled = super()._sample_time(motion_ids)
        if self._ours_hybrid_random_init_mode == "random":
            return sampled
        if self._ours_hybrid_start_mask is None:
            return sampled
        if self._ours_hybrid_start_mask.shape != sampled.shape:
            raise RuntimeError("Studio Hybrid mask and motion batch differ")
        if self._ours_hybrid_random_init_mode == "pre_contact":
            if (
                self._ours_first_contact_frame is None
                or self._ours_first_contact_frame == 0
            ):
                raise ValueError(
                    "pre_contact Hybrid init requires a contact label after frame zero"
                )
            frames = torch.randint(
                self._ours_first_contact_frame,
                motion_ids.shape,
                device=self.device,
            )
            sampled_pre_contact = frames.to(dtype=torch.float32) / float(
                self._target_qpos_fps
            )
            return torch.where(
                self._ours_hybrid_start_mask,
                torch.zeros_like(sampled_pre_contact),
                sampled_pre_contact,
            )
        motion_length = self._motion_lib.get_motion_length(motion_ids).to(sampled)
        rollout_bound = torch.clamp(motion_length - float(self.max_episode_length) * float(self.dt), min=0.0)
        fraction_bound = motion_length * self._ours_hybrid_init_max_fraction
        upper = torch.minimum(rollout_bound, fraction_bound)
        phase = sampled / torch.clamp(motion_length, min=1.0e-8)
        bounded = torch.floor(phase * upper * float(self._target_qpos_fps) + 1.0e-6) / float(self._target_qpos_fps)
        return torch.where(self._ours_hybrid_start_mask, torch.zeros_like(bounded), bounded)

    def _reset_hybrid_state_init(self, env_ids):
        """Apply the selected Hybrid initialization distribution."""

        probability = float(self._hybrid_init_prob)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Studio Hybrid probability must be in [0,1]")
        if self._ours_hybrid_random_init_mode == "random":
            self._reset_ref_state_init(env_ids)
            return
        self._ours_hybrid_start_mask = (
            torch.rand(len(env_ids), device=self.device) < probability
        )
        try:
            self._reset_ref_state_init(env_ids)
        finally:
            self._ours_hybrid_start_mask = None

    def _ours_prepare_files(self):
        for path, description in (
            (self._ours_reference_path, "articulated reference"),
            (self._ours_tracker_path, "PHC-X tracker checkpoint"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Studio {description} is missing: {path}")
        if self._ours_tracker_activation != "silu":
            raise ValueError("Studio tracker activation must be SiLU")
        if self._ours_teacher_frame_offset not in (0, 1):
            raise ValueError("diagnostic teacher frame offset must be 0 or 1")
        if self._ours_reward_weights["ig"] != 0.0 or self._ours_reward_weights["handle_normal"] != 0.0:
            raise ValueError("baseline requires neutral IG and handle-normal")
        if self._ours_progress_mode not in (
            "phase_relative_completion",
            "reference_paced_completion",
        ):
            raise ValueError(
                "ours progress mode must be phase_relative_completion or "
                "reference_paced_completion"
            )
        if self._ours_progress_prerequisite_mode not in ("previous", "all"):
            raise ValueError("ours progress prerequisite mode must be previous or all")
        if not 0.0 < self._ours_progress_prerequisite_threshold <= 1.0:
            raise ValueError("ours progress prerequisite threshold must be in (0,1]")
        if not np.isfinite(self._ours_progress_target_cap_scale) or (
            self._ours_progress_target_cap_scale <= 0.0
        ):
            raise ValueError("ours progress_target_cap_scale must be positive and finite")
        if not np.isfinite(self._ours_progress_lead_tolerance) or not (
            0.0 <= self._ours_progress_lead_tolerance <= 1.0
        ):
            raise ValueError(
                "ours progress lead tolerance must be finite and in [0,1]"
            )
        if not 0.0 <= float(self._ours_reward_weights["progress_contact_base"]) <= 1.0:
            raise ValueError("ours progress contact base must be in [0,1]")
        progress_weight = float(self._ours_reward_weights["progress"])
        if not np.isfinite(progress_weight) or progress_weight < 0.0:
            raise ValueError(
                "ours progress reward weight must be non-negative and finite"
            )
        eg3 = float(self._ours_reward_weights["eg3"])
        if not np.isfinite(eg3) or eg3 < 0.0:
            raise ValueError("ours eg3 must be finite and non-negative")
        if not 0.0 < float(self._ours_reward_weights["finger_contact_distance"]):
            raise ValueError("ours live finger-contact distance must be positive")
        if float(self._ours_reward_weights["finger_contact_force"]) < 0.0:
            raise ValueError("ours live finger-contact force threshold must be non-negative")
        progress_overshoot_penalty = float(
            self._ours_reward_weights.get("progress_overshoot_penalty", 0.0)
        )
        if not np.isfinite(progress_overshoot_penalty):
            raise ValueError("ours progress overshoot penalty must be finite")
        if progress_overshoot_penalty < 0.0:
            raise ValueError("ours progress overshoot penalty must be non-negative")
        progress_penalty_enabled = self._ours_reward_weights.get(
            "progress_overshoot_penalty_enabled", False,
        )
        if not isinstance(progress_penalty_enabled, bool):
            raise ValueError(
                "ours progress overshoot penalty switch must be boolean"
            )
        if (
            progress_penalty_enabled
            and self._ours_progress_mode != "reference_paced_completion"
        ):
            raise ValueError(
                "ours progress overshoot penalty requires "
                "reference_paced_completion"
            )
        if (
            progress_penalty_enabled
            and "progress_overshoot_penalty" not in self._ours_reward_weights
        ):
            raise ValueError(
                "enabled progress overshoot penalty requires its weight"
            )
        for name in (
            "contact_binary_reward_enabled", "contact_distance_reward_enabled",
            "coarse_contact_reward_enabled", "fine_contact_reward_enabled",
        ):
            if not isinstance(self._ours_reward_weights[name], bool):
                raise ValueError(f"ours {name} must be boolean")
        if not isinstance(self._ours_part_aware_contact_reward_enabled, bool):
            raise ValueError("ours part_aware_contact_reward_enabled must be boolean")
        if not isinstance(self._ours_active_joint_torque_enabled, bool):
            raise ValueError("ours active_joint_torque.enabled must be boolean")
        if isinstance(self._ours_part_knn_k, bool) or not isinstance(
            self._ours_part_knn_k, int
        ) or self._ours_part_knn_k < 1:
            raise ValueError("ours part-aware part KNN count must be a positive integer")
        for name in ("coarse_contact_distance_weight", "fine_contact_distance_weight"):
            if float(self._ours_reward_weights[name]) < 0.0:
                raise ValueError(f"ours {name} must be non-negative")
        if not 0.0 < float(self._ours_reset["termination_height"]):
            raise ValueError("ours termination height must be positive")
        if not 0.0 < float(self._ours_reset["human_key_body_error"]):
            raise ValueError("ours human reset threshold must be positive")
        if not 0.0 < float(self._ours_reset["object_surface_error"]):
            raise ValueError("ours object reset threshold must be positive")
        if int(self._ours_reset["required_hand_contact_mismatch_steps"]) < 1:
            raise ValueError("ours contact reset streak must be positive")

        needed = {
            "object_root_pos", "object_root_rot_xyzw", "object_root_vel", "object_root_ang_vel",
            "object_joint_qpos", "object_joint_qvel", "object_link_pos", "object_link_rot_xyzw",
            "object_link_vel", "object_link_ang_vel", "joint_names", "body_names", "active_joint_names",
            "active_parent_link_names", "active_child_link_names", "active_parent_bbox", "active_child_bbox",
            "policy_root_link_names", "policy_active_child_link_names",
            "policy_root_bbox", "policy_active_child_bbox",
            "object_surface_mode", "collision_surface_points_link_local_scaled",
            "collision_surface_point_link_names",
            "finger_contact_surface_points_link_local_scaled",
            "finger_contact_surface_point_link_names",
            "finger_contact_body_names", "finger_contact_labels",
            "contact_labels",
        }
        if self._ours_active_joint_torque_enabled:
            needed.update((
                "active_joint_origin_parent_local_scaled",
                "active_joint_axis_parent_local",
            ))
        with np.load(self._ours_reference_path, allow_pickle=False) as data:
            if self._ours_needs_part_attribution:
                needed.add("finger_contact_nearest_object_link_names")
                has_part_names = "finger_contact_part_link_names" in data.files
                has_part_schema = "finger_contact_part_schema" in data.files
                if has_part_names != has_part_schema:
                    raise ValueError("Studio part-aware finger GT fields must appear together")
                if has_part_names:
                    needed.update((
                        "finger_contact_part_link_names",
                        "finger_contact_part_schema",
                    ))
            missing = sorted(needed.difference(data.files))
            if missing:
                raise ValueError("Studio articulated reference lacks: " + ", ".join(missing))
            self._ours_reference_np = {name: np.asarray(data[name]) for name in needed}
        mode_value = self._ours_reference_np["object_surface_mode"]
        if mode_value.size != 1:
            raise ValueError("Studio object_surface_mode must be scalar")
        self._ours_object_surface = str(mode_value.reshape(-1)[0])
        if self._ours_object_surface not in ("active_part", "full_object"):
            raise ValueError("Studio object_surface_mode is invalid")
        active = tuple(str(name) for name in self._ours_reference_np["active_joint_names"].reshape(-1))
        if len(active) != 1:
            raise ValueError("ours requires exactly one active joint")
        self._ours_active_joint_name = active[0]
        self._ours_parent_link = str(self._ours_reference_np["active_parent_link_names"].reshape(-1)[0])
        self._ours_child_link = str(self._ours_reference_np["active_child_link_names"].reshape(-1)[0])
        qpos = self._ours_reference_np["object_joint_qpos"]
        qvel = self._ours_reference_np["object_joint_qvel"]
        if qpos.ndim != 2 or qvel.shape != qpos.shape or qpos.shape[0] < 2:
            raise ValueError("Studio qpos/qvel reference must be aligned [T,J]")
        self._ours_reference_joint_names = tuple(str(name) for name in self._ours_reference_np["joint_names"].reshape(-1))
        if len(self._ours_reference_joint_names) != qpos.shape[1] or active[0] not in self._ours_reference_joint_names:
            raise ValueError("Studio joint names do not resolve the active qpos/qvel column")
        self._ours_action_replay_np = None
        if self._ours_action_replay_path is not None:
            if not self._ours_action_replay_path.is_file():
                raise FileNotFoundError(f"bounded action replay is missing: {self._ours_action_replay_path}")
            with np.load(self._ours_action_replay_path, allow_pickle=False) as replay:
                if "residual_action" not in replay.files:
                    raise ValueError("bounded action replay lacks residual_action")
                action = np.asarray(replay["residual_action"], dtype=np.float32)
            if (
                action.ndim != 2 or action.shape[1] != 153
                or (
                    not self._ours_batch_replay
                    and action.shape[0] < self._ours_closed_loop_trace_steps
                )
            ):
                raise ValueError("bounded action replay must contain at least the requested [T,153] actions")
            if not np.isfinite(action).all():
                raise FloatingPointError("bounded action replay contains NaN/Inf")
            self._ours_action_replay_np = (
                action if self._ours_batch_replay
                else action[:self._ours_closed_loop_trace_steps]
            )
        self._ours_reset_replay_np = None
        if self._ours_reset_replay_path is not None:
            if not self._ours_reset_replay_path.is_file():
                raise FileNotFoundError(
                    f"bounded reset replay is missing: {self._ours_reset_replay_path}"
                )
            with np.load(self._ours_reset_replay_path, allow_pickle=False) as replay:
                needed = {
                    "pre_frame", "pre_body_state", "pre_dof_pos", "pre_dof_vel",
                    "solver_pre_object_actor_state",
                    "pre_object_dof_pos", "pre_object_dof_vel",
                }
                missing = sorted(needed.difference(replay.files))
                if missing:
                    raise ValueError(
                        "bounded reset replay lacks: " + ", ".join(missing)
                    )
                frames = np.asarray(replay["pre_frame"])
                body = np.asarray(replay["pre_body_state"], dtype=np.float32)
                values = {
                    "human_body": body,
                    "human_root": body[:, 0],
                    "human_dof_pos": np.asarray(replay["pre_dof_pos"], dtype=np.float32),
                    "human_dof_vel": np.asarray(replay["pre_dof_vel"], dtype=np.float32),
                    "object_root": np.asarray(
                        replay["solver_pre_object_actor_state"],
                        dtype=np.float32,
                    ),
                    "object_dof_pos": np.asarray(replay["pre_object_dof_pos"], dtype=np.float32),
                    "object_dof_vel": np.asarray(replay["pre_object_dof_vel"], dtype=np.float32),
                }
            if (
                frames.ndim != 1 or frames.size == 0 or frames[0] != 0
                or (self._ours_batch_replay and not np.equal(frames, 0).all())
            ):
                raise ValueError("bounded reset replay must start at frame zero")
            if not self._ours_batch_replay:
                values = {name: value[0] for name, value in values.items()}
            if any(not np.isfinite(value).all() for value in values.values()):
                raise FloatingPointError("bounded reset replay contains NaN/Inf")
            prefix = (frames.shape[0],) if self._ours_batch_replay else ()
            if values["human_root"].shape != prefix + (13,):
                raise ValueError("bounded reset replay human root has invalid shape")
            if values["human_body"].shape != prefix + (52, 13):
                raise ValueError("bounded reset replay human body has invalid shape")
            if (
                values["human_dof_pos"].shape != prefix + (153,)
                or values["human_dof_vel"].shape != prefix + (153,)
            ):
                raise ValueError("bounded reset replay human DOFs have invalid shape")
            self._ours_reset_replay_np = values
        self._ours_final_action_replay_np = None
        if self._ours_final_action_replay_path is not None:
            if not self._ours_final_action_replay_path.is_file():
                raise FileNotFoundError(
                    "bounded final-action replay is missing: "
                    + str(self._ours_final_action_replay_path)
                )
            with np.load(self._ours_final_action_replay_path, allow_pickle=False) as replay:
                if "final_action" not in replay.files:
                    raise ValueError("bounded final-action replay lacks final_action")
                action = np.asarray(replay["final_action"], dtype=np.float32)
            if (
                action.ndim != 2 or action.shape[1] != 153
                or (
                    not self._ours_batch_replay
                    and action.shape[0] < self._ours_closed_loop_trace_steps
                )
            ):
                raise ValueError(
                    "bounded final-action replay must contain requested [T,153] actions"
                )
            if not np.isfinite(action).all():
                raise FloatingPointError("bounded final-action replay contains NaN/Inf")
            self._ours_final_action_replay_np = (
                action if self._ours_batch_replay
                else action[:self._ours_closed_loop_trace_steps]
            )

    def _load_contact_region_points(self, object_config):
        """Read Ours' object geometry from the native PHC config."""

        points = np.asarray(object_config["contact_region_points"], dtype=np.float32)
        names = [str(name) for name in object_config["contact_region_point_link_names"]]
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or len(names) != len(points)
            or not np.isfinite(points).all()
            or any(not name for name in names)
        ):
            raise ValueError("Studio contact-region points must be finite named [P,3]")
        return points, names

    def _load_contact_reference(self):
        """Use the single Ours reference instead of a PHC-X sidecar."""

        labels_hand2 = np.asarray(
            self._ours_reference_np["contact_labels"], dtype=np.float32
        )
        if labels_hand2.shape != (self._target_joint_qpos.shape[0], 2):
            raise ValueError("Studio contact labels must align with object qpos")
        labels = np.concatenate(
            (
                np.repeat(labels_hand2[:, 0:1], 5, axis=1),
                np.repeat(labels_hand2[:, 1:2], 5, axis=1),
            ),
            axis=1,
        )
        self._phc_contact_labels_10 = torch.tensor(
            labels, dtype=torch.float32, device=self.device
        )
        self._phc_contact_labels_hand2 = torch.tensor(
            labels_hand2, dtype=torch.float32, device=self.device
        )
        self._phc_contact_obj_ref = torch.tensor(
            np.any(labels_hand2 > 0.5, axis=1, keepdims=True).astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )

    def _load_target_joint_qpos(self):
        """Use the single Ours reference instead of a PHC-X sidecar."""

        qpos = np.asarray(self._ours_reference_np["object_joint_qpos"], dtype=np.float32)
        qvel = np.asarray(self._ours_reference_np["object_joint_qvel"], dtype=np.float32)
        names = list(self._ours_reference_joint_names)
        if qpos.ndim != 2 or not len(qpos) or not np.isfinite(qpos).all():
            raise ValueError("Studio object qpos must be finite [T,J]")
        if qvel.shape != qpos.shape or not np.isfinite(qvel).all():
            raise ValueError("Studio object qvel must be finite [T,J]")
        if names != list(self._target_joint_names) or qpos.shape[1] != len(names):
            raise ValueError("Studio object qpos joint names disagree with PHC config")
        dof_indices = []
        for name in names:
            if name not in self._target_asset_dof_names:
                raise ValueError(f"Studio object joint is missing from asset: {name}")
            dof_indices.append(self._target_asset_dof_names.index(name))
        full_qpos = np.zeros((qpos.shape[0], self._target_max_dof), dtype=np.float32)
        full_qpos[:, dof_indices] = qpos
        full_qvel = np.zeros((qvel.shape[0], self._target_max_dof), dtype=np.float32)
        full_qvel[:, dof_indices] = qvel
        self._target_active_dof_indices = torch.tensor(
            dof_indices, dtype=torch.long, device=self.device
        )
        self._target_joint_qpos = torch.tensor(
            full_qpos, dtype=torch.float32, device=self.device
        )
        self._target_joint_qvel = torch.tensor(
            full_qvel, dtype=torch.float32, device=self.device
        )

    def _ours_bind_studio_tensors(self):
        surface_mode = str(self._ours_reference_np["object_surface_mode"].item())
        if self._phc_object_config["object_surface_mode"] != surface_mode:
            raise ValueError(
                "Studio PHC contact gates must use the selected collision surface"
            )
        if self.obs_buf.shape[1] != OBSERVATION_DIM or self.num_bodies != 52:
            raise ValueError(
                f"ours requires a {OBSERVATION_DIM}D observation and 52-body humanoid"
            )
        if self._ours_action_replay_np is not None and not self._ours_batch_replay and self.num_envs != 1:
            raise ValueError("bounded action replay requires exactly one Studio environment")
        if self._ours_active_joint_name not in self._target_joint_names or self._ours_active_joint_name not in self._target_asset_dof_names:
            raise ValueError("Studio active joint does not resolve in PHC object DOFs")
        if self._ours_reference_joint_names != tuple(self._target_joint_names):
            raise ValueError("Studio PHC and articulated references disagree on object joint order")
        if self._ours_reference_np["object_joint_qpos"].shape[1] != len(self._target_joint_names):
            raise ValueError("Studio qpos reference and PHC object-joint order differ")
        self._ours_active_reference_index = self._target_joint_names.index(self._ours_active_joint_name)
        self._ours_active_dof = self._target_asset_dof_names.index(self._ours_active_joint_name)
        if self._ours_reset_replay_np is not None:
            if not self._ours_batch_replay and self.num_envs != 1:
                raise ValueError("bounded reset replay requires exactly one Studio environment")
            prefix = (self.num_envs,) if self._ours_batch_replay else ()
            if self._ours_reset_replay_np["human_body"].shape != prefix + (52, 13):
                raise ValueError("bounded reset replay environment count differs from Studio")
            for name in ("object_dof_pos", "object_dof_vel"):
                shape = self._ours_reset_replay_np[name].shape
                if shape[:-1] != prefix or shape[-1] not in (1, self._target_max_dof):
                    raise ValueError(
                        "bounded reset replay object DOFs must be the named active joint or all Studio DOFs"
                    )
            if (
                self._ours_reset_replay_np["object_dof_pos"].shape
                != self._ours_reset_replay_np["object_dof_vel"].shape
            ):
                raise ValueError("bounded reset replay object position/velocity widths differ")
        if self._ours_batch_replay:
            for name, value in (
                ("action", self._ours_action_replay_np),
                ("final action", self._ours_final_action_replay_np),
            ):
                if value is not None and value.shape != (self.num_envs, 153):
                    raise ValueError(f"bounded batch replay {name} must be [N,153]")
        if self._ours_final_action_replay_np is not None and not self._ours_batch_replay and self.num_envs != 1:
            raise ValueError("bounded final-action replay requires exactly one Studio environment")
        object_names = [str(name) for name in self._target_body_names]
        if self._ours_parent_link not in object_names or self._ours_child_link not in object_names:
            raise ValueError("Studio active parent/child link is absent from the PHC object")
        if "object_surface_mode" not in self._phc_object_config:
            raise ValueError("Studio PHC object config is missing object_surface_mode")
        if self._phc_object_config["object_surface_mode"] != self._ours_object_surface:
            raise ValueError("Studio PHC object-surface mode differs from its reference")
        policy_roots = tuple(
            str(name)
            for name in self._ours_reference_np["policy_root_link_names"].reshape(-1)
        )
        policy_children = tuple(
            str(name)
            for name in self._ours_reference_np["policy_active_child_link_names"].reshape(-1)
        )
        if policy_roots != (object_names[0],) or policy_children != (self._ours_child_link,):
            raise ValueError(
                "Studio policy bbox owners do not match actor-root/active-child states"
            )
        self._ours_child_body_id = self.num_bodies + object_names.index(self._ours_child_link)
        self._ours_child_object_index = object_names.index(self._ours_child_link)
        self._ours_root_body_id = self.num_bodies + object_names.index(
            object_names[0]
        )
        self._ours_all_object_ids = torch.tensor(
            [self.num_bodies + object_names.index(name) for name in object_names],
            dtype=torch.long,
            device=self.device,
        )
        if self._ours_active_joint_torque_enabled:
            self._ours_parent_body_id = self.num_bodies + object_names.index(
                self._ours_parent_link
            )

        body_names = list(self._body_names)
        finger_order = ("Index", "Middle", "Pinky", "Ring", "Thumb")
        left = list(FINGER_SEGMENT_BODY_NAMES[:15])
        right = list(FINGER_SEGMENT_BODY_NAMES[15:])
        tips = [f"{side}_{finger}3" for side in ("L", "R") for finger in finger_order]
        required = ["L_Wrist", "R_Wrist", *left, *right, *tips, "L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
        missing = [name for name in required if name not in body_names]
        if missing:
            raise ValueError("Studio body mapping is incomplete: " + ", ".join(missing))
        self._ours_finger_ids = torch.tensor([body_names.index(name) for name in (*left, *right)], dtype=torch.long, device=self.device)
        self._ours_wrist_ids = torch.tensor([body_names.index("L_Wrist"), body_names.index("R_Wrist")], dtype=torch.long, device=self.device)
        self._ours_tip_ids = torch.tensor([body_names.index(name) for name in tips], dtype=torch.long, device=self.device)
        self._ours_reward_indices = reward_body_indices(
            body_names,
            device=self.device,
            hand_pos_weight=float(self._ours_reward_weights["hand_pos_reward_weight"]),
            hand_velocity_weight=float(self._ours_reward_weights["hand_velocity_reward_weight"]),
            hand_rot_weight=float(self._ours_reward_weights["hand_rot_reward_weight"]),
        )
        self._ours_feet_ids = self._ours_reward_indices["feet"]
        if not torch.equal(self._ours_finger_ids, self._ours_reward_indices["finger"]):
            raise ValueError("observation and reward finger ordering differ")
        ref = self._ours_reference_np
        reference_finger_names = tuple(
            str(name) for name in ref["finger_contact_body_names"].reshape(-1)
        )
        reference_finger = np.asarray(ref["finger_contact_labels"])
        if (
            reference_finger.shape != (self._target_joint_qpos.shape[0], 30)
            or reference_finger.dtype != np.bool_
        ):
            raise ValueError("Studio finger contact labels must be bool [T,30]")
        reference_finger_order = canonical_finger_segment_order(reference_finger_names)
        self._ours_finger_contact_reference = torch.tensor(
            reference_finger[:, reference_finger_order],
            dtype=torch.bool,
            device=self.device,
        )
        if self._ours_needs_part_attribution:
            part_names = load_finger_contact_part_link_names(ref)[
                :, reference_finger_order
            ]
            part_ids = np.full(part_names.shape, -1, dtype=np.int64)
            for name in np.unique(part_names):
                if name:
                    part_ids[part_names == name] = object_names.index(str(name))
            self._ours_finger_contact_part_ids = torch.tensor(
                part_ids, dtype=torch.long, device=self.device,
            )
        else:
            self._ours_finger_contact_part_ids = None

        reference_names = [str(name) for name in self._ours_reference_np["body_names"].reshape(-1)]
        if set(reference_names) != set(object_names) or len(reference_names) != len(object_names):
            raise ValueError("Studio reference and PHC object body maps differ")
        reference_order = [reference_names.index(name) for name in object_names]
        self._ours_ref_all_links = torch.tensor(np.concatenate((
            ref["object_link_pos"][:, reference_order],
            ref["object_link_rot_xyzw"][:, reference_order],
            ref["object_link_vel"][:, reference_order],
            ref["object_link_ang_vel"][:, reference_order],
        ), axis=-1), dtype=torch.float32, device=self.device)
        self._ours_ref_root_state = torch.tensor(np.concatenate((
            ref["object_root_pos"], ref["object_root_rot_xyzw"],
            ref["object_root_vel"], ref["object_root_ang_vel"],
        ), axis=-1), dtype=torch.float32, device=self.device)
        surface_names = [str(name) for name in ref["collision_surface_point_link_names"].reshape(-1)]
        if set(surface_names).difference(object_names):
            raise ValueError("Studio surface geometry names are absent from the PHC object")
        if self._ours_object_surface == "active_part" and set(surface_names) != {self._ours_child_link}:
            raise ValueError("Studio active_part surface must contain only the active child link")
        surface_points = np.asarray(
            ref["collision_surface_points_link_local_scaled"], dtype=np.float32
        )
        contact_points = np.asarray(
            self._phc_object_config["contact_region_points"], dtype=np.float32
        )
        contact_names = [
            str(name)
            for name in self._phc_object_config["contact_region_point_link_names"]
        ]
        if (
            contact_names != surface_names
            or not np.array_equal(contact_points, surface_points)
            or tuple(self._phc_contact_region_names)
            != tuple(dict.fromkeys(surface_names))
        ):
            raise ValueError(
                "Studio PHC contact gates must use the selected collision surface"
            )
        self._ours_surface_points_local = torch.tensor(surface_points, dtype=torch.float32, device=self.device)
        self._ours_surface_link_ids = torch.tensor([object_names.index(name) for name in surface_names], dtype=torch.long, device=self.device)
        if self._ours_active_joint_torque_enabled:
            origin = np.asarray(
                ref["active_joint_origin_parent_local_scaled"], dtype=np.float32,
            )
            axis = np.asarray(
                ref["active_joint_axis_parent_local"], dtype=np.float32,
            )
            if (
                origin.shape != (3,)
                or axis.shape != (3,)
                or not np.isfinite(origin).all()
                or not np.isfinite(axis).all()
                or not np.isclose(np.linalg.norm(axis), 1.0, rtol=0.0, atol=1.0e-5)
            ):
                raise ValueError("Studio active joint frame is invalid")
            self._ours_active_joint_origin_parent_local = torch.tensor(
                origin, dtype=torch.float32, device=self.device,
            )
            self._ours_active_joint_axis_parent_local = torch.tensor(
                axis, dtype=torch.float32, device=self.device,
            )
        self._ours_reference_surface = self._ours_world_surface_points(self._ours_ref_all_links)
        finger_surface_names = [
            str(name)
            for name in ref["finger_contact_surface_point_link_names"].reshape(-1)
        ]
        if set(finger_surface_names).difference(object_names):
            raise ValueError("Studio finger contact surface names are absent from the object")
        if (
            self._ours_child_link not in set(finger_surface_names)
            or not set(finger_surface_names).difference({self._ours_child_link})
        ):
            raise ValueError("Studio finger contact surface must include active and fixed links")
        self._ours_finger_surface_points_local = torch.tensor(
            ref["finger_contact_surface_points_link_local_scaled"],
            dtype=torch.float32,
            device=self.device,
        )
        self._ours_finger_surface_link_ids = torch.tensor(
            [object_names.index(name) for name in finger_surface_names],
            dtype=torch.long,
            device=self.device,
        )
        self._ours_finger_surface_part_ids = torch.unique(
            self._ours_finger_surface_link_ids, sorted=True,
        )
        if self._ours_needs_part_attribution:
            for link_id in self._ours_finger_surface_part_ids:
                if int((self._ours_finger_surface_link_ids == link_id).sum()) < self._ours_part_knn_k:
                    raise ValueError(
                        "Studio part-aware finger surface has fewer points than part KNN count"
                    )
        self._ours_bbox = torch.tensor(
            np.concatenate((ref["policy_root_bbox"], ref["policy_active_child_bbox"]), axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        region_points = self._target_contact_region_positions()
        if region_points.ndim != 3 or region_points.shape[0] != self.num_envs or not region_points.shape[1] or region_points.shape[2] != 3:
            raise ValueError("Studio live active contact-region geometry must be [N,P,3]")
        if not torch.isfinite(region_points).all():
            raise FloatingPointError("Studio live active contact-region geometry is non-finite")

        checkpoint = torch_ext.load_checkpoint(str(self._ours_tracker_path))
        self._ours_pnn = load_pnn(
            checkpoint,
            num_prim=int(self.cfg["env"]["num_prim"]),
            has_lateral=bool(self.cfg["env"]["has_lateral"]),
            activation=self._ours_tracker_activation,
            device=self.device,
        )
        self._ours_mean = checkpoint["running_mean_std"]["running_mean"].float().to(self.device)
        self._ours_var = checkpoint["running_mean_std"]["running_var"].float().to(self.device)

        qref = self._target_joint_qpos[:, self._ours_active_dof:self._ours_active_dof + 1]
        phase = build_phase_progress_tables(qref.detach().cpu().numpy())
        self._ours_phase_start = torch.tensor(phase[0], dtype=torch.long, device=self.device)
        self._ours_phase_target = torch.tensor(phase[1], dtype=torch.float32, device=self.device)
        self._ours_phase_direction = torch.tensor(phase[2], dtype=torch.float32, device=self.device)
        self._ours_phase_active = torch.tensor(phase[3], dtype=torch.bool, device=self.device)
        self._ours_phase_predecessor = torch.tensor(
            build_phase_predecessor_table(phase[0], phase[3]),
            dtype=torch.long, device=self.device,
        )
        midpoint = 0.5 * (self.dof_limits_lower + self.dof_limits_upper)
        half_range = 0.5 * (self.dof_limits_upper - self.dof_limits_lower)
        ratio = float(self._ours_reward_weights["soft_dof_pos_limit"])
        self._ours_soft_dof_lower = midpoint - ratio * half_range
        self._ours_soft_dof_upper = midpoint + ratio * half_range
        self._ours_episode_start_frames = self._ours_frames().clone()
        self._ours_episode_start_qpos = self._target_dof_pos[
            :, self._ours_active_dof
        ].clone()
        self._ours_progress_phase_start = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device,
        )
        self._ours_progress_phase_entry_qpos = self._ours_episode_start_qpos.clone()
        self._ours_progress_gate_open = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self._ours_progress_all_previous_passed = torch.ones_like(
            self._ours_progress_gate_open,
        )
        self._ours_previous_active_qpos = self._ours_episode_start_qpos.clone()
        self._ours_previous_residual = torch.zeros((self.num_envs, 153), dtype=torch.float32, device=self.device)
        self._ours_previous_dof_velocity = torch.zeros_like(self._dof_vel)
        self._ours_previous_object_linear_velocity = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._ours_previous_object_angular_velocity = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._ours_previous_active_link_linear_velocity = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device,
        )
        self._ours_previous_active_link_angular_velocity = torch.zeros_like(
            self._ours_previous_active_link_linear_velocity,
        )
        self._ours_action_rate_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ours_human_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ours_object_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ours_contact_reset = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._ours_required_hand_contact = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device,
        )
        self._ours_live_hand_contact = torch.zeros_like(self._ours_required_hand_contact)
        self._ours_live_hand_region_contact = torch.zeros_like(
            self._ours_required_hand_contact
        )
        self._ours_reset_samples = torch.zeros((), dtype=torch.float64, device=self.device)
        self._ours_reset_frame_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self._ours_reset_event_frame_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self._ours_reset_diagnostic_sum = {}
        self._ours_action_replay = None if self._ours_action_replay_np is None else torch.tensor(
            self._ours_action_replay_np, dtype=torch.float32, device=self.device,
        )
        self._ours_final_action_replay = (
            None if self._ours_final_action_replay_np is None else torch.tensor(
                self._ours_final_action_replay_np,
                dtype=torch.float32,
                device=self.device,
            )
        )

    def _ours_reference_motion(self):
        # Same-frame lookup is deliberate; PHC's generic task observation uses t+1.
        motion_times = self.progress_buf.float() * self.dt + self._motion_start_times + self._motion_start_times_offset
        reference = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)
        frames = torch.round(motion_times * float(self._target_qpos_fps)).long().clamp(0, self._ours_ref_root_state.shape[0] - 1)
        return reference, frames

    def _ours_frames(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return self._target_frame_indices(env_ids).clamp(0, self._ours_ref_root_state.shape[0] - 1)

    def _ours_teacher(self, reference, body_observation):
        teacher_times = (
            (self.progress_buf.float() + self._ours_teacher_frame_offset) * self.dt
            + self._motion_start_times
            + self._motion_start_times_offset
        )
        self._ours_teacher_input_frames = torch.round(
            teacher_times * float(self._target_qpos_fps)
        ).long().clamp(0, self._ours_ref_root_state.shape[0] - 1)
        if self._ours_teacher_frame_offset:
            reference = self._get_state_from_motionlib_cache(
                self._sampled_motion_ids, teacher_times, self._global_offset,
            )
        body_ids = self._track_bodies_id
        task_observation = compute_imitation_observations_v6(
            self._rigid_body_pos[:, 0],
            self._rigid_body_rot[:, 0],
            self._rigid_body_pos.index_select(1, body_ids),
            self._rigid_body_rot.index_select(1, body_ids),
            self._rigid_body_vel.index_select(1, body_ids),
            self._rigid_body_ang_vel.index_select(1, body_ids),
            reference["rg_pos"].index_select(1, body_ids),
            reference["rb_rot"].index_select(1, body_ids),
            reference["body_vel"].index_select(1, body_ids),
            reference["body_ang_vel"].index_select(1, body_ids),
            1,
            self._has_upright_start,
        )
        observation = torch.cat((body_observation, task_observation), dim=-1)
        if observation.shape != (self.num_envs, 2026):
            raise ValueError(f"Studio PHC-X teacher observation must be [N,2026], got {observation.shape}")
        self._ours_teacher_observation = observation
        self._ours_teacher_normalized_observation = normalize_teacher_observation(
            observation, self._ours_mean, self._ours_var,
        )
        return teacher_action(
            self._ours_pnn,
            observation,
            self._ours_mean,
            self._ours_var,
            int(self.cfg["env"]["training_prim"]),
        )

    def _compute_observations(self, env_ids=None):
        if not hasattr(self, "_ours_pnn"):
            return self.obs_buf
        if env_ids is not None and len(env_ids) > 0:
            # PHC's reset squash step can leave stale forces in the refreshed view.
            self._contact_forces[env_ids] = 0.0
        reference, reference_frames = self._ours_reference_motion()
        frames = self._ours_frames()
        teacher_body_observation = self._compute_humanoid_obs()
        teacher = self._ours_teacher(reference, teacher_body_observation)
        # The Studio PHC-X teacher keeps its native local-root self observation,
        # while the reference residual actor was trained with localRootObs=False.
        # Only the root-rotation encoding differs; all tensors remain Studio-owned.
        residual_body_observation = compute_humanoid_observations_max(
            self._rigid_body_pos,
            self._rigid_body_rot,
            self._rigid_body_vel,
            self._rigid_body_ang_vel,
            False,
            True,
        )
        expected_teacher_frames = (
            frames + self._ours_teacher_frame_offset
        ).clamp(0, self._ours_ref_root_state.shape[0] - 1)
        if (
            not torch.equal(frames, reference_frames)
            or not torch.equal(expected_teacher_frames, self._ours_teacher_input_frames)
        ):
            raise RuntimeError("ours state/reference/teacher frames diverged")
        object_links = torch.stack((
            self._target_states,
            self._rigid_body_state_reshaped[:, self._ours_child_body_id],
        ), dim=1)
        active = self._target_dof_pos[:, self._ours_active_dof:self._ours_active_dof + 1]
        active_reference = self._target_qpos_ref_for_frames(frames)[:, self._ours_active_dof:self._ours_active_dof + 1]
        # GT actor root 与 GT active child 沿用 live object_links 的同一双-slot 顺序。
        reference_object_links = torch.stack((
            self._ours_ref_root_state[frames],
            self._ours_ref_all_links[frames, self._ours_child_object_index],
        ), dim=1)
        observation = build_observation(
            human={
                "body_observation": residual_body_observation,
                "root_pos": self._rigid_body_pos[:, 0],
                "root_rot": self._rigid_body_rot[:, 0],
                "dof_pos": self._dof_pos,
                "dof_vel": self._dof_vel,
                "finger_body_state": self._rigid_body_state_reshaped[:, self._ours_finger_ids].view(self.num_envs, 2, 15, 13),
                "wrist_body_state": self._rigid_body_state_reshaped[:, self._ours_wrist_ids],
            },
            object_state={
                "link_state": object_links,
                "active_qpos": active,
            },
            reference={
                "dof_pos": reference["dof_pos"],
                "active_qpos": active_reference,
                "human_root_pos": reference["rg_pos"][:, 0],
                "human_root_rot": reference["rb_rot"][:, 0],
                "object_root_state": self._ours_ref_root_state[frames],
                "object_link_state": reference_object_links,
            },
            teacher_action=teacher,
            contact={"fingertip_force": self._contact_forces[:, self._ours_tip_ids]},
        )
        self._ours_reference = reference
        self._ours_reference_lookup_frames = reference_frames
        self._ours_teacher_action = teacher
        self._ours_object_links = object_links
        if env_ids is None:
            self.obs_buf[:] = observation
        else:
            self.obs_buf[env_ids] = observation[env_ids]
        return observation

    def pre_physics_step(self, actions):
        if not hasattr(self, "_ours_pnn"):
            return super().pre_physics_step(actions)
        residual = actions.to(self.device)
        if self._ours_diagnostic_zero_residual:
            residual = torch.zeros_like(residual)
        if self._ours_action_replay is not None:
            if self._ours_batch_replay:
                if self._ours_action_replay_index:
                    raise RuntimeError("bounded batch action replay is one step only")
                residual = self._ours_action_replay
                self._ours_action_replay_index = 1
            elif self._ours_action_replay_index < self._ours_action_replay.shape[0]:
                residual = self._ours_action_replay[self._ours_action_replay_index].unsqueeze(0)
                self._ours_action_replay_index += 1
            elif len(self._ours_closed_loop_rows) < self._ours_closed_loop_trace_steps:
                raise RuntimeError("bounded action replay ended before the requested trace")
        final = compose_residual_action(
            self._ours_teacher_action, residual, self._ours_residual_scale
        )
        if self._ours_final_action_replay is not None:
            if self._ours_batch_replay:
                if self._ours_final_action_replay_index:
                    raise RuntimeError("bounded batch final-action replay is one step only")
                final = self._ours_final_action_replay
                self._ours_final_action_replay_index = 1
            elif self._ours_final_action_replay_index >= self._ours_final_action_replay.shape[0]:
                raise RuntimeError("bounded final-action replay ended before the trace")
            else:
                final = self._ours_final_action_replay[
                    self._ours_final_action_replay_index
                ].unsqueeze(0)
                self._ours_final_action_replay_index += 1
        self._ours_capture_closed_loop_pre(residual, final)
        self._ours_last_residual = residual.clamp(-1.0, 1.0).detach().clone()
        super().pre_physics_step(final)
        if self._ours_batch_step_trace_path is not None and self._ours_batch_step_pre is None:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self._ours_batch_step_pre = self._ours_batch_step_state()
            solver_body = torch.cat((
                self._rigid_body_pos, self._rigid_body_rot,
                self._rigid_body_vel, self._rigid_body_ang_vel,
            ), dim=-1)
            solver_object = self._rigid_body_state_reshaped[:, [
                self._ours_root_body_id, self._ours_child_body_id,
            ]]
            self._ours_batch_step_pre.update({
                "teacher_observation": self._ours_numpy(
                    self._ours_teacher_observation
                ),
                "teacher_normalized_observation": self._ours_numpy(
                    self._ours_teacher_normalized_observation
                ),
                "teacher_action": self._ours_numpy(self._ours_teacher_action),
                "residual_action": self._ours_numpy(residual.clamp(-1.0, 1.0)),
                "final_action": self._ours_numpy(final),
                "pd_target": self._ours_numpy(self._action_to_pd_targets(final)),
                "solver_body_state": self._ours_numpy(solver_body),
                "solver_object_link_state": self._ours_numpy(solver_object),
            })
        if self._ours_closed_loop_pending is not None:
            # Read the actual PhysX FK state after reset/action setters and
            # immediately before the first simulate call.  This distinguishes
            # reference-state observation caches from collision-solver input.
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            solver_body = torch.cat((
                self._rigid_body_pos, self._rigid_body_rot,
                self._rigid_body_vel, self._rigid_body_ang_vel,
            ), dim=-1)
            solver_object = self._rigid_body_state_reshaped[:, [
                self._ours_root_body_id, self._ours_child_body_id,
            ]]
            self._ours_closed_loop_pending.update({
                "solver_pre_body_state": self._ours_numpy(solver_body[0]),
                "solver_pre_object_actor_state": self._ours_numpy(
                    self._target_states[0]
                ),
                "solver_pre_object_link_state": self._ours_numpy(
                    solver_object[0]
                ),
                "solver_pre_dof_pos": self._ours_numpy(self._dof_pos[0]),
                "solver_pre_dof_vel": self._ours_numpy(self._dof_vel[0]),
            })

    def _physics_step(self):
        if self._ours_closed_loop_pending is None and not self._ours_batch_replay:
            return super()._physics_step()
        substeps = []
        # Match Humanoid._physics_step for Studio's isaac_pd path, inserting
        # read-only snapshots immediately after each simulation substep.
        self.render(i=0)
        for _ in range(self.control_freq_inv):
            if not self.paused and self.enable_viewer_sync:
                if self.control_mode == "pd":
                    self.torques = self._compute_torques(self.actions)
                    self.gym.set_dof_actuation_force_tensor(
                        self.sim,
                        gymtorch.unwrap_tensor(
                            self._pad_target_dofs(self.torques)
                        ),
                    )
                self.gym.simulate(self.sim)
                if self.device == "cpu":
                    self.gym.fetch_results(self.sim, True)
                substeps.append(self._ours_substep_physics_state())
        if len(substeps) != self.control_freq_inv:
            raise RuntimeError("Studio substep audit did not execute every substep")
        axis = 1 if self._ours_batch_replay else 0
        stacked = {
            name: np.stack([state[name] for state in substeps], axis=axis)
            for name in substeps[0]
        }
        if self._ours_batch_replay:
            self._ours_batch_substeps = stacked
        else:
            for name, value in stacked.items():
                self._ours_closed_loop_pending["substep_" + name] = value

    def _ours_substep_physics_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        body_state = torch.cat((
            self._rigid_body_pos, self._rigid_body_rot,
            self._rigid_body_vel, self._rigid_body_ang_vel,
        ), dim=-1)
        object_links = self._rigid_body_state_reshaped[:, [
            self._ours_root_body_id, self._ours_child_body_id,
        ]]
        take = slice(None) if self._ours_batch_replay else 0
        return {
            "body_state": self._ours_numpy(body_state[take]),
            "dof_pos": self._ours_numpy(self._dof_pos[take]),
            "dof_vel": self._ours_numpy(self._dof_vel[take]),
            "dof_force": self._ours_numpy(self.dof_force_tensor[take]),
            "body_contact_force": self._ours_numpy(self._contact_forces[take]),
            "object_root_state": self._ours_numpy(self._target_states[take]),
            "object_link_state": self._ours_numpy(object_links[take]),
            "object_dof_pos": self._ours_numpy(self._target_dof_pos[take]),
            "object_dof_vel": self._ours_numpy(self._target_dof_vel[take]),
        }

    def _ours_world_surface_points(self, object_link_state):
        link_state = object_link_state.index_select(1, self._ours_surface_link_ids)
        points = self._ours_surface_points_local.unsqueeze(0).expand(object_link_state.shape[0], -1, -1)
        world = quat_apply(link_state[..., 3:7].reshape(-1, 4), points.reshape(-1, 3)).reshape(object_link_state.shape[0], -1, 3)
        return world + link_state[..., :3]

    def _ours_world_finger_contact_surface(self, object_link_state):
        link_state = object_link_state.index_select(
            1, self._ours_finger_surface_link_ids,
        )
        points = self._ours_finger_surface_points_local.unsqueeze(0).expand(
            object_link_state.shape[0], -1, -1,
        )
        world = quat_apply(
            link_state[..., 3:7].reshape(-1, 4), points.reshape(-1, 3),
        ).reshape(object_link_state.shape[0], -1, 3)
        return world + link_state[..., :3]

    def _ours_active_joint_frame(self):
        parent = self._rigid_body_state_reshaped[:, self._ours_parent_body_id]
        rotation = parent[:, 3:7] / parent[:, 3:7].norm(
            dim=-1, keepdim=True,
        ).clamp_min(1.0e-9)
        origin = parent[:, :3] + quat_apply(
            rotation,
            self._ours_active_joint_origin_parent_local.expand_as(parent[:, :3]),
        )
        axis = quat_apply(
            rotation,
            self._ours_active_joint_axis_parent_local.expand_as(parent[:, :3]),
        )
        return origin, axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)

    def _compute_reward(self, actions):
        reference, reference_frames = self._ours_reference_motion()
        frames = self._ours_frames()
        if not torch.equal(frames, reference_frames):
            raise RuntimeError("ours state and reward-reference frames diverged")
        active = self._target_dof_pos[:, self._ours_active_dof:self._ours_active_dof + 1]
        active_reference = self._target_qpos_ref_for_frames(frames)[:, self._ours_active_dof:self._ours_active_dof + 1]
        active_velocity = self._target_dof_vel[:, self._ours_active_dof:self._ours_active_dof + 1]
        reference_velocity = self._target_joint_qvel[
            frames, self._ours_active_dof:self._ours_active_dof + 1,
        ]
        actual_contact = body_contact_indicator(self._contact_forces)
        current_links = self._rigid_body_state_reshaped[:, self._ours_all_object_ids]
        current_surface = self._ours_world_surface_points(current_links)
        full_object_surface = self._ours_world_finger_contact_surface(current_links)
        finger_state = self._rigid_body_state_reshaped[:, self._ours_finger_ids]
        reference_finger_contact = self._ours_finger_contact_reference[frames]
        reference_hand_contact = self._phc_contact_labels_hand2[frames].bool()
        part_aware = self._ours_part_aware_contact_reward_enabled
        if self._ours_needs_part_attribution:
            if self._ours_active_joint_torque_enabled:
                (
                    whole_finger_object_distance, part_distance, selected_part_ids,
                    active_part_nearest_point,
                ) = joint_region_and_part_surface_distances_chunked(
                    finger_state[..., :3],
                    full_object_surface,
                    self._ours_finger_surface_link_ids,
                    self._ours_finger_surface_part_ids,
                    self._ours_finger_contact_part_ids[frames],
                    part_knn_k=self._ours_part_knn_k,
                    return_selected_nearest_point=True,
                )
            else:
                (
                    whole_finger_object_distance, part_distance, selected_part_ids,
                ) = joint_region_and_part_surface_distances_chunked(
                    finger_state[..., :3],
                    full_object_surface,
                    self._ours_finger_surface_link_ids,
                    self._ours_finger_surface_part_ids,
                    self._ours_finger_contact_part_ids[frames],
                    part_knn_k=self._ours_part_knn_k,
                )
            target_part_selected = (
                selected_part_ids == self._ours_finger_contact_part_ids[frames]
            )
        else:
            # 关闭 part-aware 时只执行历史 whole-object 距离路径。
            whole_finger_object_distance = joint_region_surface_distances_chunked(
                finger_state[..., :3], full_object_surface,
            )
            part_distance = whole_finger_object_distance
            target_part_selected = None
        # Coarse hand reward 始终沿用 whole-object 距离，不改变粗粒度语义。
        hand_object_distance = whole_hand_object_distance(whole_finger_object_distance)
        finger_force = self._contact_forces.index_select(1, self._ours_finger_ids)
        finger_object_distance, finger_binary_contact_distance = finger_part_aware_distances(
            whole_finger_object_distance,
            part_distance,
            reference_finger_contact,
            target_part_selected=target_part_selected,
            part_aware=part_aware,
        )
        # Reset 继续使用旧的 whole-object live contact；part-aware 只改变 reward。
        live_finger_contact = finger_object_contact_indicator(
            whole_finger_object_distance,
            finger_force,
            distance_threshold=float(self._ours_reward_weights["finger_contact_distance"]),
            force_threshold=float(self._ours_reward_weights["finger_contact_force"]),
        )
        # 旧 telemetry 保持 whole-object 语义，统一评测口径不随 reward 开关变化。
        self._ours_finger_object_distance = whole_finger_object_distance
        self._ours_reward_finger_object_distance = finger_object_distance
        self._ours_reward_finger_contact = live_finger_contact
        if part_aware:
            self._ours_reward_finger_contact = finger_object_contact_indicator(
                finger_binary_contact_distance,
                finger_force,
                distance_threshold=float(self._ours_reward_weights["finger_contact_distance"]),
                force_threshold=float(self._ours_reward_weights["finger_contact_force"]),
            )
        self._ours_hand_object_distance = hand_object_distance
        self._ours_live_finger_contact = live_finger_contact
        episode_start = self._ours_episode_start_frames
        phase_start = self._ours_phase_start[frames, 0]
        phase_active = self._ours_phase_active[frames, 0]
        phase_direction = self._ours_phase_direction[frames, 0]
        phase_target = self._ours_phase_target[frames, 0]
        phase_predecessor = self._ours_phase_predecessor[frames, 0]
        # 新阶段入口只看上一阶段在切换时的实际完成度，不记录历史最大值。
        # cap 只扩展当前阶段的 reward；前序门始终按 canonical [0,1] 完成率判断，
        # 避免超过 GT 终点的 overshoot 被用来补足未完成的 reference 行程。
        previous_phase_index = self._ours_progress_phase_start.clamp_min(0)
        previous_phase_completion = phase_relative_progress(
            self._ours_previous_active_qpos,
            self._ours_progress_phase_entry_qpos,
            self._target_joint_qpos[
                previous_phase_index, self._ours_active_dof
            ],
            self._ours_phase_target[previous_phase_index, 0],
            self._ours_phase_direction[previous_phase_index, 0],
            self._ours_progress_phase_start >= 0,
            target_cap_scale=1.0,
        )
        gate, phase_id, all_previous_passed, phase_changed = (
            phase_prerequisite_transition(
                phase_start, phase_predecessor, phase_active,
                self._ours_progress_phase_start, previous_phase_completion,
                self._ours_progress_gate_open,
                self._ours_progress_all_previous_passed,
                mode=self._ours_progress_prerequisite_mode,
                threshold=self._ours_progress_prerequisite_threshold,
            )
        )
        phase_entry_qpos = torch.where(
            phase_changed,
            self._ours_previous_active_qpos,
            self._ours_progress_phase_entry_qpos,
        )
        reference_start_qpos = self._target_joint_qpos[
            phase_start, self._ours_active_dof
        ]
        phase_completion = phase_relative_progress(
            active[:, 0],
            phase_entry_qpos,
            reference_start_qpos,
            phase_target,
            phase_direction,
            phase_active,
            target_cap_scale=self._ours_progress_target_cap_scale,
        )
        self._ours_progress_phase_start = phase_id
        self._ours_progress_phase_entry_qpos = torch.where(
            phase_changed, phase_entry_qpos, self._ours_progress_phase_entry_qpos,
        )
        self._ours_progress_gate_open = gate
        self._ours_progress_all_previous_passed = all_previous_passed
        progress_gate = gate & phase_active
        progress_bonus = phase_completion
        progress_overshoot = torch.zeros_like(phase_completion)
        if self._ours_progress_mode == "reference_paced_completion":
            progress_bonus, progress_overshoot = reference_paced_progress(
                phase_completion,
                active[:, 0],
                phase_entry_qpos,
                active_reference[:, 0],
                reference_start_qpos,
                phase_target,
                phase_direction,
                phase_active,
                lead_tolerance=self._ours_progress_lead_tolerance,
            )
        residual = self._ours_last_residual if self._ours_last_residual is not None else torch.zeros_like(self._ours_previous_residual)
        acceleration_valid = frames - episode_start > 2
        acceleration_scale = float(self._target_qpos_fps)
        active_link_state = current_links[:, self._ours_child_object_index]
        reference_active_link_state = self._ours_ref_all_links[
            frames, self._ours_child_object_index
        ]
        contact = {
            "reference_hand_contact": reference_hand_contact,
            "reference_finger_contact": reference_finger_contact,
            "hand_object_distance": hand_object_distance,
            "finger_whole_object_distance": whole_finger_object_distance,
            "finger_object_distance": finger_object_distance,
            "finger_binary_contact_distance": finger_binary_contact_distance,
            "finger_contact_force": finger_force,
            "contact_force": self._contact_forces,
        }
        articulation = {
            "active_qpos": active,
            "reference_active_qpos": active_reference,
            "active_qvel": active_velocity,
            "reference_active_qvel": reference_velocity,
            "active_progress": phase_completion,
            "active_progress_bonus": progress_bonus,
            "active_progress_overshoot": progress_overshoot,
            "progress_gate_open": progress_gate,
        }
        if self._ours_active_joint_torque_enabled:
            selected_distance = torch.linalg.norm(
                finger_state[..., :3] - active_part_nearest_point, dim=-1,
            )
            origin, axis = self._ours_active_joint_frame()
            contact.update({
                "finger_contact_points": active_part_nearest_point,
                "finger_active_joint_contact": (
                    selected_distance
                    < float(self._ours_reward_weights["finger_contact_distance"])
                ) & (
                    finger_force.abs().amax(dim=-1)
                    > float(self._ours_reward_weights["finger_contact_force"])
                ) & (selected_part_ids == self._ours_child_object_index),
                "reference_active_joint_contact": (
                    reference_finger_contact
                    & (
                        self._ours_finger_contact_part_ids[frames]
                        == self._ours_child_object_index
                    )
                ),
            })
            articulation.update({
                "active_joint_origin_world": origin,
                "active_joint_axis_world": axis,
                "active_joint_phase_direction": phase_direction,
                "active_joint_phase_active": phase_active,
            })
        reward, terms = compute_reward(
            human={
                "body_pos": self._rigid_body_pos,
                "body_rot": self._rigid_body_rot,
                "body_vel": self._rigid_body_vel,
                "body_ang_vel": self._rigid_body_ang_vel,
                "reference_body_pos": reference["rg_pos"],
                "reference_body_rot": reference["rb_rot"],
                "reference_body_vel": reference["body_vel"],
                "reference_body_ang_vel": reference["body_ang_vel"],
                "reference_surface_points": self._ours_reference_surface[frames],
                "dof_acceleration": (self._dof_vel - self._ours_previous_dof_velocity) * acceleration_scale,
                "acceleration_valid": acceleration_valid,
            },
            object_state={
                "root_state": self._target_states,
                "reference_root_state": self._ours_ref_root_state[frames],
                "linear_acceleration": (self._target_states[:, 7:10] - self._ours_previous_object_linear_velocity) * acceleration_scale,
                "angular_acceleration": (self._target_states[:, 10:13] - self._ours_previous_object_angular_velocity) * acceleration_scale,
                "active_link_state": active_link_state,
                "reference_active_link_state": reference_active_link_state,
                # active child 的平滑项只使用 live PhysX 速度，不从 noisy reference 求二阶差分。
                "active_link_linear_acceleration": (
                    active_link_state[:, 7:10]
                    - self._ours_previous_active_link_linear_velocity
                ) * acceleration_scale,
                "active_link_angular_acceleration": (
                    active_link_state[:, 10:13]
                    - self._ours_previous_active_link_angular_velocity
                ) * acceleration_scale,
            },
            contact=contact,
            articulation=articulation,
            regularization={
                "feet_velocity_xy": self._rigid_body_vel[:, self._ours_feet_ids, :2],
                "feet_in_contact": self._contact_forces[:, self._ours_feet_ids, 2] > 1.0,
                "dof_pos": self._dof_pos,
                "soft_dof_lower": self._ours_soft_dof_lower,
                "soft_dof_upper": self._ours_soft_dof_upper,
                "residual_action": residual,
                "previous_residual_action": self._ours_previous_residual,
                "action_rate_valid": self._ours_action_rate_valid,
            },
            indices=self._ours_reward_indices,
            weights=self._ours_reward_weights,
        )
        self.rew_buf[:] = reward
        self._ours_required_hand_contact = reference_hand_contact
        self._ours_live_hand_region_contact = live_finger_contact.reshape(
            self.num_envs, 2, 15,
        ).any(dim=-1)
        self._ours_live_hand_contact = self._ours_live_hand_region_contact
        self._ours_human_reset, self._ours_object_reset, self._ours_contact_reset = reward_reset_signals(
            self._rigid_body_pos,
            reference["rg_pos"],
            current_surface,
            self._ours_reference_surface[frames],
            actual_contact,
            torch.zeros_like(actual_contact),
            self._ours_reward_indices,
            self._ours_contact_reset,
            float(self._ours_reset["human_key_body_error"]),
            float(self._ours_reset["object_surface_error"]),
            live_hand_contact=self._ours_live_hand_region_contact,
            reference_hand_contact=self._ours_required_hand_contact,
        )
        if (
            self._ours_closed_loop_trace_path is not None
            or self._ours_batch_step_trace_path is not None
        ):
            self._ours_last_reward_terms = {name: value.detach().clone() for name, value in terms.items()}
        self._ours_previous_residual = residual.detach().clone()
        # Isaac Gym refreshes the live state tensors in place.  ``detach()``
        # alone would keep an alias to the current frame, making every
        # acceleration difference zero on the following step.
        self._ours_previous_dof_velocity = self._dof_vel.detach().clone()
        self._ours_previous_object_linear_velocity = self._target_states[:, 7:10].detach().clone()
        self._ours_previous_object_angular_velocity = self._target_states[:, 10:13].detach().clone()
        self._ours_previous_active_link_linear_velocity = (
            active_link_state[:, 7:10].detach().clone()
        )
        self._ours_previous_active_link_angular_velocity = (
            active_link_state[:, 10:13].detach().clone()
        )
        self._ours_previous_active_qpos = active[:, 0].detach().clone()
        self._ours_action_rate_valid[:] = True
        self._ours_reward_count += 1
        opening = phase_active & (phase_direction > 0.0)
        opening_weight = opening.to(dtype=torch.float64)
        self._ours_opening_q_sum += (
            terms["active_hinge_q"].detach().to(dtype=torch.float64)
            * opening_weight
        ).sum()
        self._ours_opening_reference_q_sum += (
            terms["reference_hinge_q"].detach().to(dtype=torch.float64)
            * opening_weight
        ).sum()
        self._ours_opening_q_error_sum += (
            terms["active_hinge_q_error"].detach().to(dtype=torch.float64)
            * opening_weight
        ).sum()
        self._ours_opening_q_count += opening.sum(dtype=torch.int64)
        values = {"total": reward, **terms}
        ranged_names = {
            "active_hinge_q",
            "reference_hinge_q",
            "active_hinge_q_error",
            "progress_metric",
        }
        for name, value in values.items():
            if name not in self._ours_reward_sum:
                self._ours_reward_sum[name] = torch.zeros((), device=self.device)
            self._ours_reward_sum[name] += value.detach().mean()
            if name not in ranged_names:
                continue
            if name not in self._ours_reward_min:
                self._ours_reward_min[name] = torch.full(
                    (), float("inf"), device=self.device,
                )
                self._ours_reward_max[name] = torch.full(
                    (), float("-inf"), device=self.device,
                )
            self._ours_reward_min[name] = torch.minimum(
                self._ours_reward_min[name], value.detach().amin(),
            )
            self._ours_reward_max[name] = torch.maximum(
                self._ours_reward_max[name], value.detach().amax(),
            )
        progress = terms["progress_metric"].detach().reshape(-1)
        peak_progress, peak_index = progress.max(dim=0)
        peak_q = terms["active_hinge_q"].detach().reshape(-1).gather(
            0, peak_index.reshape(1),
        )[0]
        if self._ours_peak_progress is None:
            self._ours_peak_progress = peak_progress
            self._ours_hinge_q_at_peak_progress = peak_q
        else:
            improved = peak_progress > self._ours_peak_progress
            self._ours_peak_progress = torch.where(
                improved, peak_progress, self._ours_peak_progress,
            )
            self._ours_hinge_q_at_peak_progress = torch.where(
                improved, peak_q, self._ours_hinge_q_at_peak_progress,
            )

    def _compute_reset(self):
        if not hasattr(self, "_ours_pnn"):
            return super()._compute_reset()
        if not torch.isfinite(self.obs_buf).all():
            raise FloatingPointError("ours observation contains NaN/Inf")

        frames = self._ours_frames()
        started = frames > self._ours_episode_start_frames + 1
        # The Studio canonical world may place its physical ground away from
        # z=0.  The 0.3 m fall threshold is a clearance above that ground,
        # matching the source convention when its ground is at zero.
        ground_height = float(self._phc_object_config["ground_height"])
        root_fall = (
            self._rigid_body_pos[:, 0, 2] - ground_height
            < float(self._ours_reset["termination_height"])
        )
        root_fall &= self.progress_buf > 1
        kinematic = started & (
            self._ours_human_reset
            | self._ours_object_reset
        )
        contact = started & self._ours_contact_reset.ge(
            int(self._ours_reset["required_hand_contact_mismatch_steps"])
        ).any(dim=-1)

        # 训练沿用全部失败 reset；fixed-horizon 评测只记录失败，不重置轨迹。
        failed = (kinematic | contact | root_fall) & self._enable_early_termination
        end_of_motion = frames >= self._ours_ref_root_state.shape[0] - 1
        end_of_rollout = self.progress_buf >= self.max_episode_length - 1
        self._ours_reset_reason_root_fall = root_fall & self._enable_early_termination
        self._ours_reset_reason_kinematic = kinematic
        self._ours_reset_reason_contact = contact
        self._ours_reset_reason_motion_end = end_of_motion
        self._ours_reset_reason_rollout_end = end_of_rollout
        self._terminate_buf[:] = failed.to(dtype=self._terminate_buf.dtype)
        self.reset_buf[:] = (failed | end_of_motion | end_of_rollout).to(dtype=self.reset_buf.dtype)
        reset = self.reset_buf.bool()
        signals = {
            "human": started & self._ours_human_reset,
            "object": started & self._ours_object_reset,
            "contact": contact,
            "root_fall": root_fall & self._enable_early_termination,
            "kinematic": kinematic,
            "motion_end": end_of_motion,
            "rollout_end": end_of_rollout,
            "terminated": self._terminate_buf.bool(),
            "reset": reset,
        }
        for side, index in (("left", 0), ("right", 1)):
            required = self._ours_required_hand_contact[:, index]
            live = self._ours_live_hand_contact[:, index]
            live_region = self._ours_live_hand_region_contact[:, index]
            signals["required_" + side] = required
            signals["live_" + side] = live
            signals["missing_" + side] = required & ~live
            signals["live_region_" + side] = live_region
            signals["missing_region_" + side] = required & ~live_region
            signals["reset_live_" + side] = live_region
            signals["contact_" + side] = started & self._ours_contact_reset[:, index].ge(
                int(self._ours_reset["required_hand_contact_mismatch_steps"])
            )
        for name, signal in signals.items():
            if name not in self._ours_reset_diagnostic_sum:
                self._ours_reset_diagnostic_sum[name] = torch.zeros(
                    (), dtype=torch.float64, device=self.device,
                )
            self._ours_reset_diagnostic_sum[name] += signal.sum(dtype=torch.float64)
        for lower in range(0, 150, 30):
            name = "frame_{:03d}_{:03d}".format(lower, lower + 29)
            if name not in self._ours_reset_diagnostic_sum:
                self._ours_reset_diagnostic_sum[name] = torch.zeros(
                    (), dtype=torch.float64, device=self.device,
                )
            self._ours_reset_diagnostic_sum[name] += (
                (frames >= lower) & (frames < lower + 30)
            ).sum(dtype=torch.float64)
        self._ours_reset_samples += float(self.num_envs)
        self._ours_reset_frame_sum += frames.sum(dtype=torch.float64)
        self._ours_reset_event_frame_sum += frames[reset].sum(dtype=torch.float64)

    def ours_reward_scalars(self, clear=True):
        if self._ours_reward_count == 0:
            return {}
        names = tuple(self._ours_reward_sum)
        values = torch.stack(tuple(self._ours_reward_sum[name] for name in names)) / self._ours_reward_count
        result = dict(zip(names, values.cpu().tolist()))
        for name in (
            "active_hinge_q",
            "reference_hinge_q",
            "active_hinge_q_error",
            "progress_metric",
        ):
            if name not in self._ours_reward_min:
                continue
            result[name + "_min"] = float(self._ours_reward_min[name].cpu())
            result[name + "_max"] = float(self._ours_reward_max[name].cpu())
        if self._ours_peak_progress is not None:
            result["progress_metric_peak"] = float(self._ours_peak_progress.cpu())
            result["hinge_q_at_peak_progress"] = float(
                self._ours_hinge_q_at_peak_progress.cpu()
            )
        if self._ours_opening_q_count.item() > 0:
            opening_count = self._ours_opening_q_count.to(dtype=torch.float64)
            result["opening_hinge_q"] = float(
                (self._ours_opening_q_sum / opening_count).cpu()
            )
            result["opening_reference_hinge_q"] = float(
                (self._ours_opening_reference_q_sum / opening_count).cpu()
            )
            result["opening_hinge_q_error"] = float(
                (self._ours_opening_q_error_sum / opening_count).cpu()
            )
            result["opening_hinge_q_sample_count"] = int(
                self._ours_opening_q_count.cpu()
            )
        if clear:
            self._ours_reward_sum.clear()
            self._ours_reward_min.clear()
            self._ours_reward_max.clear()
            self._ours_peak_progress = None
            self._ours_hinge_q_at_peak_progress = None
            self._ours_reward_count = 0
            self._ours_opening_q_sum.zero_()
            self._ours_opening_reference_q_sum.zero_()
            self._ours_opening_q_error_sum.zero_()
            self._ours_opening_q_count.zero_()
        return result

    def ours_reset_scalars(self, clear=True):
        """Return bounded per-update reset hazards and canonical-frame occupancy."""

        if self._ours_reset_samples.item() == 0:
            return {}
        samples = self._ours_reset_samples
        reset_events = self._ours_reset_diagnostic_sum["reset"].clamp_min(1.0)
        result = {
            "mean_frame": float((self._ours_reset_frame_sum / samples).item()),
            "mean_reset_frame": float(
                (self._ours_reset_event_frame_sum / reset_events).item()
            ),
        }
        result.update({
            "hazard_" + name: float((value / samples).item())
            for name, value in self._ours_reset_diagnostic_sum.items()
        })
        if clear:
            self._ours_reset_samples.zero_()
            self._ours_reset_frame_sum.zero_()
            self._ours_reset_event_frame_sum.zero_()
            self._ours_reset_diagnostic_sum.clear()
        return result

    def _reset_env_tensors(self, env_ids):
        humanoid_ids = self._humanoid_actor_ids[env_ids]
        target_ids = self._tar_actor_ids[env_ids]
        combined_ids = torch.cat((humanoid_ids, target_ids)).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(combined_ids),
            len(combined_ids),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(combined_ids),
            len(combined_ids),
        )
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        if not hasattr(self, "_ours_episode_start_frames") or len(env_ids) == 0:
            return
        frames = self._ours_frames()
        self._ours_episode_start_frames[env_ids] = frames[env_ids]
        self._ours_episode_start_qpos[env_ids] = self._target_dof_pos[
            env_ids, self._ours_active_dof
        ]
        self._ours_progress_phase_start[env_ids] = -1
        self._ours_progress_phase_entry_qpos[env_ids] = self._ours_episode_start_qpos[env_ids]
        self._ours_progress_gate_open[env_ids] = True
        self._ours_progress_all_previous_passed[env_ids] = True
        self._ours_previous_active_qpos[env_ids] = self._ours_episode_start_qpos[env_ids]
        self._ours_action_rate_valid[env_ids] = False
        self._ours_previous_residual[env_ids] = 0.0
        self._ours_previous_dof_velocity[env_ids] = 0.0
        self._ours_previous_object_linear_velocity[env_ids] = 0.0
        self._ours_previous_object_angular_velocity[env_ids] = 0.0
        self._ours_previous_active_link_linear_velocity[env_ids] = 0.0
        self._ours_previous_active_link_angular_velocity[env_ids] = 0.0
        self._ours_human_reset[env_ids] = False
        self._ours_object_reset[env_ids] = False
        self._ours_contact_reset[env_ids] = 0.0

    def _ours_capture_trace(self):
        if self._ours_trace_path is None or len(self._ours_trace_rows) >= 64:
            return
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        physics_frames = self._target_frame_indices(env_ids).clamp(0, self._ours_ref_root_state.shape[0] - 1)
        reference_frames = self._ours_reference_lookup_frames
        teacher_frames = self._ours_teacher_input_frames
        expected_teacher_frames = (
            physics_frames + self._ours_teacher_frame_offset
        ).clamp(0, self._ours_ref_root_state.shape[0] - 1)
        if (
            not torch.equal(physics_frames, reference_frames)
            or not torch.equal(expected_teacher_frames, teacher_frames)
        ):
            raise RuntimeError("ours state/reference/teacher frames diverged")
        self._ours_trace_rows.append({
            "physics_frame": int(physics_frames[0].item()),
            "reference_lookup_frame": int(reference_frames[0].item()),
            "teacher_input_frame": int(teacher_frames[0].item()),
            "episode_start_frame": int(self._ours_episode_start_frames[0].item()),
            "observation_shape": list(self.obs_buf.shape),
            "object_link_state_shape": list(self._ours_object_links.shape),
            "bbox_abs_sum": float(self._ours_bbox.abs().sum().item()),
        })
        if len(self._ours_trace_rows) == 64:
            self._ours_write_trace()

    def _ours_write_trace(self):
        if self._ours_trace_path is None or not self._ours_trace_rows:
            return
        self._ours_trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._ours_trace_path.write_text(json.dumps({
            "status": "PASS",
            "timeline": "same-frame",
            "trace": self._ours_trace_rows,
        }, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _ours_numpy(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)

    def _ours_closed_loop_state(self):
        body_state = torch.cat((self._rigid_body_pos, self._rigid_body_rot, self._rigid_body_vel, self._rigid_body_ang_vel), dim=-1)
        region_points = self._target_contact_region_positions()
        body_region_distance = torch.cdist(
            self._rigid_body_pos, region_points,
        ).amin(dim=-1)
        hand_region_distance = torch.stack(tuple(
            body_region_distance.index_select(
                1, self._ours_reward_indices[side]
            ).amin(dim=-1)
            for side in ("left_hand", "right_hand")
        ), dim=-1)
        return {
            "body_state": self._ours_numpy(body_state[0]),
            "wrist_state": self._ours_numpy(body_state[0, self._ours_wrist_ids]),
            "fingertip_state": self._ours_numpy(body_state[0, self._ours_tip_ids]),
            "dof_pos": self._ours_numpy(self._dof_pos[0]),
            "dof_vel": self._ours_numpy(self._dof_vel[0]),
            "body_contact_force": self._ours_numpy(self._contact_forces[0]),
            "wrist_contact_force": self._ours_numpy(
                self._contact_forces[0, self._ours_wrist_ids]
            ),
            "fingertip_contact_force": self._ours_numpy(
                self._contact_forces[0, self._ours_tip_ids]
            ),
            "body_region_distance": self._ours_numpy(body_region_distance[0]),
            "hand_region_distance": self._ours_numpy(hand_region_distance[0]),
            "fingertip_region_distance": self._ours_numpy(
                body_region_distance[0, self._ours_tip_ids]
            ),
            "dof_force": self._ours_numpy(self.dof_force_tensor[0]),
            "object_root_state": self._ours_numpy(self._target_states[0]),
            "object_link_state": self._ours_numpy(self._ours_object_links[0]),
            "object_dof_pos": self._ours_numpy(self._target_dof_pos[0]),
            "object_dof_vel": self._ours_numpy(self._target_dof_vel[0]),
        }

    def _ours_batch_step_state(self):
        body_state = torch.cat((
            self._rigid_body_pos, self._rigid_body_rot,
            self._rigid_body_vel, self._rigid_body_ang_vel,
        ), dim=-1)
        region_points = self._target_contact_region_positions()
        body_distance = torch.cdist(
            self._rigid_body_pos, region_points,
        ).amin(dim=-1)
        hand_distance = torch.stack(tuple(
            body_distance.index_select(
                1, self._ours_reward_indices[side]
            ).amin(dim=-1)
            for side in ("left_hand", "right_hand")
        ), dim=-1)
        return {
            "frame": self._ours_numpy(self._ours_reference_lookup_frames),
            "human_root_state": self._ours_numpy(self._humanoid_root_states),
            "body_state": self._ours_numpy(body_state),
            "wrist_state": self._ours_numpy(
                body_state.index_select(1, self._ours_wrist_ids)
            ),
            "fingertip_state": self._ours_numpy(
                body_state.index_select(1, self._ours_tip_ids)
            ),
            "dof_pos": self._ours_numpy(self._dof_pos),
            "dof_vel": self._ours_numpy(self._dof_vel),
            "body_contact_force": self._ours_numpy(self._contact_forces),
            "dof_force": self._ours_numpy(self.dof_force_tensor),
            "object_root_state": self._ours_numpy(self._target_states),
            "object_link_state": self._ours_numpy(self._ours_object_links),
            "object_dof_pos": self._ours_numpy(self._target_dof_pos),
            "object_dof_vel": self._ours_numpy(self._target_dof_vel),
            "hand_region_distance": self._ours_numpy(hand_distance),
            "body_region_distance": self._ours_numpy(body_distance),
            "fingertip_region_distance": self._ours_numpy(
                body_distance.index_select(1, self._ours_tip_ids)
            ),
            "region_center": self._ours_numpy(region_points.mean(dim=1)),
        }

    def _ours_write_batch_step_trace(self):
        if self._ours_batch_step_trace_path is None or self._ours_batch_step_pre is None:
            return
        post = self._ours_batch_step_state()
        post["body_contact_force"] = self._ours_numpy(self._contact_forces)
        post["total_reward"] = self._ours_numpy(self.rew_buf)
        for name, value in self._ours_last_reward_terms.items():
            post["reward_" + name] = self._ours_numpy(value)
        arrays = {"pre_" + name: value for name, value in self._ours_batch_step_pre.items()}
        arrays.update({"post_" + name: value for name, value in post.items()})
        if self._ours_batch_substeps is not None:
            arrays.update({
                "substep_" + name: value
                for name, value in self._ours_batch_substeps.items()
            })
        self._ours_batch_step_trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._ours_batch_step_trace_path, **arrays)
        self._ours_batch_step_trace_path = None

    def _ours_capture_closed_loop_pre(self, residual, final):
        if self._ours_closed_loop_trace_path is None or len(self._ours_closed_loop_rows) >= self._ours_closed_loop_trace_steps:
            return
        if self._ours_closed_loop_pending is not None:
            raise RuntimeError("closed-loop trace has an unconsumed pre-step row")
        pd_target = self._action_to_pd_targets(final).clone()
        if self._freeze_hand:
            for name in ("L_Hand", "R_Hand"):
                start = self._dof_names.index(name) * 3
                pd_target[:, start:start + 3] = 0
        if self._freeze_toe:
            for name in ("L_Toe", "R_Toe"):
                start = self._dof_names.index(name) * 3
                pd_target[:, start:start + 3] = 0
        row = {
            "pre_frame": self._ours_numpy(self._ours_reference_lookup_frames[0]),
            "pre_observation": self._ours_numpy(self.obs_buf[0]),
            "teacher_observation": self._ours_numpy(
                self._ours_teacher_observation[0]
            ),
            "teacher_normalized_observation": self._ours_numpy(
                self._ours_teacher_normalized_observation[0]
            ),
            "teacher_action": self._ours_numpy(self._ours_teacher_action[0]),
            "residual_action": self._ours_numpy(residual.clamp(-1.0, 1.0)[0]),
            "final_action": self._ours_numpy(final[0]),
            "pd_target": self._ours_numpy(pd_target[0]),
        }
        row.update({"pre_" + name: value for name, value in self._ours_closed_loop_state().items()})
        self._ours_closed_loop_pending = row

    def _ours_capture_closed_loop_post(self):
        if self._ours_closed_loop_pending is None:
            return
        row = self._ours_closed_loop_pending
        self._ours_closed_loop_pending = None
        row.update({"post_" + name: value for name, value in self._ours_closed_loop_state().items()})
        row["post_frame"] = self._ours_numpy(self._ours_reference_lookup_frames[0])
        row["post_observation"] = self._ours_numpy(self.obs_buf[0])
        row["total_reward"] = self._ours_numpy(self.rew_buf[0])
        row["human_reset"] = self._ours_numpy(self._ours_human_reset[0])
        row["object_reset"] = self._ours_numpy(self._ours_object_reset[0])
        row["required_contact_streak"] = self._ours_numpy(self._ours_contact_reset[0])
        row["root_fall_reset"] = self._ours_numpy(self._ours_reset_reason_root_fall[0])
        row["kinematic_reset"] = self._ours_numpy(self._ours_reset_reason_kinematic[0])
        row["contact_reset"] = self._ours_numpy(self._ours_reset_reason_contact[0])
        row["motion_end_reset"] = self._ours_numpy(self._ours_reset_reason_motion_end[0])
        row["rollout_end_reset"] = self._ours_numpy(self._ours_reset_reason_rollout_end[0])
        row["reset"] = self._ours_numpy(self.reset_buf[0])
        row["terminated"] = self._ours_numpy(self._terminate_buf[0])
        if self._ours_last_reward_terms is None:
            raise RuntimeError("closed-loop trace requires named reward terms")
        for name, value in self._ours_last_reward_terms.items():
            row["reward_" + name] = self._ours_numpy(value[0])
        row["reward_rig"] = np.asarray(1.0, dtype=np.float32)
        row["reward_r_handle_normal"] = np.asarray(1.0, dtype=np.float32)
        self._ours_closed_loop_rows.append(row)
        if len(self._ours_closed_loop_rows) == self._ours_closed_loop_trace_steps:
            self._ours_write_closed_loop_trace()

    def _ours_write_closed_loop_trace(self):
        if self._ours_closed_loop_trace_path is None or not self._ours_closed_loop_rows:
            return
        self._ours_closed_loop_trace_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.stack([row[name] for row in self._ours_closed_loop_rows]).astype(np.float32, copy=False)
            for name in self._ours_closed_loop_rows[0]
        }
        np.savez_compressed(self._ours_closed_loop_trace_path, **arrays)
        from scripts.physics.capture_isaac_runtime import write_task_physics
        write_task_physics(
            self,
            self._ours_closed_loop_trace_path.with_suffix(".physics.json"),
        )

    def _ours_capture_evaluation_telemetry(self):
        if not flags.im_eval:
            return
        rollout = self.extras["physics_rollout"]
        frames = self._ours_reference_lookup_frames
        finger_ids = self._ours_reward_indices["finger"]
        force_components = self._contact_forces.index_select(1, finger_ids)
        force = torch.linalg.norm(force_components, dim=-1)
        contact = self._ours_live_finger_contact
        reference_body_state = torch.cat((
            self._ours_reference["rg_pos"],
            self._ours_reference["rb_rot"],
            self._ours_reference["body_vel"],
            self._ours_reference["body_ang_vel"],
        ), dim=-1)
        rollout["ours_evaluation"] = {
            "reference_frame": frames.detach().cpu().numpy(),
            "reference_body_state": reference_body_state.detach().cpu().numpy(),
            "finger_reference_contact": self._ours_finger_contact_reference[frames].detach().cpu().numpy(),
            "finger_contact": contact.to(dtype=force.dtype).detach().cpu().numpy(),
            "finger_contact_force": force.detach().cpu().numpy(),
            "finger_object_distance": self._ours_finger_object_distance.detach().cpu().numpy(),
            "finger_part_aware_distance": self._ours_reward_finger_object_distance.detach().cpu().numpy(),
            "finger_part_aware_contact": self._ours_reward_finger_contact.to(
                dtype=force.dtype
            ).detach().cpu().numpy(),
            "hand_object_distance": self._ours_hand_object_distance.detach().cpu().numpy(),
            "finger_object_contact": contact.to(dtype=force.dtype).detach().cpu().numpy(),
            # 兼容旧分析器字段名；语义已升级为 full-object，而不是 handle-only。
            "finger_handle_distance": self._ours_finger_object_distance.detach().cpu().numpy(),
            "finger_handle_contact": contact.to(dtype=force.dtype).detach().cpu().numpy(),
            "hinge_phase_active": self._ours_phase_active[frames].detach().cpu().numpy(),
        }
    def finalize_ours_outputs(self):
        """Flush bounded diagnostics once at the agent/player lifecycle boundary."""

        if self._ours_outputs_finalized:
            return
        self._ours_write_trace()
        self._ours_write_closed_loop_trace()
        if flags.im_eval and self._ours_eval_path is not None:
            means = self.ours_reward_scalars(clear=False)
            if "total" not in means:
                raise RuntimeError("ours evaluation has no completed reward step")
            self._ours_eval_path.parent.mkdir(parents=True, exist_ok=True)
            self._ours_eval_path.write_text(json.dumps({
                "status": "PASS",
                "frame": int(self._ours_reference_lookup_frames[0].item()),
                "episode_steps": int(self._ours_reward_count),
                "episode_reward_mean": means["total"],
                "episode_named_reward_means": means,
            }, indent=2) + "\n", encoding="utf-8")
        self._ours_outputs_finalized = True

    def post_physics_step(self):
        super().post_physics_step()
        if not hasattr(self, "_ours_pnn"):
            return
        self._ours_capture_evaluation_telemetry()
        self._ours_capture_trace()
        self._ours_capture_closed_loop_post()
        self._ours_write_batch_step_trace()
