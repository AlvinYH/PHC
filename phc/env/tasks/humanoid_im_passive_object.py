"""PHC HumanoidIm task with one passive articulated object actor."""

from __future__ import annotations

import json
from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch
import torch.nn.functional as F

from phc.env.tasks.humanoid_im import HumanoidIm
from phc.utils.flags import flags


class HumanoidImPassiveObject(HumanoidIm):
    """HumanoidIm + passive articulated object.

    Required cfg key:
        env.articulatedObjectConfigPath: path to the case-local object config.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        if not headless or flags.server_mode or flags.add_proj:
            raise ValueError(
                "PHC-X Studio evaluation requires headless=True, "
                "server_mode=False, and add_proj=False"
            )
        object_config_path = Path(env_cfg["articulatedObjectConfigPath"]).expanduser().resolve()
        if not object_config_path.is_file():
            raise FileNotFoundError(
                f"articulatedObjectConfigPath does not exist: {object_config_path}"
            )
        if env_cfg["phcObjectTrackingMode"] != "passive_dynamic":
            raise ValueError("HumanoidImPassiveObject requires phcObjectTrackingMode=passive_dynamic")

        object_config = json.loads(object_config_path.read_text(encoding="utf-8"))
        self._phc_object_config = object_config
        self._target_name = object_config["object_name"]
        self._target_default_root_pos_np = np.asarray(
            object_config["target_default_root_pos"], dtype=np.float32
        )
        self._target_default_root_quat_np = np.asarray(
            object_config["target_default_root_quat_xyzw"], dtype=np.float32
        )
        self._target_qpos_fps = float(object_config["fps"])
        self._target_joint_names = list(object_config["articulated_target_joint_names"])
        if self._target_default_root_pos_np.shape != (3,):
            raise ValueError(
                "target_default_root_pos must have shape (3,), got "
                f"{self._target_default_root_pos_np.shape}"
            )
        if self._target_default_root_quat_np.shape != (4,):
            raise ValueError(
                "target_default_root_quat_xyzw must have shape (4,), got "
                f"{self._target_default_root_quat_np.shape}"
            )
        if self._target_qpos_fps <= 0.0:
            raise ValueError(f"Object reference fps must be positive, got {self._target_qpos_fps}")
        if not self._target_joint_names:
            raise ValueError("HumanoidImPassiveObject requires at least one articulated joint")
        (
            self._phc_contact_region_points_local_np,
            self._phc_contact_region_point_body_names,
        ) = self._load_contact_region_points(object_config)
        self._phc_contact_region_names = tuple(
            str(name) for name in object_config["contact_region_body_names"]
        )
        if (
            not self._phc_contact_region_names
            or len(set(self._phc_contact_region_names))
            != len(self._phc_contact_region_names)
            or set(self._phc_contact_region_names)
            != set(self._phc_contact_region_point_body_names)
        ):
            raise ValueError(
                "contact_region_body_names must exactly cover the point-link names"
            )

        # This field is a diagnostic runtime control, not case data.
        self._phc_eval_start_frame = (
            int(env_cfg["phcEvalStartFrame"])
            if "phcEvalStartFrame" in env_cfg
            else None
        )

        # Isaac Gym calls the overridden asset/env builders from super().__init__,
        # so their runtime tensors must exist before PHC constructs the simulator.
        self._target_handles = []
        self._target_asset = None
        self._target_dof_properties = None
        self._target_asset_dof_names = []
        self._target_active_dof_indices = None
        self._target_static_box_assets = []
        self._target_static_box_handles = []
        self._target_asset_dof_count = 0
        self._target_asset_body_count = 0
        self._target_max_dof = 0
        self._target_joint_qpos = None
        self._target_initial_joint_qpos = None
        self._target_last_reset_joint_qpos = None
        self._target_last_reset_reference_frame = None
        self._phc_reset_human_root_state = None
        self._phc_reset_human_dof_pos = None
        self._phc_reset_human_body_state = None
        self._phc_reset_object_root_state = None
        self._phc_reset_hand_region_distance = None
        self._target_body_names = []
        self._target_contact_forces = None
        self._left_hand_body_ids = []
        self._right_hand_body_ids = []
        self._phc_contact_labels_10 = None
        self._phc_contact_obj_ref = None
        self._phc_contact_label_names_10 = [
            "left_thumb",
            "left_index",
            "left_middle",
            "left_ring",
            "left_pinky",
            "right_thumb",
            "right_index",
            "right_middle",
            "right_ring",
            "right_pinky",
        ]
        self._phc_contact_label_body_ids = None
        self._phc_contact_region_body_local_ids = None
        self._phc_contact_region_points_local = None
        self._phc_contact_region_point_body_local_ids = None
        self._phc_contact_region_point_indices = None
        self._phc_contact_hand_body_ids = None
        self._phc_contact_label_capsule_endpoints_local = None
        self._phc_contact_label_capsule_radii = None
        self._phc_contact_label_capsule_valid = None
        self._phc_hand_body_ids_flat = None
        self._phc_hand_group_slices = None
        self._phc_hand_capsule_endpoints_local = None
        self._phc_hand_capsule_radii = None
        self._phc_hand_capsule_valid = None
        self._phc_hand_box_centers_local = None
        self._phc_hand_box_quaternions_local = None
        self._phc_hand_box_half_extents = None
        self._phc_hand_box_valid = None
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        self._build_target_tensors()
        self._load_contact_label_capsules()
        self._load_target_joint_qpos()
        self._load_contact_reference()
        self._phc_contact_label_body_ids = self._resolve_contact_label_body_ids()

    def _load_contact_label_capsules(self):
        from pipeline.physics.contact import (
            load_mjcf_body_boxes,
            load_mjcf_body_capsules,
        )

        asset_cfg = self.cfg["robot"]["asset"]
        asset_root = Path(asset_cfg["assetRoot"]).expanduser()
        asset_file = Path(asset_cfg["assetFileName"])
        path = asset_file if asset_file.is_absolute() else asset_root / asset_file
        label_body_names = tuple(
            self._contact_label_to_body_name(label)
            for label in self._phc_contact_label_names_10
        )
        capsule_endpoints, capsule_radii, capsule_valid = load_mjcf_body_capsules(
            path, list(label_body_names)
        )
        if not bool(capsule_valid.all()):
            missing = [
                name for name, valid in zip(label_body_names, capsule_valid.tolist())
                if not bool(valid)
            ]
            raise RuntimeError(
                f"Missing fingertip capsule geoms in case humanoid MJCF {path}: {missing}"
        )
        self._phc_contact_label_capsule_endpoints_local = torch.tensor(
            capsule_endpoints, dtype=torch.float32, device=self.device
        )
        self._phc_contact_label_capsule_radii = torch.tensor(
            capsule_radii, dtype=torch.float32, device=self.device
        )
        self._phc_contact_label_capsule_valid = torch.tensor(
            capsule_valid, dtype=torch.bool, device=self.device
        )

        hand_groups = (tuple(self._left_hand_body_ids), tuple(self._right_hand_body_ids))
        flat_ids = tuple(body_id for group in hand_groups for body_id in group)
        hand_names = [str(self._body_names[body_id]) for body_id in flat_ids]
        hand_capsule_endpoints, hand_capsule_radii, hand_capsule_valid = load_mjcf_body_capsules(path, hand_names)
        (
            hand_box_centers,
            hand_box_quaternions,
            hand_box_half_extents,
            hand_box_valid,
        ) = load_mjcf_body_boxes(path, hand_names)
        if not bool(np.logical_or(hand_capsule_valid, hand_box_valid).all()):
            missing = [
                name
                for name, has_capsule, has_box in zip(
                    hand_names,
                    hand_capsule_valid.tolist(),
                    hand_box_valid.tolist(),
                )
                if not has_capsule and not has_box
            ]
            raise RuntimeError(f"Hand collision bodies have no capsule or box geometry: {missing}")
        self._phc_hand_body_ids_flat = torch.tensor(
            flat_ids, dtype=torch.long, device=self.device
        )
        self._phc_hand_group_slices = (
            slice(0, len(hand_groups[0])),
            slice(len(hand_groups[0]), len(flat_ids)),
        )
        self._phc_hand_capsule_endpoints_local = torch.tensor(
            hand_capsule_endpoints, dtype=torch.float32, device=self.device
        )
        self._phc_hand_capsule_radii = torch.tensor(
            hand_capsule_radii, dtype=torch.float32, device=self.device
        )
        self._phc_hand_capsule_valid = torch.tensor(
            hand_capsule_valid, dtype=torch.bool, device=self.device
        )
        self._phc_hand_box_centers_local = torch.tensor(
            hand_box_centers, dtype=torch.float32, device=self.device
        )
        self._phc_hand_box_quaternions_local = torch.tensor(
            hand_box_quaternions, dtype=torch.float32, device=self.device
        )
        self._phc_hand_box_half_extents = torch.tensor(
            hand_box_half_extents, dtype=torch.float32, device=self.device
        )
        self._phc_hand_box_valid = torch.tensor(
            hand_box_valid, dtype=torch.bool, device=self.device
        )
    @staticmethod
    def _load_contact_region_points(object_config: dict):
        path = Path(object_config["contact_region_points_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"contact_region_points_path does not exist: {path}")
        with np.load(str(path), allow_pickle=False) as payload:
            required = {"points_link_local_scaled", "point_link_names"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"Missing contact-region point fields {missing} in {path}")
            points = np.asarray(payload["points_link_local_scaled"], dtype=np.float32)
            names = [str(value) for value in np.asarray(payload["point_link_names"]).reshape(-1).tolist()]
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"contact-region points must have shape (P,3), got {points.shape} in {path}")
        if len(names) != int(points.shape[0]) or any(not name for name in names):
            raise ValueError(
                f"contact-region point/link mismatch in {path}: "
                f"points={points.shape[0]}, names={len(names)}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"contact-region points contain non-finite values: {path}")
        return points, names

    def _load_contact_reference(self):
        path = Path(self._phc_object_config["contact_reference_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"contact_reference_path does not exist: {path}")
        with np.load(str(path), allow_pickle=False) as payload:
            if set(payload.files) != {"labels"}:
                raise ValueError(f"Contact reference must contain only labels: {path}")
            labels_hand2 = np.asarray(payload["labels"], dtype=np.float32).reshape(-1, 2)
        labels = np.concatenate(
            [
                np.repeat(labels_hand2[:, 0:1], 5, axis=1),
                np.repeat(labels_hand2[:, 1:2], 5, axis=1),
            ],
            axis=1,
        )
        contact_obj = np.any(labels_hand2 > 0.5, axis=1, keepdims=True).astype(np.float32)
        if labels.shape[0] != self._target_joint_qpos.shape[0]:
            raise ValueError(
                "Contact labels and object qpos must have the same frame count: "
                f"labels={labels.shape[0]}, qpos={self._target_joint_qpos.shape[0]}"
            )
        if contact_obj.shape[0] != labels.shape[0]:
            raise ValueError(
                "contact_obj and contact labels must have the same frame count: "
                f"object={contact_obj.shape[0]}, labels={labels.shape[0]}"
            )
        if not np.isfinite(labels).all() or not np.isfinite(contact_obj).all():
            raise ValueError(f"Contact reference contains non-finite values: {path}")
        self._phc_contact_labels_10 = torch.tensor(
            labels, dtype=torch.float32, device=self.device
        )
        self._phc_contact_obj_ref = torch.tensor(
            contact_obj, dtype=torch.float32, device=self.device
        )

    def _create_ground_plane(self):
        from pipeline.physics.articulated_scene import add_ground_plane, load_static_box_assets

        add_ground_plane(self.gym, self.sim, self._phc_object_config)

        self._target_static_box_assets = load_static_box_assets(
            self.gym,
            self.sim,
            self._phc_object_config,
        )

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._target_static_box_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        from pipeline.physics.articulated_scene import validate_humanoid_object_collision_filters

        validate_humanoid_object_collision_filters(
            self.gym,
            env_ptr,
            self.humanoid_handles[env_id],
        )
        self._build_target(env_id, env_ptr)
        from pipeline.physics.articulated_scene import (
            STATIC_SCENE_COLLISION_FILTER,
            create_static_box_actors,
        )

        self._target_static_box_handles.extend(
            create_static_box_actors(
                self.gym,
                env_ptr,
                env_id,
                self._target_static_box_assets,
                self._phc_object_config,
                collision_filter=STATIC_SCENE_COLLISION_FILTER,
            )
        )

    def _load_target_asset(self):
        from pipeline.physics.articulated_scene import load_articulated_asset

        self._target_asset, self._target_dof_properties = load_articulated_asset(
            self.gym,
            self.sim,
            self._phc_object_config,
        )

        self._target_asset_dof_count = int(self.gym.get_asset_dof_count(self._target_asset))
        self._target_asset_dof_names = [str(name) for name in self.gym.get_asset_dof_names(self._target_asset)]
        self._target_asset_body_count = int(self.gym.get_asset_rigid_body_count(self._target_asset))
        self._target_body_names = list(self.gym.get_asset_rigid_body_names(self._target_asset))
        if self._target_asset_dof_count == 0:
            raise RuntimeError("PHC-X Studio articulated object asset has no DOFs")

    def _build_target(self, env_id, env_ptr):
        if self._target_asset is None:
            raise RuntimeError("Target asset must be loaded before building envs")

        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(*[float(v) for v in self._target_default_root_pos_np])
        default_pose.r = gymapi.Quat(*[float(v) for v in self._target_default_root_quat_np])

        from pipeline.physics.articulated_scene import (
            ARTICULATED_OBJECT_COLLISION_FILTER,
            configure_articulated_actor,
        )

        target_handle = self.gym.create_actor(
            env_ptr,
            self._target_asset,
            default_pose,
            self._target_name,
            env_id,
            ARTICULATED_OBJECT_COLLISION_FILTER,
            0,
        )
        configure_articulated_actor(
            self.gym,
            env_ptr,
            target_handle,
            self._target_dof_properties,
            self._phc_object_config,
        )

        # The staged URDF meshes already contain the reconstruction scale.
        self.gym.set_actor_scale(env_ptr, target_handle, 1.0)
        self._target_handles.append(target_handle)

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        root_view = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])
        handles_long = torch.tensor(self._target_handles, dtype=torch.long, device=self.device)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._target_states = root_view[env_ids, handles_long, :]
        self._tar_actor_ids = self._humanoid_actor_ids + handles_long.to(torch.int32)

        self._target_default_root_pos = torch.tensor(
            self._target_default_root_pos_np, dtype=torch.float32, device=self.device
        )
        self._target_default_root_quat = torch.tensor(
            self._target_default_root_quat_np, dtype=torch.float32, device=self.device
        )

        self._target_max_dof = int(self._target_asset_dof_count)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        if dofs_per_env < self.num_dof + self._target_max_dof:
            raise RuntimeError(
                f"Expected at least {self.num_dof + self._target_max_dof} dofs/env, got {dofs_per_env}"
            )
        target_dof_view = self._dof_state.view(self.num_envs, dofs_per_env, 2)[
            ..., self.num_dof : self.num_dof + self._target_max_dof, :
        ]
        self._target_dof_pos = target_dof_view[..., 0]
        self._target_dof_vel = target_dof_view[..., 1]

        bodies_per_env = self._rigid_body_state_reshaped.shape[1]
        required_bodies = self.num_bodies + self._target_asset_body_count
        if bodies_per_env < required_bodies:
            raise RuntimeError(
                f"Expected at least {required_bodies} rigid bodies/env, got {bodies_per_env}"
            )
        contact_force_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        )
        self._target_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., self.num_bodies : required_bodies, :]

        self._left_hand_body_ids = self._hand_body_ids("L_")
        self._right_hand_body_ids = self._hand_body_ids("R_")
        self._phc_contact_region_body_local_ids = self._resolve_contact_region_body_local_ids()
        self._phc_contact_region_points_local = torch.tensor(
            self._phc_contact_region_points_local_np,
            dtype=torch.float32,
            device=self.device,
        )
        self._phc_contact_region_point_body_local_ids = self._resolve_contact_region_point_body_local_ids()
        self._phc_contact_region_point_indices = tuple(
            torch.tensor(
                [
                    index
                    for index, point_name in enumerate(
                        self._phc_contact_region_point_body_names
                    )
                    if point_name == region_name
                ],
                dtype=torch.long,
                device=self.device,
            )
            for region_name in self._phc_contact_region_names
        )
        hand_ids = sorted(set(self._left_hand_body_ids + self._right_hand_body_ids))
        self._phc_contact_hand_body_ids = torch.tensor(hand_ids, dtype=torch.long, device=self.device)
        self._phc_contact_label_body_ids = self._resolve_contact_label_body_ids()
    def _hand_body_ids(self, side_prefix: str):
        names = list(self._body_names)
        suffixes = ("Wrist", "Hand", "Index", "Middle", "Pinky", "Ring", "Thumb")
        ids = [
            idx
            for idx, name in enumerate(names)
            if str(name).startswith(side_prefix) and any(token in str(name) for token in suffixes)
        ]
        return ids

    def _resolve_contact_region_body_local_ids(self):
        names = list(self._target_body_names)
        selected = []
        for requested in self._phc_contact_region_names:
            exact = [
                index for index, body_name in enumerate(names)
                if str(body_name) == requested
            ]
            suffix = [
                index for index, body_name in enumerate(names)
                if str(body_name).endswith(requested)
            ]
            matches = exact or suffix
            if len(matches) != 1:
                raise RuntimeError(
                    f"Contact-region link {requested!r} resolved "
                    f"{len(matches)} Isaac bodies: {names}"
                )
            selected.append(matches[0])
        return torch.tensor(selected, dtype=torch.long, device=self.device)

    def _resolve_contact_region_point_body_local_ids(self):
        point_names = list(self._phc_contact_region_point_body_names)
        if not point_names:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        body_names = [str(name) for name in self._target_body_names]
        resolved = []
        for requested in point_names:
            exact = [idx for idx, name in enumerate(body_names) if name == requested]
            suffix = [idx for idx, name in enumerate(body_names) if name.endswith(requested)]
            matches = exact or suffix
            if len(matches) != 1:
                raise RuntimeError(
                    f"Contact-region link {requested!r} resolved {len(matches)} Isaac bodies; "
                    f"target_body_names={body_names}"
                )
            resolved.append(int(matches[0]))
        return torch.tensor(resolved, dtype=torch.long, device=self.device)

    @staticmethod
    def _contact_label_to_body_name(label_name: str):
        text = label_name.lower()
        if text.startswith("left") or text.startswith("l_"):
            prefix = "L_"
        elif text.startswith("right") or text.startswith("r_"):
            prefix = "R_"
        else:
            return ""
        finger = ""
        for candidate in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            if candidate.lower() in text:
                finger = candidate
                break
        return f"{prefix}{finger}3" if finger else ""

    def _resolve_contact_label_body_ids(self):
        names = list(self._body_names)
        ids = []
        missing = []
        for label_name in self._phc_contact_label_names_10:
            target_name = self._contact_label_to_body_name(label_name)
            body_id = -1
            if target_name:
                for idx, body_name in enumerate(names):
                    text = str(body_name)
                    if text == target_name or text.endswith(target_name):
                        body_id = int(idx)
                        break
            if body_id < 0:
                missing.append((label_name, target_name))
            ids.append(body_id)
        if missing:
            raise RuntimeError(
                f"Contact labels do not resolve to humanoid bodies: {missing}; body_names={names}"
            )
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _load_target_joint_qpos(self):
        qpos_path = Path(
            self._phc_object_config["object_joint_qpos_reference_path"]
        ).expanduser().resolve()
        if not qpos_path.is_file():
            raise FileNotFoundError(
                f"object_joint_qpos_reference_path does not exist: {qpos_path}"
            )
        with np.load(str(qpos_path), allow_pickle=False) as payload:
            qpos_np = np.asarray(payload["joint_qpos"], dtype=np.float32)
            reference_joint_names = [
                str(name) for name in np.asarray(payload["joint_names"]).tolist()
            ]
        if qpos_np.ndim != 2:
            raise ValueError(f"joint_qpos must be (T, K), got {qpos_np.shape} from {qpos_path}")
        if qpos_np.shape[0] == 0:
            raise ValueError(f"joint_qpos must contain at least one frame: {qpos_path}")
        if not np.isfinite(qpos_np).all():
            raise ValueError(f"joint_qpos contains non-finite values: {qpos_path}")
        if qpos_np.shape[1] != len(reference_joint_names):
            raise ValueError(
                f"joint_qpos width {qpos_np.shape[1]} does not match "
                f"reference joint_names {len(reference_joint_names)}"
            )
        if reference_joint_names != list(self._target_joint_names):
            raise ValueError(
                f"object reference joint_names do not match object config: "
                f"reference={reference_joint_names}, config={self._target_joint_names}"
            )
        dof_indices = []
        for name in reference_joint_names:
            if name not in self._target_asset_dof_names:
                raise ValueError(
                    f"object reference joint {name!r} is missing from asset "
                    f"DOFs {self._target_asset_dof_names}"
                )
            dof_indices.append(self._target_asset_dof_names.index(name))
        full_qpos = np.zeros((qpos_np.shape[0], self._target_max_dof), dtype=np.float32)
        full_qpos[:, dof_indices] = qpos_np
        self._target_active_dof_indices = torch.tensor(dof_indices, dtype=torch.long, device=self.device)
        self._target_joint_qpos = torch.tensor(full_qpos, dtype=torch.float32, device=self.device)
        initial_qpos = np.asarray(
            self._phc_object_config["initial_joint_qpos"],
            dtype=np.float32,
        ).reshape(-1)
        if initial_qpos.shape[0] != len(reference_joint_names):
            raise ValueError(
                "initial_joint_qpos length does not match articulated_target_joint_names: "
                f"qpos={initial_qpos.shape[0]}, joints={len(reference_joint_names)}"
            )
        if not np.isfinite(initial_qpos).all():
            raise ValueError("initial_joint_qpos contains non-finite values")
        full_initial_qpos = np.zeros((self._target_max_dof,), dtype=np.float32)
        full_initial_qpos[dof_indices] = initial_qpos
        self._target_initial_joint_qpos = torch.tensor(
            full_initial_qpos,
            dtype=torch.float32,
            device=self.device,
        )

    def _sample_time(self, motion_ids):
        if self._phc_eval_start_frame is not None:
            frame = max(0, self._phc_eval_start_frame)
            return torch.full(
                (motion_ids.shape[0],),
                float(frame) / float(self._target_qpos_fps),
                dtype=torch.float32,
                device=self.device,
            )
        return super()._sample_time(motion_ids)

    def _target_frame_indices(self, env_ids):
        motion_times = (
            self.progress_buf[env_ids].float() * self.dt
            + self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )
        return torch.round(motion_times * float(self._target_qpos_fps)).long()

    def _target_qpos_ref_for_frames(self, frame_indices):
        frames = torch.clamp(frame_indices.long(), 0, self._target_joint_qpos.shape[0] - 1)
        return self._target_joint_qpos[frames]

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self._target_default_root_pos
        self._target_states[env_ids, 3:7] = self._target_default_root_quat
        self._target_states[env_ids, 7:13] = 0.0
        self._write_target_reset_joint_qpos(env_ids)

    def _write_target_reset_joint_qpos(self, env_ids):
        """Use case q0 at frame zero and the absolute reference for later RSI frames."""

        if self._target_initial_joint_qpos is None:
            raise RuntimeError("Missing case-level target initial joint qpos")
        frames = self._target_frame_indices(env_ids)
        reset_qpos = self._target_qpos_ref_for_frames(frames).clone()
        reset_qpos[frames == 0] = self._target_initial_joint_qpos
        self._target_dof_pos[env_ids, :] = reset_qpos
        self._target_dof_vel[env_ids, :] = 0.0
        if self._target_last_reset_joint_qpos is None:
            self._target_last_reset_joint_qpos = torch.zeros_like(self._target_dof_pos)
            self._target_last_reset_reference_frame = torch.zeros(
                (self.num_envs,), dtype=torch.long, device=self.device
            )
        self._target_last_reset_joint_qpos[env_ids] = reset_qpos
        self._target_last_reset_reference_frame[env_ids] = frames

    def _pad_target_dofs(self, tensor):
        target_k = self._target_max_dof
        tensor = tensor.reshape(self.num_envs, -1)
        pad = torch.zeros((self.num_envs, target_k), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1).contiguous()

    def _reset_env_tensors(self, env_ids):
        humanoid_ids = self._humanoid_actor_ids[env_ids]
        target_ids = self._tar_actor_ids[env_ids]
        combined_ids = torch.cat([humanoid_ids, target_ids]).to(torch.int32)

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
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(
                self._dof_state.view(
                    self.num_envs,
                    self._dof_state.shape[0] // self.num_envs,
                    2,
                )[..., 0].contiguous()
            ),
            gymtorch.unwrap_tensor(combined_ids),
            len(combined_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._contact_forces[env_ids] = 0

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (
            not flags.im_eval
            or len(env_ids) == 0
            or self._target_dof_pos is None
        ):
            return

        human_body_state = self._rigid_body_state.view(
            self.num_envs,
            -1,
            13,
        )[:, : self.num_bodies]
        if self._phc_reset_human_root_state is None:
            self._phc_reset_human_root_state = torch.zeros_like(
                self._humanoid_root_states
            )
            self._phc_reset_human_dof_pos = torch.zeros_like(self._dof_pos)
            self._phc_reset_human_body_state = torch.zeros_like(human_body_state)
            self._phc_reset_object_root_state = torch.zeros_like(
                self._target_states
            )
            self._phc_reset_hand_region_distance = torch.zeros(
                (
                    self.num_envs,
                    2,
                    len(self._phc_contact_region_names),
                ),
                dtype=torch.float32,
                device=self.device,
            )
        self._phc_reset_human_root_state[env_ids] = self._humanoid_root_states[
            env_ids
        ]
        self._phc_reset_human_dof_pos[env_ids] = self._dof_pos[env_ids]
        self._phc_reset_human_body_state[env_ids] = human_body_state[env_ids]
        self._phc_reset_object_root_state[env_ids] = self._target_states[env_ids]
        self._phc_reset_hand_region_distance[env_ids] = (
            self._contact_hand_region_distance()[env_ids]
        )

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if self.collect_dataset:
            self.clean_actions = actions.to(self.device).clone()
            if self.add_action_noise:
                noise = torch.normal(
                    mean=0.0,
                    std=float(self.action_noise_std),
                    size=actions.shape,
                    device=self.device,
                )
                self.actions += noise

        if len(self.actions.shape) == 1:
            self.actions = self.actions[None, :]

        if self._pd_control:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                if self.reduce_action:
                    actions_full = torch.zeros([self.actions.shape[0], self._dof_size], device=self.device)
                    actions_full[:, self.action_idx] = self.actions
                    pd_tar = self._action_to_pd_targets(actions_full)
                else:
                    pd_tar = self._action_to_pd_targets(self.actions)
                    if self._freeze_hand:
                        pd_tar[:, self._dof_names.index("L_Hand") * 3 : (self._dof_names.index("L_Hand") * 3 + 3)] = 0
                        pd_tar[:, self._dof_names.index("R_Hand") * 3 : (self._dof_names.index("R_Hand") * 3 + 3)] = 0
                    if self._freeze_toe:
                        pd_tar[:, self._dof_names.index("L_Toe") * 3 : (self._dof_names.index("L_Toe") * 3 + 3)] = 0
                        pd_tar[:, self._dof_names.index("R_Toe") * 3 : (self._dof_names.index("R_Toe") * 3 + 3)] = 0
            elif self.humanoid_type in ["h1", "g1"]:
                pd_tar = self._action_to_pd_targets(self.actions)
            else:
                pd_tar = self._action_to_pd_targets(self.actions)

            self.gym.set_dof_position_target_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self._pad_target_dofs(pd_tar)),
            )
        else:
            if self.control_mode == "force":
                forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
                self.gym.set_dof_actuation_force_tensor(
                    self.sim,
                    gymtorch.unwrap_tensor(self._pad_target_dofs(forces)),
                )
            elif self.control_mode == "pd":
                clip_actions = 10
                self.actions = torch.clip(self.actions, -clip_actions, clip_actions).to(self.device)

        self._update_cycle_count()
        if self._occl_training:
            self._update_occl_training()

    def _target_contact_region_positions(self):
        from pipeline.physics.contact import quat_rotate

        target_body_ids = (
            int(self.num_bodies) + self._phc_contact_region_point_body_local_ids
        )
        body_state = self._rigid_body_state_reshaped[:, target_body_ids, :]
        body_pos = body_state[..., 0:3]
        body_quat = F.normalize(body_state[..., 3:7], dim=-1)
        local = self._phc_contact_region_points_local.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        return body_pos + quat_rotate(body_quat, local)

    def _target_contact_region_force_norm(self):
        local_ids = self._phc_contact_region_body_local_ids
        return torch.linalg.norm(
            self._target_contact_forces[:, local_ids, :],
            dim=-1,
        )

    def _contact_ref_for_frames(self, frame_indices):
        n = int(frame_indices.shape[0])
        frames = torch.clamp(
            frame_indices.long(), 0, self._phc_contact_labels_10.shape[0] - 1
        )
        labels = self._phc_contact_labels_10[frames]
        contact_frames = torch.clamp(
            frame_indices.long(), 0, self._phc_contact_obj_ref.shape[0] - 1
        )
        contact_obj = self._phc_contact_obj_ref[contact_frames].reshape(n, -1).max(dim=-1).values
        return labels, contact_obj

    def _contact_label_region_distance(self):
        from pipeline.physics.contact import capsule_region_surface_distances

        region_pos = self._target_contact_region_positions()
        label_body_ids = self._phc_contact_label_body_ids
        body_state = self._rigid_body_state_reshaped[:, label_body_ids, :]
        return torch.stack(
            [
                capsule_region_surface_distances(
                    body_pos=body_state[..., 0:3],
                    body_quat_xyzw=body_state[..., 3:7],
                    endpoints_local=self._phc_contact_label_capsule_endpoints_local,
                    radii=self._phc_contact_label_capsule_radii,
                    valid=self._phc_contact_label_capsule_valid,
                    region_points=region_pos[:, point_indices],
                )
                for point_indices in self._phc_contact_region_point_indices
            ],
            dim=-1,
        )

    def _contact_label_force_norm(self):
        label_body_ids = self._phc_contact_label_body_ids
        return torch.linalg.norm(self._contact_forces[:, label_body_ids, :], dim=-1)

    def _contact_hand_region_distance(self):
        from pipeline.physics.contact import (
            box_region_surface_distances,
            capsule_region_surface_distances,
        )

        region_pos = self._target_contact_region_positions()
        body_state = self._rigid_body_state_reshaped[:, self._phc_hand_body_ids_flat, :]
        distance_by_region = []
        for point_indices in self._phc_contact_region_point_indices:
            capsule_distance = capsule_region_surface_distances(
                body_pos=body_state[..., 0:3],
                body_quat_xyzw=body_state[..., 3:7],
                endpoints_local=self._phc_hand_capsule_endpoints_local,
                radii=self._phc_hand_capsule_radii,
                valid=self._phc_hand_capsule_valid,
                region_points=region_pos[:, point_indices],
            )
            box_distance = box_region_surface_distances(
                body_pos=body_state[..., 0:3],
                body_quat_xyzw=body_state[..., 3:7],
                centers_local=self._phc_hand_box_centers_local,
                quaternions_local_xyzw=self._phc_hand_box_quaternions_local,
                half_extents=self._phc_hand_box_half_extents,
                valid=self._phc_hand_box_valid,
                region_points=region_pos[:, point_indices],
            )
            body_distance = torch.minimum(capsule_distance, box_distance)
            distance_by_region.append(
                torch.stack(
                    [
                        body_distance[:, group].min(dim=1).values
                        for group in self._phc_hand_group_slices
                    ],
                    dim=1,
                )
            )
        return torch.stack(distance_by_region, dim=-1)

    def _physics_step(self):
        if self.control_mode != "pd":
            super()._physics_step()
            return

        self.render(i=0)
        for _ in range(self.control_freq_inv):
            if not self.paused and self.enable_viewer_sync:
                self.torques = self._compute_torques(self.actions)
                self.gym.set_dof_actuation_force_tensor(
                    self.sim,
                    gymtorch.unwrap_tensor(self._pad_target_dofs(self.torques)),
                )
                self.gym.simulate(self.sim)
                if self.device == "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        super().post_physics_step()

        if flags.im_eval:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            frame_indices = self._target_frame_indices(env_ids)
            qpos_ref = self._target_qpos_ref_for_frames(frame_indices)
            qpos_sim = self._target_dof_pos[:, : qpos_ref.shape[1]]
            active_dof_indices = self._target_active_dof_indices
            if active_dof_indices is not None:
                qpos_sim = qpos_sim[:, active_dof_indices]
                qpos_ref = qpos_ref[:, active_dof_indices]
            if self._target_contact_forces is not None and self._target_contact_forces.shape[1] > 0:
                target_force = torch.linalg.norm(
                    self._target_contact_forces, dim=-1
                ).max(dim=-1).values.detach().cpu().numpy()
                target_force_by_body = torch.linalg.norm(
                    self._target_contact_forces, dim=-1
                ).detach().cpu().numpy()
            else:
                target_force = np.zeros((self.num_envs,), dtype=np.float32)
                target_force_by_body = np.zeros((self.num_envs, 0), dtype=np.float32)

            if self._left_hand_body_ids:
                left_forces = self._contact_forces[:, self._left_hand_body_ids, :]
                left_force = torch.linalg.norm(
                    left_forces, dim=-1
                ).max(dim=-1).values.detach().cpu().numpy()
            else:
                left_force = np.zeros((self.num_envs,), dtype=np.float32)

            if self._right_hand_body_ids:
                right_forces = self._contact_forces[:, self._right_hand_body_ids, :]
                right_force = torch.linalg.norm(
                    right_forces, dim=-1
                ).max(dim=-1).values.detach().cpu().numpy()
            else:
                right_force = np.zeros((self.num_envs,), dtype=np.float32)
            labels, contact_obj = self._contact_ref_for_frames(frame_indices)
            contact = {
                "hand_region_distance": self._contact_hand_region_distance().detach().cpu().numpy(),
                "region_distance": self._contact_label_region_distance().detach().cpu().numpy(),
                "label_force": self._contact_label_force_norm().detach().cpu().numpy(),
                "intended": labels.detach().cpu().numpy(),
                "object_reference": contact_obj.detach().cpu().numpy(),
                "hand_force_n": np.stack((left_force, right_force), axis=-1),
                "region_force_n": self._target_contact_region_force_norm().detach().cpu().numpy(),
                "target_force": target_force,
                "target_force_by_body": target_force_by_body,
                "left_hand_force": left_force,
                "right_hand_force": right_force,
            }
            self.extras["physics_rollout"] = {
                "reset": {
                    "human_root_state": self._phc_reset_human_root_state.detach().cpu().numpy(),
                    "human_dof_pos": self._phc_reset_human_dof_pos.detach().cpu().numpy(),
                    "human_body_state": self._phc_reset_human_body_state.detach().cpu().numpy(),
                    "object_root_state": self._phc_reset_object_root_state.detach().cpu().numpy(),
                    "hand_region_distance": self._phc_reset_hand_region_distance.detach().cpu().numpy(),
                },
                "human": {
                    "root_state": self._humanoid_root_states.detach().cpu().numpy(),
                    "dof_pos": self._dof_pos.detach().cpu().numpy(),
                    "body_state": self._rigid_body_state.view(
                        self.num_envs, -1, 13
                    )[:, : self.num_bodies].detach().cpu().numpy(),
                },
                "object": {
                    "root_state": self._target_states.detach().cpu().numpy(),
                    "qpos": qpos_sim.detach().cpu().numpy(),
                    "qpos_reference": qpos_ref.detach().cpu().numpy(),
                    "frame": frame_indices.detach().cpu().numpy(),
                    # Preserve the simulator-written reset state so the final
                    # method-neutral rollout can prove q0/RSI semantics.
                    "reset_joint_qpos": self._target_last_reset_joint_qpos[
                        :, active_dof_indices
                    ].detach().cpu().numpy(),
                    "reset_reference_frame": self._target_last_reset_reference_frame.detach().cpu().numpy(),
                },
                "contact": contact,
                "policy": {
                    "terminate": self._terminate_buf.detach().cpu().numpy(),
                },
            }
