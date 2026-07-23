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
from pipeline.physics.mimic.contact_pairs import (
    aggregate_rigid_contact_body_matrix,
    aggregate_rigid_contact_pairs,
    summarize_handle_contact_pairs,
)
from pipeline.physics.mimic.collision_distance import (
    box_region_surface_distances,
    capsule_region_surface_distances,
    load_mjcf_body_boxes,
    load_mjcf_body_capsules,
)
from pipeline.physics.mimic.contact_region import quat_rotate_xyzw


_DEFAULT_OBJECT_DENSITY = 150.0
_DEFAULT_RIGID_SHAPE_PROPERTIES = {
    "restitution": 0.05,
    "friction": 0.6,
    "rolling_friction": 0.01,
    "torsion_friction": 0.01,
    "rest_offset": 0.002,
}

# Shared only by the passive object and its synthetic ground.  PHC's SMPL-X
# self-collision filters use low bits, so this bit suppresses object-ground
# contact without suppressing any humanoid contact.
_OBJECT_GROUND_FILTER = 1 << 15


class HumanoidImPassiveObject(HumanoidIm):
    """HumanoidIm + passive articulated object.

    Required cfg key:
        env.articulatedObjectConfigPath: path to the case-local object config.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        object_config_path = Path(env_cfg["articulatedObjectConfigPath"]).expanduser().resolve()
        if not object_config_path.is_file():
            raise FileNotFoundError(
                f"articulatedObjectConfigPath does not exist: {object_config_path}"
            )
        if env_cfg["phcObjectTrackingMode"] != "passive_dynamic":
            raise ValueError("HumanoidImPassiveObject requires phcObjectTrackingMode=passive_dynamic")

        object_config = json.loads(object_config_path.read_text(encoding="utf-8"))
        asset_info = object_config["asset_info"]
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
        self._target_physical_props = asset_info["physical_props"]
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

        # These fields are diagnostic runtime controls, not case data. Training
        # leaves them absent; evaluation writes both explicitly.
        self._phc_eval_start_frame = (
            int(env_cfg["phcEvalStartFrame"])
            if "phcEvalStartFrame" in env_cfg
            else None
        )
        self._phc_exact_contact_telemetry = (
            bool(env_cfg["phcExactContactTelemetry"])
            if "phcExactContactTelemetry" in env_cfg
            else False
        )

        # Isaac Gym calls the overridden asset/env builders from super().__init__,
        # so their runtime tensors must exist before PHC constructs the simulator.
        self._target_handles = []
        self._target_asset = None
        self._target_asset_dof_names = []
        self._target_active_dof_indices = None
        self._target_filtered_ground_asset = None
        self._target_filtered_ground_handles = []
        self._target_asset_dof_count = 0
        self._target_asset_body_count = 0
        self._target_max_dof = 0
        self._target_joint_qpos = None
        self._target_initial_joint_qpos = None
        self._target_body_names = []
        self._target_contact_forces = None
        self._left_hand_body_ids = []
        self._right_hand_body_ids = []
        self._phc_contact_labels_10 = None
        self._phc_contact_granularity = "none"
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
        self._phc_exact_contact_humanoid_env_body_indices = None
        self._phc_exact_contact_all_humanoid_env_body_indices = None
        self._phc_exact_contact_target_env_body_indices = None
        self._phc_exact_env_body_names = []
        # Isaac Gym exact rigid-contact pair reads are CPU-only telemetry.
        if self._phc_exact_contact_telemetry and bool(sim_params.use_gpu_pipeline):
            raise RuntimeError(
                "phcExactContactTelemetry requires use_gpu_pipeline=False"
            )

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        self._build_target_tensors()
        self._load_contact_label_capsules()
        self._load_target_joint_qpos()
        self._load_contact_reference()
        self._phc_contact_label_body_ids = self._resolve_contact_label_body_ids()

    def _load_contact_label_capsules(self):
        asset_cfg = self.cfg["robot"]["asset"]
        asset_root = Path(asset_cfg["assetRoot"]).expanduser()
        asset_file = Path(asset_cfg["assetFileName"])
        path = asset_file if asset_file.is_absolute() else asset_root / asset_file
        label_body_names = tuple(
            self._contact_label_to_body_name(label)
            for label in self._phc_contact_label_names_10
        )
        capsules = load_mjcf_body_capsules(path, label_body_names)
        if not bool(capsules.valid.all()):
            missing = [
                name for name, valid in zip(capsules.body_names, capsules.valid.tolist())
                if not bool(valid)
            ]
            raise RuntimeError(
                f"Missing fingertip capsule geoms in case humanoid MJCF {path}: {missing}"
            )
        self._phc_contact_label_capsule_endpoints_local = torch.tensor(
            capsules.endpoints_local, dtype=torch.float32, device=self.device
        )
        self._phc_contact_label_capsule_radii = torch.tensor(
            capsules.radii, dtype=torch.float32, device=self.device
        )
        self._phc_contact_label_capsule_valid = torch.tensor(
            capsules.valid, dtype=torch.bool, device=self.device
        )

        hand_groups = (tuple(self._left_hand_body_ids), tuple(self._right_hand_body_ids))
        flat_ids = tuple(body_id for group in hand_groups for body_id in group)
        hand_names = [str(self._body_names[body_id]) for body_id in flat_ids]
        hand_capsules = load_mjcf_body_capsules(path, hand_names)
        hand_boxes = load_mjcf_body_boxes(path, hand_names)
        if not bool(np.logical_or(hand_capsules.valid, hand_boxes.valid).all()):
            missing = [
                name
                for name, capsule_valid, box_valid in zip(
                    hand_names,
                    hand_capsules.valid.tolist(),
                    hand_boxes.valid.tolist(),
                )
                if not capsule_valid and not box_valid
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
            hand_capsules.endpoints_local, dtype=torch.float32, device=self.device
        )
        self._phc_hand_capsule_radii = torch.tensor(
            hand_capsules.radii, dtype=torch.float32, device=self.device
        )
        self._phc_hand_capsule_valid = torch.tensor(
            hand_capsules.valid, dtype=torch.bool, device=self.device
        )
        self._phc_hand_box_centers_local = torch.tensor(
            hand_boxes.centers_local, dtype=torch.float32, device=self.device
        )
        self._phc_hand_box_quaternions_local = torch.tensor(
            hand_boxes.quaternions_local_xyzw, dtype=torch.float32, device=self.device
        )
        self._phc_hand_box_half_extents = torch.tensor(
            hand_boxes.half_extents, dtype=torch.float32, device=self.device
        )
        self._phc_hand_box_valid = torch.tensor(
            hand_boxes.valid, dtype=torch.bool, device=self.device
        )
    @staticmethod
    def _load_contact_region_points(object_config: dict):
        path = Path(object_config["contact_region_points_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"contact_region_points_path does not exist: {path}")
        with np.load(str(path), allow_pickle=False) as payload:
            required = {"points_link_local_scaled", "point_link_names", "coordinate_space"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"Missing contact-region point fields {missing} in {path}")
            points = np.asarray(payload["points_link_local_scaled"], dtype=np.float32)
            names = [str(value) for value in np.asarray(payload["point_link_names"]).reshape(-1).tolist()]
            coordinate_space = str(np.asarray(payload["coordinate_space"]).reshape(()))
        if coordinate_space != "scaled_source_urdf_link_local":
            raise ValueError(f"Unsupported contact-region coordinate space {coordinate_space!r} in {path}")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"contact-region points must have shape (P,3), got {points.shape} in {path}")
        if len(names) != int(points.shape[0]) or any(not name for name in names):
            raise ValueError(f"contact-region point/link mismatch in {path}: points={points.shape[0]}, names={len(names)}")
        if not np.isfinite(points).all():
            raise ValueError(f"contact-region points contain non-finite values: {path}")
        return points, names

    def _load_contact_reference(self):
        path = Path(self._phc_object_config["contact_reference_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"contact_reference_path does not exist: {path}")
        with np.load(str(path), allow_pickle=False) as payload:
            granularity = str(np.asarray(payload["contact_granularity"]).reshape(()))
            if granularity == "fingertip10":
                labels = np.asarray(payload["contact_labels_10"], dtype=np.float32).reshape(-1, 10)
            elif granularity == "hand2":
                labels_hand2 = np.asarray(payload["contact_labels"], dtype=np.float32).reshape(-1, 2)
                labels = np.concatenate(
                    [
                        np.repeat(labels_hand2[:, 0:1], 5, axis=1),
                        np.repeat(labels_hand2[:, 1:2], 5, axis=1),
                    ],
                    axis=1,
                )
                granularity = "hand2_expanded_to_fingertips"
            else:
                raise ValueError(
                    f"Unsupported contact granularity {granularity!r} in {path}; "
                    "expected fingertip10 or hand2"
                )
            contact_obj = np.asarray(payload["contact_obj"], dtype=np.float32).reshape(-1, 1)
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
        self._phc_contact_granularity = granularity

    def _create_ground_plane(self):
        # Isaac Gym's global ground plane has no per-actor collision filter.
        # For passive articulated objects, object-ground contact is not
        # meaningful because the object root is fixed and gravity is disabled;
        # it only lets thick COACD colliders push passive joints open.  Use a
        # filterable static box as the ground instead: humanoid-ground remains
        # active, while object-ground is filtered by a dedicated high bit.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.use_mesh_materials = True

        ground_size = 200.0
        ground_thickness = 0.05
        self._target_filtered_ground_asset = self.gym.create_box(
            self.sim,
            ground_size,
            ground_size,
            ground_thickness,
            asset_options,
        )

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._target_filtered_ground_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        humanoid_props = self.gym.get_actor_rigid_shape_properties(
            env_ptr, self.humanoid_handles[env_id]
        )
        if any(int(prop.filter) & _OBJECT_GROUND_FILTER for prop in humanoid_props):
            raise RuntimeError("Object-ground collision bit overlaps a humanoid shape filter")
        self._build_target(env_id, env_ptr)
        self._build_filtered_ground(env_id, env_ptr)

    def _build_filtered_ground(self, env_id, env_ptr):
        if self._target_filtered_ground_asset is None:
            raise RuntimeError("Filtered ground asset must be created before building envs")

        ground_thickness = 0.05
        ground_height = float(self.cfg["env"]["plane"]["height"])
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, ground_height - 0.5 * ground_thickness)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        ground_handle = self.gym.create_actor(
            env_ptr,
            self._target_filtered_ground_asset,
            pose,
            "filtered_ground",
            env_id,
            _OBJECT_GROUND_FILTER,
            0,
        )
        self._target_filtered_ground_handles.append(ground_handle)

    def _load_target_asset(self):
        object_config = self._phc_object_config
        asset_info = object_config["asset_info"]
        asset_root = Path(object_config["asset_root"]).expanduser().resolve()
        urdf_path = Path(object_config["urdf_path"]).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Object URDF does not exist: {urdf_path}")
        try:
            asset_file = str(urdf_path.relative_to(asset_root))
        except ValueError as exc:
            raise ValueError(
                f"Object URDF {urdf_path} must be inside asset_root {asset_root}"
            ) from exc

        density = (
            float(self._target_physical_props["density"])
            if "density" in self._target_physical_props
            else _DEFAULT_OBJECT_DENSITY
        )
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.density = density
        asset_options.override_com = False
        asset_options.override_inertia = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = bool(asset_info["isaac_vhacd_enabled"])
        asset_options.convex_decomposition_from_submeshes = bool(
            asset_info["isaac_convex_decomposition_from_submeshes"]
        )
        asset_options.use_mesh_materials = True
        if asset_options.vhacd_enabled:
            asset_options.vhacd_params.max_convex_hulls = 64
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000
        asset_options.flip_visual_attachments = False

        self._target_asset = self.gym.load_asset(self.sim, str(asset_root), asset_file, asset_options)
        target_asset_shape_props = self.gym.get_asset_rigid_shape_properties(self._target_asset)
        for prop in target_asset_shape_props:
            prop.filter = _OBJECT_GROUND_FILTER
        self.gym.set_asset_rigid_shape_properties(self._target_asset, target_asset_shape_props)

        self._target_asset_dof_count = int(self.gym.get_asset_dof_count(self._target_asset))
        self._target_asset_dof_names = [str(name) for name in self.gym.get_asset_dof_names(self._target_asset)]
        self._target_asset_body_count = int(self.gym.get_asset_rigid_body_count(self._target_asset))
        self._target_body_names = list(self.gym.get_asset_rigid_body_names(self._target_asset))
        if self._target_asset_dof_count == 0:
            raise RuntimeError(f"Articulated object asset has no DOFs: {urdf_path}")

    def _build_target(self, env_id, env_ptr):
        if self._target_asset is None:
            raise RuntimeError("Target asset must be loaded before building envs")

        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(*[float(v) for v in self._target_default_root_pos_np])
        default_pose.r = gymapi.Quat(*[float(v) for v in self._target_default_root_quat_np])

        col_group = env_id
        target_handle = self.gym.create_actor(
            env_ptr,
            self._target_asset,
            default_pose,
            self._target_name,
            col_group,
            _OBJECT_GROUND_FILTER,
            0,
        )

        shape_cfg = dict(_DEFAULT_RIGID_SHAPE_PROPERTIES)
        if "rigid_shape" in self._target_physical_props:
            shape_cfg.update(self._target_physical_props["rigid_shape"])
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for prop in props:
            # Use a dedicated object-only shape filter bit.  Object shapes share
            # this bit, so PhysX ignores object internal self-collision.  The
            # humanoid shapes keep their existing filters, so humanoid-object
            # collision is preserved as long as the humanoid does not explicitly
            # use this dedicated bit.
            prop.filter = _OBJECT_GROUND_FILTER
            prop.restitution = float(shape_cfg["restitution"])
            prop.friction = float(shape_cfg["friction"])
            prop.rolling_friction = float(shape_cfg["rolling_friction"])
            prop.torsion_friction = float(shape_cfg["torsion_friction"])
            prop.rest_offset = float(shape_cfg["rest_offset"])
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)

        dof_props = self.gym.get_actor_dof_properties(env_ptr, target_handle)
        if len(dof_props) > 0:
            dof_props["driveMode"][:] = gymapi.DOF_MODE_NONE
            if "stiffness" in dof_props.dtype.names:
                dof_props["stiffness"][:] = 0.0
            if "damping" in dof_props.dtype.names:
                dof_props["damping"][:] = float(self.cfg["env"]["phcObjectJointDamping"])
            if "friction" in dof_props.dtype.names:
                dof_props["friction"][:] = float(self.cfg["env"]["phcObjectJointFriction"])
            self.gym.set_actor_dof_properties(env_ptr, target_handle, dof_props)

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
        hand_ids = sorted(set(self._left_hand_body_ids + self._right_hand_body_ids))
        self._phc_contact_hand_body_ids = torch.tensor(hand_ids, dtype=torch.long, device=self.device)
        self._phc_contact_label_body_ids = self._resolve_contact_label_body_ids()
        if self._phc_exact_contact_telemetry:
            self._build_exact_contact_index_maps()

    def _build_exact_contact_index_maps(self):
        """Cache environment-domain rigid-body indices for exact pair reads."""

        label_local = [
            int(value)
            for value in self._phc_contact_label_body_ids.detach().cpu().tolist()
        ]

        humanoid_indices = []
        all_humanoid_indices = []
        target_indices = []
        for env_idx, env_ptr in enumerate(self.envs):
            humanoid_handle = self.humanoid_handles[env_idx]
            target_handle = self._target_handles[env_idx]
            humanoid_row = []
            for body_local in label_local:
                if body_local < 0:
                    humanoid_row.append(-1)
                else:
                    humanoid_row.append(
                        int(
                            self.gym.get_actor_rigid_body_index(
                                env_ptr,
                                humanoid_handle,
                                int(body_local),
                                gymapi.DOMAIN_ENV,
                            )
                        )
                    )
            target_row = [
                int(
                    self.gym.get_actor_rigid_body_index(
                        env_ptr,
                        target_handle,
                        int(body_local),
                        gymapi.DOMAIN_ENV,
                    )
                )
                for body_local in range(int(self._target_asset_body_count))
            ]
            all_humanoid_row = [
                int(
                    self.gym.get_actor_rigid_body_index(
                        env_ptr,
                        humanoid_handle,
                        int(body_local),
                        gymapi.DOMAIN_ENV,
                    )
                )
                for body_local in range(int(self.num_bodies))
            ]
            humanoid_indices.append(humanoid_row)
            all_humanoid_indices.append(all_humanoid_row)
            target_indices.append(target_row)

        self._phc_exact_contact_humanoid_env_body_indices = np.asarray(
            humanoid_indices, dtype=np.int64
        )
        self._phc_exact_contact_target_env_body_indices = np.asarray(
            target_indices, dtype=np.int64
        )
        self._phc_exact_contact_all_humanoid_env_body_indices = np.asarray(
            all_humanoid_indices, dtype=np.int64
        )
        bodies_per_env = int(self._rigid_body_state_reshaped.shape[1])
        env_body_names = [f"unknown:{idx}" for idx in range(bodies_per_env)]
        if self.envs:
            env_ptr = self.envs[0]
            actor_specs = [
                ("humanoid", self.humanoid_handles[0]),
                ("object", self._target_handles[0]),
                ("filtered_ground", self._target_filtered_ground_handles[0]),
            ]
            for actor_prefix, actor_handle in actor_specs:
                actor_names = self.gym.get_actor_rigid_body_names(env_ptr, actor_handle)
                for actor_local, body_name in enumerate(actor_names):
                    env_body_idx = int(
                        self.gym.get_actor_rigid_body_index(
                            env_ptr,
                            actor_handle,
                            int(actor_local),
                            gymapi.DOMAIN_ENV,
                        )
                    )
                    if 0 <= env_body_idx < bodies_per_env:
                        env_body_names[env_body_idx] = f"{actor_prefix}:{body_name}"
        self._phc_exact_env_body_names = env_body_names

    def _compute_exact_contact_pair_telemetry(self, frame_indices):
        """Return true fingertip-object rigid-body collision-pair statistics."""

        if not self._phc_exact_contact_telemetry:
            return None
        humanoid_indices = self._phc_exact_contact_humanoid_env_body_indices
        target_indices = self._phc_exact_contact_target_env_body_indices
        all_humanoid_indices = self._phc_exact_contact_all_humanoid_env_body_indices
        num_labels = int(humanoid_indices.shape[1])
        num_target_bodies = int(target_indices.shape[1])
        pair_count = np.zeros(
            (self.num_envs, num_labels, num_target_bodies), dtype=np.int32
        )
        all_pair_count = np.zeros(
            (self.num_envs, int(self.num_bodies), num_target_bodies), dtype=np.int32
        )
        bodies_per_env = int(self._rigid_body_state_reshaped.shape[1])
        env_pair_count = np.zeros(
            (self.num_envs, bodies_per_env, bodies_per_env), dtype=np.int32
        )
        for env_idx, env_ptr in enumerate(self.envs):
            contacts = self.gym.get_env_rigid_contacts(env_ptr)
            env_aggregation = aggregate_rigid_contact_body_matrix(
                contacts, num_bodies=bodies_per_env
            )
            env_pair_count[env_idx] = env_aggregation.count
            aggregation = aggregate_rigid_contact_pairs(
                contacts,
                humanoid_body_indices=humanoid_indices[env_idx],
                target_body_indices=target_indices[env_idx],
            )
            pair_count[env_idx] = aggregation.count
            all_aggregation = aggregate_rigid_contact_pairs(
                contacts,
                humanoid_body_indices=all_humanoid_indices[env_idx],
                target_body_indices=target_indices[env_idx],
            )
            all_pair_count[env_idx] = all_aggregation.count

        active_label, _ = self._contact_ref_for_frames(frame_indices)
        active_mask = active_label.detach().cpu().numpy()[:, :num_labels] > 0.0
        region_body_ids = self._phc_contact_region_body_local_ids
        if region_body_ids is None:
            region_ids = []
        else:
            region_ids = [int(value) for value in region_body_ids.detach().cpu().tolist()]
        summary = summarize_handle_contact_pairs(
            pair_count,
            active_label_mask=active_mask,
            target_region_body_indices=region_ids,
        )
        return {
            "exact_contact_pair_count": pair_count.astype(np.float32),
            "exact_humanoid_target_pair_count": all_pair_count.astype(np.float32),
            "exact_env_body_pair_count": env_pair_count.astype(np.float32),
            "exact_any_humanoid_target_contact_count": all_pair_count.sum(
                axis=(1, 2), dtype=np.int32
            ).astype(np.float32),
            "exact_handle_contact_count": summary.any_label_count.astype(np.float32),
            "exact_active_handle_contact_count": summary.active_label_count.astype(np.float32),
        }

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
        # The exported point/link assignments are authoritative. Using broader
        # source link lists can select a stationary second door in multi-door
        # objects and incorrectly count it as the manipulated region.
        requested_names = sorted(set(self._phc_contact_region_point_body_names))
        selected = []
        for requested in requested_names:
            for idx, body_name in enumerate(names):
                text = str(body_name)
                if text == requested or text.endswith(requested):
                    selected.append(idx)
        if not selected:
            raise RuntimeError(
                "No contact-region point link resolved to an Isaac object body: "
                f"point_links={requested_names}, target_body_names={names}"
            )
        selected = sorted(set(int(idx) for idx in selected))
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
            self._phc_object_config["obj_joint_qpos_sidecar_path"]
        ).expanduser().resolve()
        if not qpos_path.is_file():
            raise FileNotFoundError(f"obj_joint_qpos_sidecar_path does not exist: {qpos_path}")
        with np.load(str(qpos_path), allow_pickle=False) as payload:
            qpos_np = np.asarray(payload["joint_qpos"], dtype=np.float32)
            sidecar_joint_names = [
                str(name) for name in np.asarray(payload["joint_names"]).tolist()
            ]
        if qpos_np.ndim != 2:
            raise ValueError(f"joint_qpos must be (T, K), got {qpos_np.shape} from {qpos_path}")
        if qpos_np.shape[0] == 0:
            raise ValueError(f"joint_qpos must contain at least one frame: {qpos_path}")
        if not np.isfinite(qpos_np).all():
            raise ValueError(f"joint_qpos contains non-finite values: {qpos_path}")
        if qpos_np.shape[1] != len(sidecar_joint_names):
            raise ValueError(
                f"joint_qpos width {qpos_np.shape[1]} does not match sidecar joint_names {len(sidecar_joint_names)}"
            )
        if sidecar_joint_names != list(self._target_joint_names):
            raise ValueError(
                f"object sidecar joint_names do not match object config: "
                f"sidecar={sidecar_joint_names}, config={self._target_joint_names}"
            )
        dof_indices = []
        for name in sidecar_joint_names:
            if name not in self._target_asset_dof_names:
                raise ValueError(
                    f"object sidecar joint {name!r} is missing from asset DOFs {self._target_asset_dof_names}"
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
        if initial_qpos.shape[0] != len(sidecar_joint_names):
            raise ValueError(
                "initial_joint_qpos length does not match articulated_target_joint_names: "
                f"qpos={initial_qpos.shape[0]}, joints={len(sidecar_joint_names)}"
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
        target_body_ids = (
            int(self.num_bodies) + self._phc_contact_region_point_body_local_ids
        )
        body_state = self._rigid_body_state_reshaped[:, target_body_ids, :]
        body_pos = body_state[..., 0:3]
        body_quat = F.normalize(body_state[..., 3:7], dim=-1)
        local = self._phc_contact_region_points_local.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )
        return body_pos + quat_rotate_xyzw(body_quat, local)

    def _target_contact_region_force_norm(self):
        local_ids = self._phc_contact_region_body_local_ids
        return torch.linalg.norm(self._target_contact_forces[:, local_ids, :], dim=-1).max(dim=-1).values

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
        region_pos = self._target_contact_region_positions()
        label_body_ids = self._phc_contact_label_body_ids
        body_state = self._rigid_body_state_reshaped[:, label_body_ids, :]
        return capsule_region_surface_distances(
            body_pos=body_state[..., 0:3],
            body_quat_xyzw=body_state[..., 3:7],
            endpoints_local=self._phc_contact_label_capsule_endpoints_local,
            radii=self._phc_contact_label_capsule_radii,
            valid=self._phc_contact_label_capsule_valid,
            region_points=region_pos,
        )

    def _contact_label_force_norm(self):
        label_body_ids = self._phc_contact_label_body_ids
        return torch.linalg.norm(self._contact_forces[:, label_body_ids, :], dim=-1)

    def _contact_hand_region_distance(self):
        region_pos = self._target_contact_region_positions()
        body_state = self._rigid_body_state_reshaped[:, self._phc_hand_body_ids_flat, :]
        capsule_distance = capsule_region_surface_distances(
            body_pos=body_state[..., 0:3],
            body_quat_xyzw=body_state[..., 3:7],
            endpoints_local=self._phc_hand_capsule_endpoints_local,
            radii=self._phc_hand_capsule_radii,
            valid=self._phc_hand_capsule_valid,
            region_points=region_pos,
        )
        box_distance = box_region_surface_distances(
            body_pos=body_state[..., 0:3],
            body_quat_xyzw=body_state[..., 3:7],
            centers_local=self._phc_hand_box_centers_local,
            quaternions_local_xyzw=self._phc_hand_box_quaternions_local,
            half_extents=self._phc_hand_box_half_extents,
            valid=self._phc_hand_box_valid,
            region_points=region_pos,
        )
        body_distance = torch.minimum(capsule_distance, box_distance)
        return torch.stack(
            [
                body_distance[:, group].min(dim=1).values
                for group in self._phc_hand_group_slices
            ],
            dim=1,
        )

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
                "target_force": target_force,
                "target_force_by_body": target_force_by_body,
                "left_hand_force": left_force,
                "right_hand_force": right_force,
            }
            exact_telemetry = self._compute_exact_contact_pair_telemetry(frame_indices)
            if exact_telemetry is not None:
                contact["exact"] = exact_telemetry
            self.extras["physics_rollout"] = {
                "object": {
                    "qpos": qpos_sim.detach().cpu().numpy(),
                    "qpos_reference": qpos_ref.detach().cpu().numpy(),
                    "frame": frame_indices.detach().cpu().numpy(),
                },
                "contact": contact,
                "policy": {
                    "terminate": self._terminate_buf.detach().cpu().numpy(),
                },
            }
