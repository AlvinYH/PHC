"""Frozen PHC-X plus a trainable residual policy for articulated HOI."""

from __future__ import annotations

import json
from pathlib import Path

from isaacgym import gymtorch

import torch
import torch.nn.functional as F

from phc.env.tasks.humanoid_im_passive_object import HumanoidImPassiveObject
from phc.utils.flags import flags


class _FrozenPhcxPolicy:
    """The selected PHC-X PNN actor without its unused trainer state."""

    def __init__(self, checkpoint_path: str, device: str, actor_index: int):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        running = checkpoint["running_mean_std"]
        self.mean = running["running_mean"].float().to(device)
        self.var = running["running_var"].float().to(device)
        self.obs_size = int(self.mean.shape[0])

        model = checkpoint["model"]
        prefix = f"a2c_network.pnn.actors.{actor_index}."
        layer_ids = (0, 2, 4, 6, 8, 10)
        self.weights = [
            model[f"{prefix}{layer}.weight"].float().to(device)
            for layer in layer_ids
        ]
        self.biases = [
            model[f"{prefix}{layer}.bias"].float().to(device)
            for layer in layer_ids
        ]
        self.mu_weight = model[f"{prefix}12.weight"].float().to(device)
        self.mu_bias = model[f"{prefix}12.bias"].float().to(device)

    @torch.no_grad()
    def action(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_size:
            raise RuntimeError(
                f"PHC-X observation size is {obs.shape[-1]}, expected {self.obs_size}"
            )
        value = (obs.float() - self.mean) / torch.sqrt(self.var + 1.0e-5)
        value = value.clamp(-5.0, 5.0)
        for weight, bias in zip(self.weights, self.biases):
            value = F.silu(value @ weight.T + bias)
        return (value @ self.mu_weight.T + self.mu_bias).clamp(-1.0, 1.0)


class HumanoidImOursResidual(HumanoidImPassiveObject):
    """Our residual action, observation, reward, and PPO training task."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        path = Path(cfg["env"]["phcOursConfigPath"]).expanduser().resolve()
        ours = json.loads(path.read_text(encoding="utf-8"))
        action = ours["action"]
        reward = ours["reward"]

        self._ours_base_checkpoint = ours["base_checkpoint"]
        self._ours_training_prim = int(ours["training_prim"])
        self._ours_action_mode = action["mode"]
        self._ours_restricted_start_body = action["restricted_start_body"]
        self._ours_residual_scale = float(action["scale"])
        self._ours_reward = reward

        self._ours_base_policy = None
        self._ours_action_ids = None
        self._ours_last_base_action = None
        self._ours_last_residual_action = None
        self._ours_previous_residual_action = None
        self._ours_last_residual_pd = None
        self._ours_history_valid = None
        self._ours_previous_valid = None
        self.ours_reward_terms = None

        super().__init__(
            cfg,
            sim_params,
            physics_engine,
            device_type,
            device_id,
            headless,
        )

        self._ours_base_policy = _FrozenPhcxPolicy(
            self._ours_base_checkpoint,
            self.device,
            self._ours_training_prim,
        )
        self._ours_action_ids = self._select_action_ids()
        native_obs_size = self._native_obs_size()
        if self._ours_base_policy.obs_size != native_obs_size:
            raise RuntimeError(
                "PHC-X observation mismatch: "
                f"checkpoint={self._ours_base_policy.obs_size}, task={native_obs_size}"
            )
        self._cache_base_action(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )
        self._ours_history_valid = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._ours_previous_valid = self._ours_history_valid.clone()

    def _ours_extra_obs_size(self) -> int:
        # q/q_ref/q_error/qvel + object root + contact + frozen base action.
        return 4 * len(self._target_joint_names) + 196

    def _native_obs_size(self) -> int:
        return int(self.obs_buf.shape[1]) - self._ours_extra_obs_size()

    def get_task_obs_size(self):
        return super().get_task_obs_size() + self._ours_extra_obs_size()

    def _select_action_ids(self) -> torch.Tensor:
        if self._ours_action_mode == "full_body":
            start = 0
        elif self._ours_action_mode == "restricted":
            if self._ours_restricted_start_body not in self._dof_names:
                raise ValueError(
                    "Unknown restricted residual start body "
                    f"{self._ours_restricted_start_body!r}"
                )
            start = self._dof_names.index(self._ours_restricted_start_body) * 3
        else:
            raise ValueError(
                "ours.action.mode must be restricted or full_body, got "
                f"{self._ours_action_mode!r}"
            )
        return torch.arange(
            start,
            self._dof_size,
            dtype=torch.long,
            device=self.device,
        )

    def _cache_base_action(self, env_ids):
        native_obs_size = self._native_obs_size()
        native = self.obs_buf[env_ids, :native_obs_size]
        base_action = self._ours_base_policy.action(native.detach())
        if self._ours_last_base_action is None:
            self._ours_last_base_action = torch.zeros(
                (self.num_envs, self.num_actions),
                dtype=base_action.dtype,
                device=self.device,
            )
        self._ours_last_base_action[env_ids] = base_action
        self.obs_buf[env_ids, -self.num_actions :] = base_action

    def pre_physics_step(self, actions):
        residual = actions.to(self.device)
        if residual.ndim == 1:
            residual = residual[None]
        residual = residual.clamp(-1.0, 1.0)
        if self._ours_base_policy is None or self._ours_last_base_action is None:
            raise RuntimeError("Frozen PHC-X was not initialized before the first action")

        base_action = self._ours_last_base_action
        base_pd = self._action_to_pd_targets(base_action)
        residual_pd = torch.zeros_like(base_pd)
        selected = self._ours_action_ids
        residual_pd[:, selected] = (
            self._pd_action_scale[selected].unsqueeze(0)
            * residual[:, selected]
            * self._ours_residual_scale
        )

        self.actions = residual
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self._pad_target_dofs(base_pd + residual_pd)),
        )
        self._ours_last_base_action = base_action.detach()
        self._ours_previous_valid = self._ours_history_valid.clone()
        self._ours_previous_residual_action = self._ours_last_residual_action
        self._ours_last_residual_action = residual.detach()
        self._ours_last_residual_pd = residual_pd.detach()
        self._ours_history_valid[:] = True
        self._update_cycle_count()
        if self._occl_training:
            self._update_occl_training()

    def _active_reference(self, frames, qpos_ref):
        if qpos_ref.shape[1] == 0:
            return torch.ones(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        previous = self._target_qpos_ref_for_frames(torch.clamp(frames - 1, min=0))
        qvel = torch.abs(qpos_ref - previous) * self._target_qpos_fps
        progress = torch.abs(
            qpos_ref - self._target_initial_joint_qpos.unsqueeze(0)
        )
        reward = self._ours_reward
        return (
            qvel.max(dim=-1).values > float(reward["active_qvel_gate"])
        ) | (
            progress.max(dim=-1).values > float(reward["active_qpos_gate"])
        )

    def _residual_penalties(self):
        if self._ours_last_residual_action is None:
            zeros = torch.zeros(
                self.num_envs,
                dtype=torch.float32,
                device=self.device,
            )
            return zeros, zeros

        selected = self._ours_last_residual_action[:, self._ours_action_ids]
        energy = selected.square().mean(dim=-1)
        if self._ours_previous_residual_action is None:
            return energy, torch.zeros_like(energy)

        previous = self._ours_previous_residual_action[:, self._ours_action_ids]
        smooth = (selected - previous).square().mean(dim=-1)
        return energy, smooth * self._ours_previous_valid.float()

    def _compute_ours_reward(self):
        reward = self._ours_reward
        env_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        frames = self._target_frame_indices(env_ids)
        qpos_ref = self._target_qpos_ref_for_frames(frames)
        qpos_sim = self._target_dof_pos[:, : qpos_ref.shape[1]]
        q_error = (qpos_sim - qpos_ref).square().mean(dim=-1)
        object_q = torch.exp(-float(reward["k_object_q"]) * q_error)

        labels, contact_obj = self._contact_ref_for_frames(frames)
        intended = labels > 0.0
        intended_any = intended.any(dim=-1)
        active = (
            intended_any
            | (contact_obj > 0.0)
            | self._active_reference(frames, qpos_ref)
        )

        distances = self._contact_label_region_distance()
        nearest = distances.min(dim=-1).values
        intended_distances = torch.where(
            intended,
            distances,
            torch.full_like(distances, 1.0e6),
        )
        distance = torch.where(
            intended_any,
            intended_distances.min(dim=-1).values,
            nearest,
        )
        pre_contact = (
            torch.exp(-float(reward["k_pre_contact"]) * distance.square())
            * active.float()
        )

        label_force = self._contact_label_force_norm()
        intended_force = torch.where(
            intended,
            label_force,
            torch.zeros_like(label_force),
        )
        hand_force = torch.where(
            intended_any,
            intended_force.max(dim=-1).values,
            label_force.max(dim=-1).values,
        )
        target_force = self._target_contact_region_force_norm()
        min_force = float(reward["min_contact_force"])
        max_force = float(reward["max_contact_force"])
        force_ok = (
            (hand_force >= min_force)
            & (target_force >= min_force)
            & (hand_force <= max_force)
            & (target_force <= max_force)
        )
        contact_filter = (
            active
            & (distance < float(reward["contact_distance"]))
            & force_ok
        )
        contact = pre_contact * contact_filter.float()

        other_force = torch.where(
            ~intended,
            label_force,
            torch.zeros_like(label_force),
        )
        wrong_force = torch.where(
            intended_any,
            other_force.max(dim=-1).values,
            torch.zeros_like(hand_force),
        )
        wrong_threshold = float(reward["wrong_contact_force"])
        wrong_contact = torch.relu(wrong_force - wrong_threshold) / wrong_threshold
        force = (
            torch.relu(hand_force - max_force)
            + torch.relu(target_force - max_force)
        ) / max_force
        energy, smooth = self._residual_penalties()

        return {
            "object_q": object_q,
            "pre_contact_region": pre_contact,
            "contact_region": contact,
            "wrong_contact_penalty": wrong_contact,
            "force_penalty": force,
            "residual_energy_penalty": energy,
            "residual_smoothness_penalty": smooth,
            "contact_region_distance": distance,
            "hand_contact_force": hand_force,
            "target_contact_force": target_force,
        }

    def _build_ours_observation(self, env_ids):
        if self._target_dof_pos is None:
            return torch.zeros(
                (env_ids.shape[0], self._ours_extra_obs_size()),
                dtype=self.obs_buf.dtype,
                device=self.device,
            )

        frames = self._target_frame_indices(env_ids)
        qpos_ref = self._target_qpos_ref_for_frames(frames)
        qpos_sim = self._target_dof_pos[env_ids, : qpos_ref.shape[1]]
        distance = self._contact_label_region_distance()[env_ids]
        labels, contact_obj = self._contact_ref_for_frames(frames)

        root_ref = torch.cat(
            [
                self._target_default_root_pos,
                self._target_default_root_quat,
            ]
        ).expand(env_ids.shape[0], -1)
        hand_force = self._contact_label_force_norm()[env_ids].max(
            dim=-1
        ).values
        target_force = self._target_contact_region_force_norm()[env_ids]
        if self._ours_last_base_action is None:
            base_action = torch.zeros(
                (env_ids.shape[0], self.num_actions),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            base_action = self._ours_last_base_action[env_ids]

        extra = torch.cat(
            [
                qpos_sim,
                qpos_ref,
                qpos_sim - qpos_ref,
                self._target_dof_vel[env_ids, : qpos_ref.shape[1]],
                self._target_states[env_ids],
                root_ref,
                distance,
                labels,
                contact_obj.unsqueeze(-1),
                torch.log1p(hand_force).unsqueeze(-1),
                torch.log1p(target_force).unsqueeze(-1),
                base_action,
            ],
            dim=-1,
        )
        expected = self._ours_extra_obs_size()
        if extra.shape[-1] != expected:
            raise RuntimeError(
                f"ours observation has {extra.shape[-1]} values, expected {expected}"
            )
        return extra

    def _compute_task_obs(self, env_ids=None, save_buffer=True):
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )
        native = super()._compute_task_obs(env_ids, save_buffer=save_buffer)
        return torch.cat([native, self._build_ours_observation(env_ids)], dim=-1)

    def _compute_observations(self, env_ids=None):
        observation = super()._compute_observations(env_ids)
        if self._ours_base_policy is None:
            return observation
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )
        self._cache_base_action(env_ids)
        return self.obs_buf[env_ids]

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        imitation = self.rew_buf.clone()
        terms = self._compute_ours_reward()
        reward = self._ours_reward
        self.rew_buf[:] = (
            float(reward["imitation"]) * imitation
            + float(reward["object_q"]) * terms["object_q"]
            + float(reward["pre_contact_region"]) * terms["pre_contact_region"]
            + float(reward["contact_region"]) * terms["contact_region"]
            - float(reward["wrong_contact"]) * terms["wrong_contact_penalty"]
            - float(reward["force"]) * terms["force_penalty"]
            - float(reward["residual_energy"])
            * terms["residual_energy_penalty"]
            - float(reward["residual_smoothness"])
            * terms["residual_smoothness_penalty"]
        )
        self.ours_reward_terms = {
            "imitation": imitation.detach(),
            **{name: value.detach() for name, value in terms.items()},
        }

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self._ours_history_valid is not None:
            self._ours_history_valid[env_ids] = False
            self._ours_previous_valid[env_ids] = False
        for value in (
            self._ours_last_residual_action,
            self._ours_previous_residual_action,
            self._ours_last_residual_pd,
        ):
            if value is not None:
                value[env_ids] = 0.0

    def post_physics_step(self):
        super().post_physics_step()
        if not flags.im_eval:
            return

        policy = self.extras["physics_rollout"]["policy"]
        if self._ours_last_base_action is not None:
            policy["base_action"] = (
                self._ours_last_base_action.detach().cpu().numpy()
            )
        if self._ours_last_residual_action is not None:
            policy["residual_action"] = (
                self._ours_last_residual_action.detach().cpu().numpy()
            )
        if self._ours_last_residual_pd is not None:
            policy["residual_pd_delta"] = (
                self._ours_last_residual_pd.detach().cpu().numpy()
            )
        if self.ours_reward_terms is not None:
            policy["reward"] = {
                name: value.detach().cpu().numpy()
                for name, value in self.ours_reward_terms.items()
            }


__all__ = ["HumanoidImOursResidual"]
