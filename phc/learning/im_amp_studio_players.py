"""Studio recorder layered on the unchanged PHC-X evaluation player."""

from __future__ import annotations

import inspect
import linecache
import os.path as osp
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from learning.im_amp_players import IMAMPPlayerContinuous
from phc.utils.flags import flags


def _skip_only_author_final_breakpoint():
    frame = inspect.currentframe()
    while frame is not None:
        path = Path(frame.f_code.co_filename)
        if path.name == "im_amp_players.py":
            following = "".join(
                linecache.getline(str(path), frame.f_lineno + offset)
                for offset in range(1, 4)
            )
            if "joblib.dump(failed_keys" in following:
                return
            raise RuntimeError(
                "PHC-X author player entered an unexpected debugger at "
                f"{path}:{frame.f_lineno}"
            )
        frame = frame.f_back
    raise RuntimeError("PHC-X author player breakpoint origin is unknown")


def _stack_frames(values):
    first = values[0]
    if isinstance(first, dict):
        keys = tuple(first)
        if any(not isinstance(value, dict) or tuple(value) != keys for value in values):
            raise ValueError("PHC-X Studio rollout schema changed within one episode")
        return {
            key: _stack_frames([value[key] for value in values])
            for key in keys
        }
    return np.stack([np.asarray(value) for value in values])


def _select_environment(values, env_index: int):
    if isinstance(values, dict):
        return {
            key: _select_environment(value, env_index)
            for key, value in values.items()
        }
    return values[:, env_index]


class IMAMPStudioPlayerContinuous(IMAMPPlayerContinuous):
    """Add common-adapter telemetry without replacing PHC-X policy execution."""

    def __init__(self, config):
        super().__init__(config)
        self._studio_frames = []
        self._studio_written = False

    def _post_step(self, info, done):
        if flags.im_eval:
            if "physics_rollout" not in info:
                raise KeyError("HumanoidImPassiveObject did not emit physics_rollout")
            self._studio_frames.append(info["physics_rollout"])

        # The author player enters ipdb after its final metric print. Studio is
        # headless, so suppress only that interactive breakpoint in this opt-in
        # subclass; all author termination and metric code still executes.
        try:
            import ipdb
        except ImportError as error:
            raise RuntimeError(
                "PHC-X Studio recording requires ipdb==0.13.13 because the "
                "unchanged author player calls ipdb.set_trace at completion"
            ) from error
        with patch.object(
            ipdb,
            "set_trace",
            side_effect=_skip_only_author_final_breakpoint,
        ):
            done = super()._post_step(info, done)

        if (
            flags.im_eval
            and not self._studio_written
            and self._studio_frames
            and bool(done.all())
        ):
            task = self.env.task
            if task.num_envs != 1 or task._motion_lib._num_unique_motions != 1:
                raise ValueError("PHC-X Studio recorder requires one env and one motion")
            rollout = _select_environment(_stack_frames(self._studio_frames), 0)
            summary = {
                "physics_rollout": {
                    str(task._motion_lib._motion_data_keys[0]): rollout,
                },
                "physics_rollout_metadata": {
                    "human": {
                        "dof_body_names": list(task._dof_names),
                    },
                    "object": {
                        "joint_names": list(task._target_joint_names),
                    },
                    "contact": {
                        "target_body_names": list(task._target_body_names),
                        "human_body_names": list(task._body_names),
                        "label_names": list(task._phc_contact_label_names_10),
                        "granularity": str(
                            getattr(task, "_phc_contact_granularity", "hand2")
                        ),
                        "source": "surface_distance_and_net_contact_force",
                    },
                },
            }
            joblib.dump(
                summary,
                osp.join(self.config["network_path"], "phc_eval_summary.pkl"),
            )
            self._studio_written = True
        return done


__all__ = ["IMAMPStudioPlayerContinuous"]
