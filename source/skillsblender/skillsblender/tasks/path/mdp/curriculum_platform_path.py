from __future__ import annotations

"""Curriculum for platform/step traversal with path-slice observations.

This module is standalone and can be plugged into an RL loop via:
- on_reset(env_ids, robot_state)
- on_episode_end(env_ids, info)

It does not modify the environment directly; instead it returns sampled
terrain parameters and generated paths for the caller to apply.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import torch


# ---------------------------
# Helpers
# ---------------------------

def _clamp(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _lerp(a: float, b: float, t: torch.Tensor) -> torch.Tensor:
    return a + (b - a) * t


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    total = torch.sum(weights)
    if float(total) <= 0.0:
        # fallback to uniform
        return torch.ones_like(weights) / weights.numel()
    return weights / total


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class CurriculumConfig:
    num_envs: int
    device: str | torch.device = "cpu"

    # Curriculum state
    difficulty_init: float = 0.0
    success_ema_alpha: float = 0.1
    fall_window: int = 20
    success_thresh: float = 0.8
    fall_rate_thresh: float = 0.1
    up_step: float = 0.02
    down_step: float = 0.01
    fail_streak_thresh: int = 5

    # Anti-forgetting mix
    mix_base: float = 0.60
    mix_down: float = 0.20
    mix_up: float = 0.20
    mix_delta: float = 0.20

    # Platform terrain ranges
    platform_height_min: float = 0.04
    platform_height_max: float = 0.20
    platform_top_len_min_easy: float = 1.2
    platform_top_len_max_easy: float = 1.8
    platform_top_len_min_hard: float = 0.4
    platform_top_len_max_hard: float = 0.8
    edge_rounding_r0: float = 0.06

    # Step count
    step_count_min: int = 1
    step_count_max: int = 8

    # Gaps/landings
    gap_enable_d: float = 0.5
    gap_len_min: float = 0.1
    gap_len_max: float = 0.35

    # Path generation
    path_resolution: float = 0.05
    curvature_amp_max: float = 0.4  # lateral amplitude (m)
    curvature_wavelength_min: float = 3.0
    curvature_wavelength_max: float = 10.0

    # Slice observation
    horizon_min: float = 0.5
    horizon_max: float = 2.0
    num_segments: int = 10

    # Logging
    num_bins: int = 6


@dataclass
class PlatformSample:
    height: float
    top_length: float
    edge_rounding: float
    num_steps: int
    gap_len: float
    terrain_type: str


@dataclass
class PathData:
    points: torch.Tensor  # (M,3)
    headings: torch.Tensor  # (M,)
    s: torch.Tensor  # (M,)


# ---------------------------
# Curriculum manager
# ---------------------------

class PlatformCurriculum:
    """Per-env curriculum for platform/step traversal."""

    TERRAIN_TYPES = ["flat", "single_step", "multi_step", "platform_landing_mix"]

    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.difficulty = torch.full((cfg.num_envs,), float(cfg.difficulty_init), device=self.device)
        self.success_ema = torch.zeros(cfg.num_envs, device=self.device)
        self.fail_streak = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)

        # fall window ring buffer
        self._fall_hist = torch.zeros(cfg.num_envs, cfg.fall_window, device=self.device)
        self._fall_ptr = 0

        # logging buffers
        self._bin_counts = torch.zeros(cfg.num_bins, device=self.device)
        self._bin_success = torch.zeros(cfg.num_bins, device=self.device)
        self._bin_fall = torch.zeros(cfg.num_bins, device=self.device)
        self._bin_track_err = torch.zeros(cfg.num_bins, device=self.device)
        self._bin_height_prog = torch.zeros(cfg.num_bins, device=self.device)
        self._bin_energy = torch.zeros(cfg.num_bins, device=self.device)

        # last sampled effective difficulty (per env)
        self._last_d_eff = torch.full((cfg.num_envs,), float(cfg.difficulty_init), device=self.device)

    # ---------------------------
    # Sampling helpers
    # ---------------------------

    def _sample_effective_d(self, env_ids: torch.Tensor) -> torch.Tensor:
        d = self.difficulty[env_ids]
        r = torch.rand(len(env_ids), device=self.device)
        d_eff = d.clone()
        # 60% keep, 20% down, 20% up
        down_mask = (r >= self.cfg.mix_base) & (r < self.cfg.mix_base + self.cfg.mix_down)
        up_mask = r >= (self.cfg.mix_base + self.cfg.mix_down)
        d_eff[down_mask] = _clamp(d_eff[down_mask] - self.cfg.mix_delta)
        d_eff[up_mask] = _clamp(d_eff[up_mask] + self.cfg.mix_delta)
        return d_eff

    def _terrain_weights(self, d: torch.Tensor) -> torch.Tensor:
        # Simple monotonic mix: flat -> single -> multi -> mix as d increases
        w_flat = torch.clamp((1.0 - d) ** 2, min=0.05)
        w_single = torch.clamp(0.6 * (1.0 - d) + 0.1, min=0.05)
        w_multi = torch.clamp(d, min=0.05)
        w_mix = torch.clamp((d - 0.5) * 2.0, min=0.0)
        weights = torch.stack([w_flat, w_single, w_multi, w_mix], dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights

    def sample_platform_params(self, d_eff: torch.Tensor) -> List[PlatformSample]:
        """Sample terrain/platform parameters for each env based on difficulty."""
        cfg = self.cfg
        n = len(d_eff)
        d = _clamp(d_eff)

        h_max = _lerp(cfg.platform_height_min, cfg.platform_height_max, d)
        h = torch.rand(n, device=self.device) * h_max

        L_min = _lerp(cfg.platform_top_len_min_easy, cfg.platform_top_len_min_hard, d)
        L_max = _lerp(cfg.platform_top_len_max_easy, cfg.platform_top_len_max_hard, d)
        L = L_min + torch.rand(n, device=self.device) * (L_max - L_min)

        r = cfg.edge_rounding_r0 * (1.0 - d)

        step_count = torch.round(_lerp(cfg.step_count_min, cfg.step_count_max, d)).to(torch.int64)
        step_count = torch.clamp(step_count, cfg.step_count_min, cfg.step_count_max)

        gap_len = torch.zeros(n, device=self.device)
        enable_gaps = d > cfg.gap_enable_d
        gap_len[enable_gaps] = cfg.gap_len_min + torch.rand(enable_gaps.sum(), device=self.device) * (
            cfg.gap_len_max - cfg.gap_len_min
        )

        weights = self._terrain_weights(d)
        terrain_idx = torch.multinomial(weights, num_samples=1).squeeze(-1)

        samples: List[PlatformSample] = []
        for i in range(n):
            samples.append(
                PlatformSample(
                    height=float(h[i].item()),
                    top_length=float(L[i].item()),
                    edge_rounding=float(r[i].item()),
                    num_steps=int(step_count[i].item()),
                    gap_len=float(gap_len[i].item()),
                    terrain_type=self.TERRAIN_TYPES[int(terrain_idx[i].item())],
                )
            )
        return samples

    # ---------------------------
    # Path generation
    # ---------------------------

    def generate_path_for_platform(self, sample: PlatformSample, d: float) -> PathData:
        """Generate a feasible path over steps/platforms with difficulty d."""
        cfg = self.cfg
        d_t = torch.tensor(float(d), device=self.device)

        # Curvature curriculum
        amp = float(_lerp(0.0, cfg.curvature_amp_max, d_t).item())
        wavelength = float(_lerp(cfg.curvature_wavelength_max, cfg.curvature_wavelength_min, d_t).item())

        # Build total length
        n_steps = max(1, sample.num_steps)
        step_height = sample.height / n_steps if n_steps > 0 else 0.0
        step_len = max(0.2, sample.top_length / n_steps)
        approach_len = max(0.6, 1.2 - 0.6 * float(d_t.item()))
        landing_len = sample.gap_len if (d_t.item() > cfg.gap_enable_d) else 0.0

        ramp_len = float(_lerp(0.6, 0.15, d_t).item())

        total_len = approach_len + n_steps * (step_len + landing_len) + 0.6
        ds = cfg.path_resolution
        s = torch.arange(0.0, total_len + ds, ds, device=self.device)

        # Elevation profile: sum of smoothed step-ups
        z = torch.zeros_like(s)
        for k in range(n_steps):
            step_start = approach_len + k * (step_len + landing_len)
            t = (s - step_start) / max(ramp_len, 1e-3)
            t = torch.clamp(t, 0.0, 1.0)
            z = z + step_height * _smoothstep(t)

        # XY path with curvature
        x = s
        if amp > 1e-4:
            y = amp * torch.sin(2.0 * math.pi * s / max(wavelength, 1e-3))
        else:
            y = torch.zeros_like(s)

        points = torch.stack([x, y, z], dim=-1)

        # Arc-length parameterization
        diffs = points[1:] - points[:-1]
        seg_len = torch.norm(diffs, dim=-1)
        s = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(seg_len, dim=0)])

        # Heading from segment tangents
        dx = diffs[:, 0]
        dy = diffs[:, 1]
        headings = torch.atan2(dy, dx)
        headings = torch.cat([headings, headings[-1:]], dim=0)

        return PathData(points=points, headings=headings, s=s)

    # ---------------------------
    # Path-slice observation
    # ---------------------------

    def slice_horizon(self, d: torch.Tensor) -> torch.Tensor:
        return _lerp(self.cfg.horizon_min, self.cfg.horizon_max, d)

    def _interp_path(self, path: PathData, s_query: float) -> Tuple[torch.Tensor, torch.Tensor]:
        s = path.s
        idx = torch.searchsorted(s, torch.tensor([s_query], device=self.device)).item()
        idx = max(1, min(idx, len(s) - 1))
        s0 = s[idx - 1]
        s1 = s[idx]
        t = float((s_query - s0) / max(float(s1 - s0), 1e-6))
        p0 = path.points[idx - 1]
        p1 = path.points[idx]
        h0 = path.headings[idx - 1]
        h1 = path.headings[idx]
        pos = p0 + (p1 - p0) * t
        yaw = h0 + (h1 - h0) * t
        return pos, yaw

    def get_path_slice_obs(
        self,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        path: PathData,
        d: float,
    ) -> torch.Tensor:
        """Compute path-slice features for a single env.

        Returns tensor of shape (num_segments, 6):
        [x_rel, y_rel, z_rel, yaw_delta, slope, curvature]
        """
        num_seg = self.cfg.num_segments
        horizon = float(self.slice_horizon(torch.tensor([d], device=self.device))[0].item())
        seg_len = horizon / num_seg

        # find nearest s on path by brute-force (small arrays ok)
        diffs = path.points[:, :2] - robot_pos[:2]
        dist2 = torch.sum(diffs * diffs, dim=-1)
        idx0 = int(torch.argmin(dist2).item())
        s0 = float(path.s[idx0].item())

        feats = []
        prev_pos, prev_yaw = self._interp_path(path, s0)
        for j in range(num_seg):
            sj = s0 + (j + 1) * seg_len
            pos_j, yaw_j = self._interp_path(path, sj)

            delta = pos_j - robot_pos
            cy = math.cos(-float(robot_yaw))
            sy = math.sin(-float(robot_yaw))
            x_rel = delta[0] * cy - delta[1] * sy
            y_rel = delta[0] * sy + delta[1] * cy
            z_rel = delta[2]

            yaw_delta = _wrap_to_pi(torch.tensor(yaw_j - float(robot_yaw), device=self.device))
            slope = (pos_j[2] - prev_pos[2]) / max(seg_len, 1e-6)
            curvature = _wrap_to_pi(torch.tensor(yaw_j - float(prev_yaw), device=self.device)) / max(seg_len, 1e-6)

            feats.append(torch.stack([x_rel, y_rel, z_rel, yaw_delta, slope, curvature]))
            prev_pos, prev_yaw = pos_j, yaw_j

        return torch.stack(feats, dim=0)

    # ---------------------------
    # Curriculum updates
    # ---------------------------

    def on_reset(self, env_ids: torch.Tensor) -> Dict[int, Dict[str, object]]:
        """Sample terrain and path for the given env ids.

        Returns a dict keyed by env_id with:
        - d_eff
        - platform_params
        - path
        """
        env_ids = env_ids.to(device=self.device)
        d_eff = self._sample_effective_d(env_ids)
        self._last_d_eff[env_ids] = d_eff

        samples = self.sample_platform_params(d_eff)
        out: Dict[int, Dict[str, object]] = {}
        for i, eid in enumerate(env_ids.tolist()):
            path = self.generate_path_for_platform(samples[i], float(d_eff[i].item()))
            out[eid] = {
                "d_eff": float(d_eff[i].item()),
                "platform_params": samples[i],
                "path": path,
            }
        return out

    def on_episode_end(self, env_ids: torch.Tensor, info: Dict[str, torch.Tensor]) -> None:
        """Update curriculum based on episode outcome.

        Expected info keys: success, fall, tracking_error, height_progress, energy
        Each is a tensor aligned with env_ids.
        """
        env_ids = env_ids.to(device=self.device)
        success = info.get("success")
        fall = info.get("fall")
        tracking_error = info.get("tracking_error")
        height_progress = info.get("height_progress")
        energy = info.get("energy")

        if success is None or fall is None:
            return

        success = success.to(self.device).float()
        fall = fall.to(self.device).float()

        # EMA success
        self.success_ema[env_ids] = (
            (1.0 - self.cfg.success_ema_alpha) * self.success_ema[env_ids]
            + self.cfg.success_ema_alpha * success
        )

        # fall window update
        self._fall_hist[env_ids, self._fall_ptr] = fall
        self._fall_ptr = (self._fall_ptr + 1) % self.cfg.fall_window
        fall_rate = torch.mean(self._fall_hist[env_ids], dim=1)

        # fail streaks
        fail_mask = (success < 0.5).long()
        self.fail_streak[env_ids] = torch.where(fail_mask > 0, self.fail_streak[env_ids] + 1, torch.zeros_like(self.fail_streak[env_ids]))

        # difficulty update
        up_mask = (self.success_ema[env_ids] > self.cfg.success_thresh) & (fall_rate < self.cfg.fall_rate_thresh)
        down_mask = self.fail_streak[env_ids] >= self.cfg.fail_streak_thresh

        d = self.difficulty[env_ids]
        d = torch.where(up_mask, d + self.cfg.up_step, d)
        d = torch.where(down_mask, d - self.cfg.down_step, d)
        self.difficulty[env_ids] = _clamp(d)

        # logging stats
        self._update_stats(env_ids, success, fall, tracking_error, height_progress, energy)

    # ---------------------------
    # Logging
    # ---------------------------

    def _update_stats(
        self,
        env_ids: torch.Tensor,
        success: torch.Tensor,
        fall: torch.Tensor,
        tracking_error: Optional[torch.Tensor],
        height_progress: Optional[torch.Tensor],
        energy: Optional[torch.Tensor],
    ) -> None:
        bins = self.cfg.num_bins
        d = self.difficulty[env_ids]
        bin_idx = torch.clamp((d * (bins - 1)).long(), 0, bins - 1)

        for i, b in enumerate(bin_idx.tolist()):
            self._bin_counts[b] += 1.0
            self._bin_success[b] += float(success[i].item())
            self._bin_fall[b] += float(fall[i].item())
            if tracking_error is not None:
                self._bin_track_err[b] += float(tracking_error[i].item())
            if height_progress is not None:
                self._bin_height_prog[b] += float(height_progress[i].item())
            if energy is not None:
                self._bin_energy[b] += float(energy[i].item())

    def summarize_curriculum_stats(self) -> Dict[str, List[float]]:
        """Return per-bin stats for logging."""
        counts = torch.clamp(self._bin_counts, min=1.0)
        return {
            "bin_counts": self._bin_counts.cpu().tolist(),
            "success_rate": (self._bin_success / counts).cpu().tolist(),
            "fall_rate": (self._bin_fall / counts).cpu().tolist(),
            "tracking_error": (self._bin_track_err / counts).cpu().tolist(),
            "height_progress": (self._bin_height_prog / counts).cpu().tolist(),
            "energy": (self._bin_energy / counts).cpu().tolist(),
        }


# ---------------------------
# Unit-test-like checks
# ---------------------------

def _test_param_ranges():
    cfg = CurriculumConfig(num_envs=4)
    cur = PlatformCurriculum(cfg)
    d = torch.tensor([0.0, 0.5, 1.0, 0.8])
    samples = cur.sample_platform_params(d)
    for i, s in enumerate(samples):
        assert 0.0 <= s.height <= cfg.platform_height_max
        assert cfg.step_count_min <= s.num_steps <= cfg.step_count_max
        assert s.top_length > 0.0


def _test_d_update():
    cfg = CurriculumConfig(num_envs=2)
    cur = PlatformCurriculum(cfg)
    env_ids = torch.tensor([0, 1])
    # simulate success
    info = {
        "success": torch.tensor([1.0, 1.0]),
        "fall": torch.tensor([0.0, 0.0]),
        "tracking_error": torch.tensor([0.1, 0.1]),
        "height_progress": torch.tensor([1.0, 1.0]),
        "energy": torch.tensor([0.5, 0.5]),
    }
    for _ in range(10):
        cur.on_episode_end(env_ids, info)
    assert float(cur.difficulty[0].item()) > 0.0


if __name__ == "__main__":
    _test_param_ranges()
    _test_d_update()


# ---------------------------
# Usage snippet
# ---------------------------
#
# cfg = CurriculumConfig(num_envs=env.num_envs, device=env.device)
# curriculum = PlatformCurriculum(cfg)
#
# def on_reset(env_ids):
#     samples = curriculum.on_reset(env_ids)
#     for eid, data in samples.items():
#         # data["platform_params"] -> use to configure terrain/props
#         # data["path"] -> use to populate PATH-SLICE instruction
#         pass
#
# def on_episode_end(env_ids, info):
#     # info should include: success, fall, tracking_error, height_progress, energy
#     curriculum.on_episode_end(env_ids, info)
#     stats = curriculum.summarize_curriculum_stats()
#     # log stats
