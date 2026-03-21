#!/usr/bin/env python3
"""
Extract LIBERO demonstrations into a DROID-style dataset layout expected by:
`vjepa2/app/vjepa_droid/droid.py`.

Output per trajectory directory:
  - metadata.json
  - trajectory.h5
  - recordings/MP4/agentview.mp4

And one CSV file listing trajectory directories for training.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

import h5py
import imageio
import numpy as np
from scipy.spatial.transform import Rotation

# Ensure local package import works when running from notebooks/ directory.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def parse_task_ids(task_ids_arg: str, max_tasks: int) -> List[int]:
    """Parse task-id spec: 'all', '5', '1,2,9', '3-7'."""
    task_ids_arg = task_ids_arg.strip().lower()
    if task_ids_arg == "all":
        return list(range(max_tasks))
    if "-" in task_ids_arg:
        start, end = task_ids_arg.split("-", 1)
        lo, hi = int(start), int(end)
        if lo > hi:
            raise ValueError(f"Invalid task range: {task_ids_arg}")
        return [t for t in range(lo, hi + 1) if 0 <= t < max_tasks]
    out = []
    for tok in task_ids_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        tid = int(tok)
        if 0 <= tid < max_tasks:
            out.append(tid)
    if not out:
        raise ValueError(f"No valid task IDs parsed from: {task_ids_arg}")
    return out


def obs_to_state7(obs: dict) -> np.ndarray:
    """
    Build 7D state as [x, y, z, rx, ry, rz, gripper] to match droid.py.
    Orientation uses XYZ Euler (radians) derived from end-effector quaternion.
    """
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
    euler = Rotation.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)

    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    gripper = np.array([gripper_qpos[0]], dtype=np.float32)
    return np.concatenate([pos, euler, gripper], axis=0)


def write_mp4(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def write_trajectory_h5(path: Path, states_7d: np.ndarray, extrinsics_6d: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        obs = f.create_group("observation")
        robot_state = obs.create_group("robot_state")
        robot_state.create_dataset("cartesian_position", data=states_7d[:, :6].astype(np.float32))
        robot_state.create_dataset("gripper_position", data=states_7d[:, 6].astype(np.float32))
        cam_extr = obs.create_group("camera_extrinsics")
        # droid.py builds key as f"{camera_name}_left", where camera_name is mp4 stem.
        cam_extr.create_dataset("agentview_left", data=extrinsics_6d.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LIBERO demos to DROID-style world-model dataset.")
    parser.add_argument("--benchmark-name", type=str, default="libero_spatial")
    parser.add_argument("--task-ids", type=str, default="all", help="Examples: all, 5, 1,2,3, 5-9")
    parser.add_argument("--max-demos-per-task", type=int, default=-1)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "notebooks" / "libero_wm_droid"),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional explicit CSV path. Default: <output-root>/<benchmark-name>_train_paths.csv",
    )
    args = parser.parse_args()

    benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark_name]()
    num_tasks = benchmark_instance.get_num_tasks()
    task_ids = parse_task_ids(args.task_ids, num_tasks)
    bddl_root = get_libero_path("bddl_files")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else output_root / f"{args.benchmark_name}_train_paths.csv"
    )

    trajectory_dirs: List[Path] = []
    print(f"Extracting benchmark={args.benchmark_name} tasks={task_ids} into {output_root}")

    for task_id in task_ids:
        task = benchmark_instance.get_task(task_id)
        demo_file = Path(get_libero_path("datasets")) / benchmark_instance.get_task_demonstration(task_id)
        if not demo_file.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_file}")

        env_args = {
            "bddl_file_name": os.path.join(bddl_root, task.problem_folder, task.bddl_file),
            "camera_heights": args.camera_height,
            "camera_widths": args.camera_width,
        }
        task_dir = output_root / args.benchmark_name / f"{task_id:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(demo_file, "r") as f:
            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))
            if args.max_demos_per_task > 0:
                demo_keys = demo_keys[: args.max_demos_per_task]

            for ep_key in demo_keys:
                ep_idx = int(ep_key.split("_")[1])
                states = f[f"data/{ep_key}/states"][()]
                actions = f[f"data/{ep_key}/actions"][()].astype(np.float32)
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"{demo_file}:{ep_key} has unexpected action shape {actions.shape}, expected [N, 7]")

                traj_dir = task_dir / f"demo_{ep_idx:05d}"
                mp4_path = traj_dir / "recordings" / "MP4" / "agentview.mp4"
                h5_path = traj_dir / "trajectory.h5"
                meta_path = traj_dir / "metadata.json"

                print(f"[task {task_id}] replaying {ep_key} -> {traj_dir}")
                env = OffScreenRenderEnv(**env_args)
                try:
                    env.seed(args.seed)
                    env.reset()
                    obs = env.set_init_state(states[0])

                    frames = []
                    pose7_seq = []
                    #settle env
                    for _ in range(5):
                        obs, _, _, _ = env.step([0.] * 7)
                    for action in actions:
                        # Keep same visual convention as quick_walkthrough.ipynb.
                        frames.append(np.asarray(obs["agentview_image"][::-1], dtype=np.uint8))
                        pose7_seq.append(obs_to_state7(obs))
                        obs, _, _, _ = env.step(action)
                    frames.append(np.asarray(obs["agentview_image"][::-1], dtype=np.uint8))
                    pose7_seq.append(obs_to_state7(obs))
                finally:
                    env.close()

                states_7d = np.stack(pose7_seq, axis=0).astype(np.float32)  # [T,7], T=N_actions+1
                extrinsics_6d = np.zeros((states_7d.shape[0], 6), dtype=np.float32)

                write_mp4(mp4_path, frames, fps=args.fps)
                write_trajectory_h5(h5_path, states_7d, extrinsics_6d)

                metadata = {
                    "left_mp4_path": "recordings/MP4/agentview.mp4",
                    "right_mp4_path": "recordings/MP4/agentview.mp4",
                    "benchmark_name": args.benchmark_name,
                    "task_id": int(task_id),
                    "demo_key": ep_key,
                    "camera_name": "agentview",
                    "fps": int(args.fps),
                    "num_actions": int(actions.shape[0]),
                    "num_frames": int(states_7d.shape[0]),
                    "notes": "State is [xyz, euler_xyz, gripper]. Extrinsics are zero placeholders [x,y,z,rx,ry,rz].",
                }
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(metadata, mf, indent=2)

                trajectory_dirs.append(traj_dir)

    with open(csv_path, "w", encoding="utf-8") as cf:
        for traj_dir in trajectory_dirs:
            cf.write(f"{traj_dir}\n")

    print(f"Done. Wrote {len(trajectory_dirs)} trajectories.")
    print(f"Train CSV: {csv_path}")


if __name__ == "__main__":
    main()
