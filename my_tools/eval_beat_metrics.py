import sys
sys.path.append(".")

import argparse
import os
from glob import glob

import numpy as np

from analysis.pymo.parsers import BVHParser
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from my_tools.plot_beats import vis_music_motion_movie
from my_tools.processing import *


def cal_beat_diff(music_beats, motion_beats, fps=20):
    """Calculate beat difference"""
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]

    if len(motion_beat_idxs) == 0 or len(music_beat_idxs) == 0:
        return []

    beat_diff = []
    for motion_beat_idx in motion_beat_idxs:
        diff_frame = min(np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32))
        diff_time = diff_frame / fps
        beat_diff.append(diff_time)
    return beat_diff


def main(args):
    # must set fps first, which affects the following evaluator
    set_fps(fps=20)

    wav_files = glob(os.path.join(args.res_dir, "*.wav"))
    base_names = [os.path.basename(wav_file).replace(".wav", "") for wav_file in wav_files]
    bvh_files = [os.path.join(args.res_dir, base_name + ".bvh") for base_name in base_names]

    metrics = {
        "beat_coverage": [],
        "beat_hit": [],
        "beat_alignment": [],
        "beat_alignment_music": [],
        "beat_energy": [],
        "motion_energy": [],
        "beat_diff": []
    }

    for wav_file, bvh_file, base_name in zip(wav_files, bvh_files, base_names):
        bvh_data = BVHParser().parse(bvh_file)
        bvh2pos = MocapParameterizer("position")
        pos_data = bvh2pos.fit_transform([bvh_data])
        joints = pos_data[0].values.values
        num_frames = joints.shape[0]
        joints = joints.reshape(num_frames, -1, 3)

        music_features = music_features_all(wav_file, tempo=120)
        music_envelope, music_beats = music_features["envelope"], music_features["beat_onehot"]
        motion_envelope, motion_beats, motion_beats_energy = motion_peak_onehot(joints)

        # align music and motion features
        min_len = min(len(music_envelope), len(motion_envelope))
        music_envelope = music_envelope[:min_len]
        music_beats = music_beats[:min_len]
        motion_envelope = motion_envelope[:min_len]
        motion_beats = motion_beats[:min_len]
        motion_beats_energy = motion_beats_energy[:min_len]

        music_beats_aligned, motion_beats_aligned = select_aligned(
            music_beats, motion_beats, tol=12
        )

        metrics["beat_coverage"].append(motion_beats.sum() / (music_beats.sum() + EPS))
        metrics["beat_hit"].append(len(motion_beats_aligned) / (motion_beats.sum() + EPS))
        metrics["beat_alignment"].append(alignment_score(music_beats, motion_beats, sigma=3))
        metrics["beat_alignment_music"].append(alignment_score(motion_beats, music_beats, sigma=3))
        metrics["beat_energy"].append(
            motion_beats_energy[motion_beats].mean() if motion_beats.sum() > 0 else 0.0
        )
        metrics["motion_energy"].append(motion_envelope.mean())
        metrics["beat_diff"].extend(cal_beat_diff(music_beats, motion_beats, fps=20))

        if args.vis:
            vis_music_motion_movie(
                music_envelope,
                music_beats,
                np.array(range(min_len)),
                motion_envelope,
                motion_beats,
                np.array(range(min_len)),
                out_dir=args.vis_dir,
                fname=base_name,
                num_frames=joints.shape[0],
                to_video=True,
                audio_path=wav_file,
                fps=20,
            )
    print("==================Metrics==================")
    for k, v in metrics.items():
        if k == "beat_diff":
            mean = np.mean(metrics[k])
            var = np.var(metrics[k])
            print(f"{k}: {mean}(mean) {var}(std)")
        else:
            metrics[k] = sum(v) / len(v)
            print(f"{k}: {metrics[k]:.3f}")
    print("===========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", default="res_dir")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis_dir", default="res_dir/videos")
    args = parser.parse_args()
    main(args)
