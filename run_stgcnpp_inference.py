#!/usr/bin/env python
import argparse
import os
import pickle
import subprocess
import sys
from typing import Tuple

import pandas as pd
import numpy as np


def convert_csv_to_pkl(csv_path: str,
                       pkl_path: str,
                       frame_dir: str,
                       label: int = 0,
                       img_shape: Tuple[int, int] = (1080, 1920)) -> None:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[CSV→PKL] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[1] % 3 != 0:
        raise ValueError(f"[CSV→PKL] Expected columns multiple of 3, got {df.shape[1]}")
    T = len(df)
    V = df.shape[1] // 3
    M, C = 1, 2
    kp = np.zeros((M, C, T, V), dtype=np.float32)
    kp_score = np.zeros((M, T, V), dtype=np.float32)
    for t, row in df.iterrows():
        for v in range(V):
            x, y, s = row[3*v:3*v+3]
            kp[0, 0, t, v] = x
            kp[0, 1, t, v] = y
            kp_score[0, t, v] = s
    sample = {
        'frame_dir': frame_dir,
        'label': label,
        'img_shape': img_shape,
        'original_shape': img_shape,
        'total_frames': T,
        'keypoint': kp,
        'keypoint_score': kp_score
    }
    data = {
        'split': {'xsub_val': [frame_dir]},
        'annotations': [sample]
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[SUCCESS] Wrote PKL: {pkl_path} (T={T}, V={V}, M={M}, C={C})")


def main():
    p = argparse.ArgumentParser(description="CSV→PKL + ST-GCN++ inference")
    p.add_argument('--csv',       required=True,  help='OpenPose CSV')
    p.add_argument('--cfg',       required=True,  help='MMAction2 config.py')
    p.add_argument('--ckpt',      required=True,  help='pretrained .pth')
    p.add_argument('--frame-dir', required=True,  help='video ID')
    p.add_argument('--label',     type=int, default=0, help='dummy label')
    p.add_argument('--img-shape', nargs=2, type=int, default=[1080, 1920],
                   help='H W')
    p.add_argument('--pkl',       default='temp_skel.pkl', help='temp PKL path')
    p.add_argument('--device',    default='cuda:0', help='cuda:0 or cpu')
    args = p.parse_args()

    try:
        convert_csv_to_pkl(
            csv_path=args.csv,
            pkl_path=args.pkl,
            frame_dir=args.frame_dir,
            label=args.label,
            img_shape=tuple(args.img_shape)
        )
    except Exception as e:
        print(f"[ERROR] conversion failed:\n  {e}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, 'tools/test.py',
        args.cfg, args.ckpt,
        '--cfg-options', f"test_dataloader.dataset.ann_file={args.pkl}",
        '--cfg-options',      "test_dataloader.dataset.split=xsub_val"
    ]
    env = os.environ.copy()
    if args.device.startswith('cuda'):
        env['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''

    print(f"[RUNNING] {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"[ERROR] inference failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)
    print("[DONE] inference succeeded.")


if __name__ == '__main__':
    main()
