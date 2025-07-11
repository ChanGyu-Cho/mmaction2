#!/usr/bin/env python
import argparse
import os
import pickle
import subprocess
import sys

import pandas as pd
import numpy as np

# BODY_25 → COCO(17) 매핑 인덱스
# COCO: [nose, l-eye, r-eye, l-ear, r-ear,
#        l-shoulder, r-shoulder, l-elbow, r-elbow,
#        l-wrist, r-wrist, l-hip, r-hip,
#        l-knee, r-knee, l-ankle, r-ankle]
MAPPING_BODY25_TO_COCO17 = [
    0,   # nose
    16,  # left eye
    15,  # right eye
    18,  # left ear
    17,  # right ear
    5,   # left shoulder
    2,   # right shoulder
    6,   # left elbow
    3,   # right elbow
    7,   # left wrist
    4,   # right wrist
    12,  # left hip
    9,   # right hip
    13,  # left knee
    10,  # right knee
    14,  # left ankle
    11   # right ankle
]

def convert_csv_to_pkl(csv_path, pkl_path, frame_dir,
                       label=0, img_shape=(1080, 1920)):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[CSV→PKL] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # 한 프레임당 3*25 컬럼이 아니면 에러
    if df.shape[1] != 25 * 3:
        raise ValueError(f"[CSV→PKL] Expect 25*3 columns, got {df.shape[1]}")
    T = len(df)
    V25 = 25
    C = 2
    M = 1

    # BODY_25 raw (M, T, V25, C) / (M, T, V25)
    kp25 = np.zeros((M, T, V25, C), dtype=np.float32)
    score25 = np.zeros((M, T, V25), dtype=np.float32)
    for t, row in df.iterrows():
        for v in range(V25):
            x, y, s = row[3*v:3*v+3]
            kp25[0, t, v, 0] = x
            kp25[0, t, v, 1] = y
            score25[0, t, v]   = s

    # COCO(17) 로 재매핑
    kp17    = kp25[:, :, MAPPING_BODY25_TO_COCO17, :]
    score17 = score25[:, :, MAPPING_BODY25_TO_COCO17]

    # NTU-2D 포맷 샘플
    sample = {
        'frame_dir': frame_dir,
        'label': label,
        'img_shape': img_shape,
        'original_shape': img_shape,
        'total_frames': T,
        'keypoint': kp17,          # (1, T, 17, 2)
        'keypoint_score': score17  # (1, T, 17)
    }
    data = {
        'split': {'xsub_val': [frame_dir]},
        'annotations': [sample]
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[SUCCESS] Wrote PKL: {pkl_path} (T={T}, V=17, M={M}, C={C})")

def main():
    parser = argparse.ArgumentParser(
        description="OpenPose BODY25→COCO17 PKL + ST-GCN++ Inference")
    parser.add_argument('--csv',       required=True,
                        help='OpenPose BODY25 CSV path')
    parser.add_argument('--frame-dir', required=True,
                        help='video ID (frame_dir)')
    parser.add_argument('--pkl',       default='temp_skel.pkl',
                        help='output PKL path')
    parser.add_argument('--label',     type=int, default=0,
                        help='dummy label')
    parser.add_argument('--img-shape',  nargs=2, type=int,
                        default=[1080,1920], help='H W')
    parser.add_argument('--cfg',       required=True,
                        help='MMAction2 config.py')
    parser.add_argument('--ckpt',      required=True,
                        help='pretrained .pth')
    parser.add_argument('--device',    default='cuda:0',
                        help='cuda:0 or cpu')
    args = parser.parse_args()

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

    # tools/test.py 로 inference
    cmd = [
        sys.executable, 'tools/test.py',
        args.cfg, args.ckpt,
        '--cfg-options',
        f"test_dataloader.dataset.ann_file={args.pkl}",
        '--cfg-options',
        "test_dataloader.dataset.split=xsub_val"
    ]
    env = os.environ.copy()
    # CUDA_VISIBLE_DEVICES 설정
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
