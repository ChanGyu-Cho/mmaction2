# infer_stgcnpp_from_pkl.py
import mmengine, numpy as np
from mmaction.apis import init_recognizer, inference_recognizer
from operator import itemgetter
from pathlib import Path
from tqdm import tqdm
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--pkl', default='sample_skeleton_data.pkl',
                    help='annotation pkl with skeleton samples')
parser.add_argument('--cfg', default=r'checkpoints/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py')
parser.add_argument('--ckpt', default=r'checkpoints/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth')
parser.add_argument('--label-map', default=r'tools/data/kinetics/label_map_k400.txt')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()

# ── 1. 모델 준비 ─────────────────────────────────────────────────────────────
print('[Info] loading model …')
model = init_recognizer(args.cfg, args.ckpt, device=args.device)
model.eval()

# 라벨 맵
labels = [l.strip() for l in open(args.label_map).readlines()]

# ── 2. PKL 불러오기 ───────────────────────────────────────────────────────────
print(f'[Info] reading {args.pkl}')
data_list = mmengine.load(args.pkl)
print(f'[Info] total samples: {len(data_list)}')

# ── 3. 반복 추론 ──────────────────────────────────────────────────────────────
for idx, item in enumerate(tqdm(data_list, desc='Infer')):
    # keypoint 파일 경로 추출 (필드명이 다르면 필요에 맞게 수정)
    kp_path = item.get('keypoint') or item.get('frame_dir') or item['filename']
    if not os.path.isabs(kp_path):
        kp_path = os.path.join(os.path.dirname(args.pkl), kp_path)
    if not Path(kp_path).is_file():
        print(f'[Warn] {kp_path} not found, skip'); continue

    result = inference_recognizer(model, kp_path)
    scores = result.pred_score.tolist()
    top5 = sorted(list(enumerate(scores)), key=itemgetter(1), reverse=True)[:5]

    print(f'\nSample {idx:04d} ({Path(kp_path).name})')
    for cls_idx, score in top5:
        print(f'  {labels[cls_idx]:30s}: {score:6.3f}')

print('\n[Done] Inference finished.')
