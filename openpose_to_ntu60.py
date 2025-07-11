#!/usr/bin/env python
"""
올바른 OpenPose BODY25 → NTU RGB+D 25 키포인트 변환기
키포인트 개수를 맞추고 올바른 매핑을 적용
"""
import argparse
import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd

# OpenPose BODY25 키포인트 인덱스 (0-24)
BODY25_KEYPOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

# NTU RGB+D 25 키포인트 형식에 맞춘 매핑
# NTU RGB+D는 Kinect V2의 25개 관절을 사용
# 참고: https://github.com/shahroudy/NTURGB-D
NTU_25_KEYPOINTS = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight"
]

# OpenPose BODY25 → NTU RGB+D 25 직접 매핑
# 완전히 일치하지 않는 키포인트는 가장 유사한 것으로 매핑
BODY25_TO_NTU25_MAPPING = [
    8,   # 0: SpineBase ← MidHip (8)
    1,   # 1: SpineMid ← Neck (1) 
    1,   # 2: Neck ← Neck (1)
    0,   # 3: Head ← Nose (0)
    5,   # 4: ShoulderLeft ← LShoulder (5)
    6,   # 5: ElbowLeft ← LElbow (6)
    7,   # 6: WristLeft ← LWrist (7)
    7,   # 7: HandLeft ← LWrist (7) - 근사
    2,   # 8: ShoulderRight ← RShoulder (2)
    3,   # 9: ElbowRight ← RElbow (3)
    4,   # 10: WristRight ← RWrist (4)
    4,   # 11: HandRight ← RWrist (4) - 근사
    12,  # 12: HipLeft ← LHip (12)
    13,  # 13: KneeLeft ← LKnee (13)
    14,  # 14: AnkleLeft ← LAnkle (14)
    21,  # 15: FootLeft ← LHeel (21)
    9,   # 16: HipRight ← RHip (9)
    10,  # 17: KneeRight ← RKnee (10)
    11,  # 18: AnkleRight ← RAnkle (11)
    24,  # 19: FootRight ← RHeel (24)
    1,   # 20: SpineShoulder ← Neck (1) - 근사
    7,   # 21: HandTipLeft ← LWrist (7) - 근사
    7,   # 22: ThumbLeft ← LWrist (7) - 근사
    4,   # 23: HandTipRight ← RWrist (4) - 근사
    4,   # 24: ThumbRight ← RWrist (4) - 근사
]

def convert_body25_to_ntu25(body25_keypoints, body25_scores):
    """
    OpenPose BODY25 → NTU RGB+D 25 키포인트 변환 (단순 매핑)
    Args:
        body25_keypoints: (M, T, 25, 2) 형태
        body25_scores: (M, T, 25) 형태
    Returns:
        ntu25_keypoints: (M, T, 25, 2) 형태
        ntu25_scores: (M, T, 25) 형태
    """
    M, T, V, C = body25_keypoints.shape
    ntu25_keypoints = np.zeros((M, T, 25, C), dtype=np.float32)
    ntu25_scores = np.zeros((M, T, 25), dtype=np.float32)
    
    # 직접 매핑
    for ntu25_idx, body25_idx in enumerate(BODY25_TO_NTU25_MAPPING):
        ntu25_keypoints[:, :, ntu25_idx, :] = body25_keypoints[:, :, body25_idx, :]
        ntu25_scores[:, :, ntu25_idx] = body25_scores[:, :, body25_idx]
    
    # SpineMid 보간 (SpineBase와 Neck의 중점)
    spine_base_valid = ntu25_scores[:, :, 0] > 0  # SpineBase
    neck_valid = ntu25_scores[:, :, 2] > 0        # Neck
    both_valid = spine_base_valid & neck_valid
    
    # 유효한 프레임에서만 보간
    ntu25_keypoints[:, :, 1, :] = (ntu25_keypoints[:, :, 0, :] + ntu25_keypoints[:, :, 2, :]) / 2.0
    ntu25_scores[:, :, 1] = (ntu25_scores[:, :, 0] + ntu25_scores[:, :, 2]) / 2.0
    
    # 둘 다 유효하지 않은 경우 0으로 설정
    for m in range(M):
        for t in range(T):
            if not both_valid[m, t]:
                ntu25_keypoints[m, t, 1, :] = 0
                ntu25_scores[m, t, 1] = 0
    
    return ntu25_keypoints, ntu25_scores

def normalize_coordinates(keypoints, img_shape, method='skeleton_center'):
    height, width = img_shape
    normalized_kp = keypoints.copy()
    
    if method == 'skeleton_center':
        for m in range(keypoints.shape[0]):
            for t in range(keypoints.shape[1]):
                valid_mask = (keypoints[m, t, :, 0] != 0) & (keypoints[m, t, :, 1] != 0)
                if valid_mask.any():
                    valid_kp = keypoints[m, t, valid_mask, :]
                    
                    # 몸통 중심 (SpineBase, SpineMid, Neck)을 기준으로 정규화
                    torso_indices = [0, 1, 2]  # NTU25 기준
                    torso_mask = np.isin(np.where(valid_mask)[0], torso_indices)
                    
                    if torso_mask.any():
                        # ⚠️ 수정된 부분: valid_kp[torso_mask] 로만 인덱싱
                        torso_kp = valid_kp[torso_mask]
                        center_x = torso_kp[:, 0].mean()
                        center_y = torso_kp[:, 1].mean()
                        
                        bbox_w = valid_kp[:, 0].max() - valid_kp[:, 0].min()
                        bbox_h = valid_kp[:, 1].max() - valid_kp[:, 1].min()
                        scale = max(bbox_w, bbox_h, 1.0)
                        
                        normalized_kp[m, t, :, 0] = (keypoints[m, t, :, 0] - center_x) / scale
                        normalized_kp[m, t, :, 1] = (keypoints[m, t, :, 1] - center_y) / scale
                    else:
                        # torso 관절이 없으면 전체 중심 사용
                        center_x = valid_kp[:, 0].mean()
                        center_y = valid_kp[:, 1].mean()
                        bbox_w = valid_kp[:, 0].max() - valid_kp[:, 0].min()
                        bbox_h = valid_kp[:, 1].max() - valid_kp[:, 1].min()
                        scale = max(bbox_w, bbox_h, 1.0)
                        
                        normalized_kp[m, t, :, 0] = (keypoints[m, t, :, 0] - center_x) / scale
                        normalized_kp[m, t, :, 1] = (keypoints[m, t, :, 1] - center_y) / scale
    elif method == '0to1':
        normalized_kp[:, :, :, 0] = keypoints[:, :, :, 0] / width
        normalized_kp[:, :, :, 1] = keypoints[:, :, :, 1] / height
    
    return normalized_kp


def convert_csv_to_pkl(csv_path, pkl_path, frame_dir,
                       label=0, img_shape=(1080, 1920),
                       normalize_method='skeleton_center',
                       confidence_threshold=0.1):
    """OpenPose BODY25 → NTU RGB+D 25 키포인트 변환"""
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[CSV→PKL] CSV not found: {csv_path}")
    
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.shape[1] != 25 * 3:
        raise ValueError(f"[CSV→PKL] Expect 25*3 columns, got {df.shape[1]}")
    
    T = len(df)
    V = 25
    M, C = 1, 2
    
    print(f"[INFO] Processing {T} frames with {V} keypoints")
    
    # 1. BODY25 데이터 로드
    body25_kp = np.zeros((M, T, V, C), dtype=np.float32)
    body25_score = np.zeros((M, T, V), dtype=np.float32)
    
    for t, row in df.iterrows():
        for v in range(V):
            x, y, s = row[3*v:3*v+3]
            body25_kp[0, t, v, 0] = x
            body25_kp[0, t, v, 1] = y
            body25_score[0, t, v] = s
    
    print(f"[INFO] BODY25 coordinate range: X[{body25_kp[:,:,:,0].min():.1f}, {body25_kp[:,:,:,0].max():.1f}], Y[{body25_kp[:,:,:,1].min():.1f}, {body25_kp[:,:,:,1].max():.1f}]")
    
    # 2. 신뢰도 필터링
    if confidence_threshold > 0:
        low_conf_mask = body25_score < confidence_threshold
        body25_kp[low_conf_mask] = 0
        body25_score[low_conf_mask] = 0
        print(f"[INFO] Applied confidence filtering (threshold: {confidence_threshold})")
    
    # 3. BODY25 → NTU25 변환
    ntu25_kp, ntu25_score = convert_body25_to_ntu25(body25_kp, body25_score)
    print(f"[INFO] Converted BODY25 → NTU25: {ntu25_kp.shape}")
    
    # 4. 좌표 정규화
    ntu25_kp = normalize_coordinates(ntu25_kp, img_shape, normalize_method)
    print(f"[INFO] Applied coordinate normalization ({normalize_method})")
    print(f"[INFO] NTU25 coordinate range: X[{ntu25_kp[:,:,:,0].min():.4f}, {ntu25_kp[:,:,:,0].max():.4f}], Y[{ntu25_kp[:,:,:,1].min():.4f}, {ntu25_kp[:,:,:,1].max():.4f}]")
    
    # 5. NTU RGB+D 형식으로 패키징
    sample = {
        'frame_dir': frame_dir,
        'label': label,
        'img_shape': img_shape,
        'original_shape': img_shape,
        'total_frames': T,
        'keypoint': ntu25_kp,          # (1, T, 25, 2)
        'keypoint_score': ntu25_score  # (1, T, 25)
    }
    
    data = {
        'split': {'xsub_val': [frame_dir]},
        'annotations': [sample]
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"[SUCCESS] Wrote PKL: {pkl_path} (T={T}, V=25, M={M}, C={C})")
    
    # 6. 변환 결과 요약
    print(f"\n=== Conversion Summary ===")
    print(f"Input format: OpenPose BODY25 (25 keypoints)")
    print(f"Output format: NTU RGB+D 25 (25 keypoints)")
    print(f"Frames: {T}")
    print(f"Normalization: {normalize_method}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # 키포인트 유효성 통계
    valid_counts = np.sum(ntu25_score > 0, axis=(0, 1))
    print(f"\nKeypoint validity statistics:")
    for i, (name, count) in enumerate(zip(NTU_25_KEYPOINTS, valid_counts)):
        print(f"  {i:2d}. {name:15s}: {count:4d}/{T} ({count/T*100:.1f}%)")
    
    return data

def main():
    parser = argparse.ArgumentParser(
        description="OpenPose BODY25 → NTU RGB+D 25 Keypoint Converter + ST-GCN++ Inference")
    
    parser.add_argument('--csv', required=True, help='OpenPose BODY25 CSV path')
    parser.add_argument('--frame-dir', required=True, help='video ID (frame_dir)')
    parser.add_argument('--pkl', default='temp_ntu25_skel.pkl', help='output PKL path')
    parser.add_argument('--label', type=int, default=0, help='dummy label')
    parser.add_argument('--img-shape', nargs=2, type=int, default=[1080,1920], help='H W')
    parser.add_argument('--normalize', default='skeleton_center', 
                       choices=['skeleton_center', '0to1'],
                       help='coordinate normalization method')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='confidence threshold for filtering keypoints')
    
    # 추론 매개변수
    parser.add_argument('--cfg', required=True, help='MMAction2 config.py')
    parser.add_argument('--ckpt', required=True, help='pretrained .pth')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--dump', default='result.pkl', help='path to dump results')
    
    args = parser.parse_args()
    
    try:
        convert_csv_to_pkl(
            csv_path=args.csv,
            pkl_path=args.pkl,
            frame_dir=args.frame_dir,
            label=args.label,
            img_shape=tuple(args.img_shape),
            normalize_method=args.normalize,
            confidence_threshold=args.confidence_threshold
        )
    except Exception as e:
        print(f"[ERROR] conversion failed:\n  {e}", file=sys.stderr)
        sys.exit(1)
    
    # MMAction2 추론 실행
    cmd = [
        sys.executable, 'tools/test.py',
        args.cfg, args.ckpt,
        '--dump', args.dump,
        '--cfg-options', f"test_dataloader.dataset.ann_file={args.pkl}",
        '--cfg-options', "test_dataloader.dataset.split=xsub_val"
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1] if args.device.startswith('cuda') else ''
    
    print(f"\n[RUNNING] {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, text=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"[ERROR] inference failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)
    
    if os.path.isfile(args.dump):
        print(f"[DONE] inference succeeded, results dumped to {args.dump}")
    else:
        print(f"[WARN] inference succeeded but no dump found at {args.dump}")

if __name__ == '__main__':
    main()