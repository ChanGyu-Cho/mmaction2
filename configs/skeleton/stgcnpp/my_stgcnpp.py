_base_ = '../../_base_/default_runtime.py'

# FP16 Mixed-Precision 설정
fp16 = dict(type='Fp16OptimizerHook', loss_scale='dynamic')

load_from = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=2,
        in_channels=256,
        dropout=0.7 # Dropout 비율 조정
    )
)

dataset_type = 'PoseDataset'
ann_file = r"D:\golfDataset\스포츠 사람 동작 영상(골프)\Training\Public\male\train\crop_pkl\skeleton_dataset_90_10.pkl"

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='xsub_train'
        )
    )
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='xsub_val',
        test_mode=True
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='xsub_val',
        test_mode=True
    )
)

val_evaluator = [
    dict(type='AccMetric'),
]
test_evaluator = val_evaluator

# Training schedule 조정으로 과적합 억제
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[3, 6],
        gamma=0.1
    )
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
    )
)

default_hooks = dict(
    checkpoint=dict(interval=1),
    logger=dict(interval=100)
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='val/top1_acc',
        patience=5,
        min_delta=1e-3
    )
]

auto_scale_lr = dict(enable=False, base_batch_size=128)
