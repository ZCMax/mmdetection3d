dataset_type = 'KittiMonoDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scales = [
    (1344, 448),
    (1312, 416),
    (1280, 384),
    (1248, 352),
    (1216, 320),
    (1344, 384),
    (1312, 384),
    (1248, 384),
    (1216, 384),
    (1280, 448),
    (1280, 416),
    (1280, 352),
    (1280, 320),
]

# img_scales=[(1088, 192), (1152, 256), (1216, 320), (1280, 384),
#             (1344, 448), (1408, 512), (1472, 576)]
# img_scales=[(1600, 840), (1600, 900), (1600, 960), (1600, 1020),
#             (1600, 1080), (1600, 1140), (1600, 1200), (1600, 1260),
#             (1540, 840), (1480, 780), (1420, 720), (1380, 680),
#             (1660, 960), (1720, 1020), (1800, 1080), (1880, 1140)]

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/kitti/':
        's3://openmmlab/datasets/detection3d/kitti/',
        'data/kitti/':
        's3://openmmlab/datasets/detection3d/kitti/'
    }))

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='CenterFilter'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
    dict(
        type='Mono3DResizeV2',
        img_scale=img_scales,
        multiscale_mode='value',
        keep_ratio=False,
        with_aff=True),
    dict(type='CenterFilter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'centers2d', 'depths'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFileMono3D', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(1280, 384)],
        img_scale=[(1280, 384)],
        flip=False,
        transforms=[
            dict(
                type='Mono3DResizeV2',
                #  img_scale=[(1280, 384)],
                img_scale=[(1280, 384)],
                down_ratio=4,
                keep_ratio=False,
                with_aff=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D', file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_train_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_train.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val_mono3d.coco.json',
        info_file=data_root + 'kitti_infos_val.pkl',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera',
        vis_3d=True))
evaluation = dict(interval=2)
