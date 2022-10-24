_base_ = [
    '../_base_/datasets/kitti-mono3d-smoke-gams.py',
    '../_base_/models/smoke.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

# change `rescale_depth` to True during ms training
model = dict(bbox_head=dict(rescale_depth=True))

# optimizer
optimizer = dict(type='Adam', lr=2.5e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', warmup=None, step=[50])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=72)
log_config = dict(interval=10)

find_unused_parameters = True
