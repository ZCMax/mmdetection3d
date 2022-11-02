_base_ = './pgd_r101-caffe_fpn_head-gn_2xb3-4x_kitti-mono3d.py'

# model settings
model = dict(
    backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage3', 'stage4', 'stage5')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 768, 1024],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True))

# load pretrained model
load_from = 'ckpts/dd3d_det_final.pth'
