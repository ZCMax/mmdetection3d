import os

import mmcv
import mmengine
import numpy as np

from mmdet3d.structures import points_cam2img


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    points,
    img,
    lidar2img,
    min_dist: float = 0.0,
):
    projected_points = points_cam2img(points, lidar2img, with_depth=True)
    mask = np.ones(projected_points.shape[0], dtype=bool)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.logical_and(mask, projected_points[:, 2] > min_dist)
    mask = np.logical_and(mask, projected_points[:, 0] > 1)
    mask = np.logical_and(mask, projected_points[:, 0] < img.shape[1] - 1)
    mask = np.logical_and(mask, projected_points[:, 1] > 1)
    mask = np.logical_and(mask, projected_points[:, 1] < img.shape[0] - 1)
    # ()
    projected_points = projected_points[mask]

    return projected_points


data_root = 'data/nuscenes/samples'
train_info_path = 'nuscenes_v2/nuscenes_infos_train.pkl'
val_info_path = 'nuscenes_v2/nuscenes_infos_val.pkl'
save_dir = '/mnt/petrelfs/share_data/zhuchenming/datasets_1.1/nuscenes'

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]


def get_lidar2img(cam2img, lidar2cam):
    """Get the projection matrix of lidar2img.

    Args:
        cam2img (torch.Tensor): A 3x3 or 4x4 projection matrix.
        lidar2cam (torch.Tensor): A 3x3 or 4x4 projection matrix.

    Returns:
        torch.Tensor: transformation matrix with shape 4x4.
    """
    if cam2img.shape == (3, 3):
        temp = np.eye(4)
        temp[:3, :3] = cam2img
        cam2img = temp

    if lidar2cam.shape == (3, 3):
        temp = np.eye(4)
        temp[:3, :3] = lidar2cam
        lidar2cam = temp
    return np.matmul(cam2img, lidar2cam)


def worker(info):
    lidar_path = os.path.join(data_root, 'LIDAR_TOP',
                              info['lidar_points']['lidar_path'])
    points = np.fromfile(
        lidar_path, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]

    for i, cam_key in enumerate(cam_keys):
        lidar2cam = np.array(info['images'][cam_key]['lidar2cam'])
        cam2img = np.array(info['images'][cam_key]['cam2img'])
        lidar2img = get_lidar2img(cam2img, lidar2cam)
        img = mmcv.imread(
            os.path.join(data_root, cam_key,
                         info['images'][cam_key]['img_path']))
        pts_img = map_pointcloud_to_image(points.copy(), img, lidar2img)
        depth_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        iy = np.round(pts_img[:, 1]).astype(np.int64)
        ix = np.round(pts_img[:, 0]).astype(np.int64)
        depth_img[iy, ix] = pts_img[:, 2]
        file_name = info['images'][cam_key]['img_path']
        mmcv.imwrite(depth_img,
                     os.path.join(save_dir, 'depth_maps', cam_key, file_name))
        # (N, 3)
        pts_img.astype(np.float32).flatten().tofile(
            os.path.join(save_dir, 'depth_points', cam_key,
                         f'{file_name}.bin'))
    # plt.savefig(f"{sample_idx}")


if __name__ == '__main__':
    # po = Pool(24)
    # mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_gt', 'train'))
    # train_infos = mmcv.load(train_info_path)
    # # import ipdb; ipdb.set_trace()
    # for info in train_infos:
    #     po.apply_async(func=worker, args=(info, 'train'))
    # po.close()
    # po.join()
    # po = Pool(24)
    mmengine.mkdir_or_exist(os.path.join(save_dir, 'depth_maps'))
    mmengine.mkdir_or_exist(os.path.join(save_dir, 'depth_points'))
    for cam_key in cam_keys:
        mmengine.mkdir_or_exist(os.path.join(save_dir, 'depth_maps', cam_key))
        mmengine.mkdir_or_exist(
            os.path.join(save_dir, 'depth_points', cam_key))
    train_infos = mmengine.load(train_info_path)['data_list']
    # import ipdb; ipdb.set_trace()
    # for info in val_infos:
    #     po.apply_async(func=worker, args=(info, 'val'))
    # po.close()
    # po.join()
    mmengine.track_parallel_progress(worker, train_infos, 24)  # 24 workers
    # mmengine.mkdir_or_exist(os.path.join(save_dir, 'depth_gt', 'val'))
    val_infos = mmengine.load(val_info_path)['data_list']
    # # import ipdb; ipdb.set_trace()
    # # for info in val_infos:
    # #     po.apply_async(func=worker, args=(info, 'val'))
    # # po.close()
    # # po.join()
    mmengine.track_parallel_progress(worker, val_infos, 24)  # 24 workers
