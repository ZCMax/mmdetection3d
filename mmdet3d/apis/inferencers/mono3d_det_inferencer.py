# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import mmengine
import numpy as np
import torch.nn as nn
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import ConfigType, register_all_modules
from mmengine.dataset import Compose
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

from ..evaluation import get_classes

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


class Mono3DDetInferencer(BaseInferencer):
    """MMDet inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "yolox-s" or "configs/yolox/yolox_s_8xb8-300e_coco.py".
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'pred_score_thr',
        'img_out_dir'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample'
    }

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmdet',
                 palette: str = 'none') -> None:
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0
        self.palette = palette
        register_all_modules()
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

    def _init_model(
        self,
        cfg: ConfigType,
        weights: str,
        device: str = 'cpu',
    ) -> nn.Module:
        if 'init_cfg' in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
        model = MODELS.build(cfg.model)

        checkpoint = load_checkpoint(model, weights, map_location='cpu')
        checkpoint_meta = checkpoint.get('meta', {})
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint_meta:
            # mmdet 3.x, all keys should be lowercase
            model.dataset_meta = {
                k.lower(): v
                for k, v in checkpoint_meta['dataset_meta'].items()
            }
        elif 'CLASSES' in checkpoint_meta:
            # < mmdet 3.x
            classes = checkpoint_meta['CLASSES']
            model.dataset_meta = {'classes': classes}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

        # Priority:  args.palette -> config -> checkpoint
        if self.palette != 'none':
            model.dataset_meta['palette'] = self.palette
        else:
            test_dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
            # lazy init. We only need the metainfo.
            test_dataset_cfg['lazy_init'] = True
            metainfo = DATASETS.build(test_dataset_cfg).metainfo
            cfg_palette = metainfo.get('palette', None)
            if cfg_palette is not None:
                model.dataset_meta['palette'] = cfg_palette
            else:
                if 'palette' not in model.dataset_meta:
                    warnings.warn(
                        'palette does not exist, random is used by default. '
                        'You can also set the palette to customize.')
                    model.dataset_meta['palette'] = 'random'

        model.cfg = cfg  # save the config in the model for convenience
        model.to(device)
        model.eval()
        return model

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if 'meta_keys' in pipeline_cfg[-1]:
            pipeline_cfg[-1]['meta_keys'] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
                if meta_key != 'img_id')

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFileMono3D')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFileMono3D is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'Mono3DInferencerLoader'
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        visualizer = super()._init_visualizer(cfg)
        visualizer.dataset_meta = self.model.dataset_meta
        return visualizer

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 draw_pred: bool = True,
                 pred_score_thr: float = 0.3,
                 img_out_dir: str = '',
                 print_result: bool = False,
                 pred_out_file: str = '',
                 **kwargs) -> dict:
        """Call the inferencer.
        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.
            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.
        Returns:
            dict: Inference and visualization results.
        """
        return super().__call__(
            inputs,
            return_datasamples,
            batch_size,
            return_vis=return_vis,
            show=show,
            wait_time=wait_time,
            draw_pred=draw_pred,
            pred_score_thr=pred_score_thr,
            img_out_dir=img_out_dir,
            print_result=print_result,
            pred_out_file=pred_out_file,
            **kwargs)

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  img_out_dir: str = '') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[Dict]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.
        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if self.visualizer is None or (not show and img_out_dir == ''
                                       and not return_vis):
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != '' \
                else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(img)
            self.num_visualized_imgs += 1

        return results

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        pred_out_file: str = '',
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.
        This method should be responsible for the following tasks:
        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.
        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_file: File to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.
        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.
            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasample=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        result_dict = {}
        results = preds
        if not return_datasample:
            results = []
            for pred in preds:
                result = self.pred2dict(pred)
                results.append(result)
        # Add img to the results after printing and dumping
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        if pred_out_file != '':
            mmengine.dump(result_dict, pred_out_file)
        result_dict['visualization'] = visualization
        return result_dict

    def pred2dict(self, data_sample: InstanceData) -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.
        """
        pred_instances = data_sample.pred_instances.numpy()
        result = {
            'bboxes': pred_instances.bboxes.tolist(),
            'labels': pred_instances.labels.tolist(),
            'scores': pred_instances.scores.tolist()
        }

        return result
