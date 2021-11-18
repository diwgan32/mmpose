# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict, defaultdict
from pycocotools.coco import COCO

import mmcv
import numpy as np
from mmcv import Config

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.datasets.datasets.base import Kpt3dSviewKpt2dDataset
from ...builder import DATASETS


@DATASETS.register_module()
class Body3DPanopticDataset(Kpt3dSviewKpt2dDataset):
    """AIST dataset for 3D human pose estimation.


    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    PANOPTIC_JOINT_NAMES = [
        'Nose',
        'L_Eye',
        'R_Eye',
        'L_Ear',
        'R_Ear', \
        'L_Shoulder',
        'R_Shoulder',
        'L_Elbow',
        'R_Elbow', \
        'L_Wrist',
        'R_Wrist',
        'L_Hip',
        'R_Hip',
        'L_Knee', \
        'R_Knee', \
        'L_Ankle',
        'R_Ankle',
        "Pelvis"
    ]

    PANOPTIC_TO_H36M = [
        17, 12, 14, 16, 11, 13, 15, -1, -1, 0, -1, 5, 7, 9, 6, 8, 10
    ]

    H36M_SPINE_IDX = 7
    H36M_THORAX_IDX = 8
    H36M_HEAD_IDX = 10
    H36M_LSHOULDER_IDX = 11
    H36M_RSHOULDER_IDX = 14
    H36M_LHIP_IDX = 4
    H36M_RHIP_IDX = 1

    # 2D joint source options:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate by the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/h36m.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

    def _transform_coords(joint_cam):
        # SPINE is average of thorax and pelvis
        head = (joint_cam[1] + joint_cam[2] + joint_cam[3] + joint_cam[4])
        transformed_coords = joint_cam[AIST_TO_H36M]
        thorax = (transformed_coords[H36M_LSHOULDER_IDX] + transformed_coords[H36M_RSHOULDER_IDX])/2.0
        pelvis = (transformed_coords[H36M_LHIP_IDX] + transformed_coords[H36M_RHIP_IDX])/2.0

        spine = (thorax + pelvis)/2.0
        transformed_coords[H36M_SPINE_IDX] = spine
        transformed_coords[H36M_THORAX_IDX] = thorax
        transformed_coords[H36M_HEAD_IDX] = head

        return transformed_coords

    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # h36m specific attributes
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}". '
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(
                data_cfg['camera_param_file'])

        # h36m specific annotation info
        ann_info = {}
        ann_info['use_different_joint_weights'] = False
        # action filter
        actions = data_cfg.get('actions', '_all_')
        self.actions = set(
            actions if isinstance(actions, (list, tuple)) else [actions])

        # subject filter
        subjects = data_cfg.get('subjects', '_all_')
        self.subjects = set(
            subjects if isinstance(subjects, (list, tuple)) else [subjects])

        self.ann_info.update(ann_info)

    def load_annotations(self):
        """
            Reads AIST annotations, returns them in the following
            format: 

            data_info = {
                'imgnames': _imgnames,
                'joints_3d': _joints_3d,
                'joints_2d': _joints_2d,
                'scales': _scales,
                'centers': _centers,
            }
        """
        # get 2D joints

        db = COCO(self.train_annot_path)
        data_info = {
            'imgnames': [],
            "annotation_id": [],
            'joints_3d': [],
            'joints_2d': [],
            'scales': [],
            'centers': [],
        }
        for aid in db.anns.keys():
            ann = db.anns[aid]
            if (not db.imgs[ann['image_id']]["is_train"]):
                continue
            img = db.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']

            bbox = process_bbox(ann['bbox'], width, height)
            if bbox is None: continue

            # joints and vis
            f = np.array(db.imgs[ann['image_id']]["camera_param"]['focal'])
            c = np.array(db.imgs[ann['image_id']]["camera_param"]['princpt'])

            joint_cam = np.array(ann['joint_cam'])
            joint_cam = self._transform_coords(joint_cam)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_img[:,2] = joint_img[:,2] - joint_cam[self.root_idx,2]
            joint_vis = np.ones((self.joint_num,1))

            img_path = osp.join(self.img_dir, db.imgs[ann['image_id']]['file_name'])
            data_info["imgnames"].append(db.imgs[ann['image_id']]['file_name'])
            data_info["annotation_id"].append((db.imgs[ann['image_id']]['file_name'], db.imgs[ann['image_id']]["subject_id"]))
            data_info["joints_3d"].append(joint_cam)
            data_info["joints_2d"].append(joint_img[, :2])
            data_info["scales"].append(max(bbox[2]/200, bbox[3]/200))
            center = [bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0]
            data_info["centers"].append(center)
        
        data_info["joints_3d"] = np.array(data_info["joints_3d"])/1000
        data_info["joints_2d"] = np.array(data_info["joints_2d"])
        data_info["scales"] = np.array(data_info["scales"])
        data_info["centers"] = np.array(data_info["centers"])
        data_info["imgnames"] = np.array(data_info["imgnames"])

        return data_info

    @staticmethod
    def _parse_panoptic_imgname(annotation_id):
        """Parse imgname to get information of subject, action and camera.

        See here for name format: https://aistdancedb.ongaaccel.jp/data_formats/
        """
        imgname, subject = annotation_id
        parts = imgname.split("/")

        action = parts[0]
        camera = parts[1]
        return subject, action, camera

    def build_sample_indices(self):
        """Split original videos into sequences and build frame indices.

        This method overrides the default one in the base class.
        """

        # Group frames into videos. Assume that self.data_info is
        # chronological.
        video_frames = defaultdict(list)
        for idx, annot in enumerate(self.data_info['annotation_id']):
            subj, action, camera = self._parse_panoptic_imgname(annot)

            # TODO: Is _all_ going to be in self.actions and self.subjects?
            if '_all_' not in self.actions and action not in self.actions:
                continue

            if '_all_' not in self.subjects and subj not in self.subjects:
                continue

            video_frames[(subj, action, camera)].append(idx)

        # build sample indices
        sample_indices = []
        _len = (self.seq_len - 1) * self.seq_frame_interval + 1
        _step = self.seq_frame_interval
        for _, _indices in sorted(video_frames.items()):
            n_frame = len(_indices)

            if self.temporal_padding:
                # Pad the sequence so that every frame in the sequence will be
                # predicted.
                if self.causal:
                    frames_left = self.seq_len - 1
                    frames_right = 0
                else:
                    frames_left = (self.seq_len - 1) // 2
                    frames_right = frames_left
                for i in range(n_frame):
                    pad_left = max(0, frames_left - i // _step)
                    pad_right = max(0,
                                    frames_right - (n_frame - 1 - i) // _step)
                    start = max(i % _step, i - frames_left * _step)
                    end = min(n_frame - (n_frame - 1 - i) % _step,
                              i + frames_right * _step + 1)
                    sample_indices.append([_indices[0]] * pad_left +
                                          _indices[start:end:_step] +
                                          [_indices[-1]] * pad_right)
            else:
                seqs_from_video = [
                    _indices[i:(i + _len):_step]
                    for i in range(0, n_frame - _len + 1)
                ]
                sample_indices.extend(seqs_from_video)

        # reduce dataset size if self.subset < 1
        assert 0 < self.subset <= 1
        subset_size = int(len(sample_indices) * self.subset)
        start = np.random.randint(0, len(sample_indices) - subset_size + 1)
        end = start + subset_size

        return sample_indices[start:end]

    def _load_joint_2d_detection(self, det_file):
        """"Load 2D joint detection results from file."""
        joints_2d = np.load(det_file).astype(np.float32)

        return joints_2d

    def evaluate(self,
                 outputs,
                 res_folder,
                 metric='mpjpe',
                 logger=None,
                 **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        res_file = osp.join(res_folder, 'result_keypoints.json')
        kpts = []
        for output in outputs:
            preds = output['preds']
            image_paths = output['target_image_paths']
            batch_size = len(image_paths)
            for i in range(batch_size):
                target_id = self.name2id[image_paths[i]]
                kpts.append({
                    'keypoints': preds[i],
                    'target_id': target_id,
                })

        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(kpts)
            elif _metric == 'p-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='p-mpjpe')
            elif _metric == 'n-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='n-mpjpe')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        return OrderedDict(name_value_tuples)

    def _report_mpjpe(self, keypoint_results, mode='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants like
        P-MPJPE or N-MPJPE.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DH36MDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:
                - ``'mpjpe'``: Standard MPJPE.
                - ``'p-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
                - ``'n-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    in scale only.
        """

        preds = []
        gts = []
        masks = []
        action_category_indices = defaultdict(list)
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            target_id = result['target_id']
            gt, gt_visible = np.split(
                self.data_info['joints_3d'][target_id], [3], axis=-1)
            preds.append(pred)
            gts.append(gt)
            masks.append(gt_visible)

            action = self._parse_h36m_imgname(
                self.data_info['imgnames'][target_id])[1]
            action_category = action.split('_')[0]
            action_category_indices[action_category].append(idx)

        preds = np.stack(preds)
        gts = np.stack(gts)
        masks = np.stack(masks).squeeze(-1) > 0

        err_name = mode.upper()
        if mode == 'mpjpe':
            alignment = 'none'
        elif mode == 'p-mpjpe':
            alignment = 'procrustes'
        elif mode == 'n-mpjpe':
            alignment = 'scale'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        error = keypoint_mpjpe(preds, gts, masks, alignment)
        name_value_tuples = [(err_name, error)]

        for action_category, indices in action_category_indices.items():
            _error = keypoint_mpjpe(preds[indices], gts[indices],
                                    masks[indices])
            name_value_tuples.append((f'{err_name}_{action_category}', _error))

        return name_value_tuples

    def _load_camera_param(self, camera_param_file):
        """Load camera parameters from file."""
        return mmcv.load(camera_param_file)

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        subj, _, camera = self._parse_h36m_imgname(imgname)
        return self.camera_param[(subj, camera)]