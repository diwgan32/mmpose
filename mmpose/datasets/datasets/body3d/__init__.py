# Copyright (c) OpenMMLab. All rights reserved.
from .body3d_h36m_dataset import Body3DH36MDataset
from .body3d_mpi_inf_3dhp_dataset import Body3DMpiInf3dhpDataset
from .body3d_semi_supervision_dataset import Body3DSemiSupervisionDataset
from .body3d_h36m_modified_dataset import Body3DH36MModifiedDataset
from .body3d_aist_dataset import Body3DAISTDataset
from .body3d_panoptic_dataset import Body3DPanopticDataset
from .body3d_combined_dataset import Body3DCombinedDataset

__all__ = [
    'Body3DH36MDataset', 'Body3DSemiSupervisionDataset',
    'Body3DMpiInf3dhpDataset', 'Body3DH36MModifiedDataset',
    'Body3DAISTDataset', "Body3DPanopticDataset",
    "Body3DCombinedDataset"
]
