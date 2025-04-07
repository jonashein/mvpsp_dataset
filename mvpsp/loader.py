from collections import defaultdict
from typing import Sequence

import numpy as np

from .loader_aux import DatasetSampleAux
from .dataset import MVPSP, MvpspSingleviewDataset, MvpspMultiviewDataset
from .utils import pose_from_Rt

try:
    import torch

    dataset_base_class = torch.utils.data.Dataset
except ImportError:
    dataset_base_class = object


class SingleViewObjectLoader(dataset_base_class):
    def __init__(
        self,
        dataset: MvpspSingleviewDataset,
        obj_ids: Sequence[int],
        auxs: Sequence[DatasetSampleAux],
    ):
        self.dataset = dataset
        self.auxs = auxs

        # remove not requested annotations
        self.dataset.apply_filter(
            require_rgb=True,
            require_depth=True,
            require_objs={MVPSP.ANY_INSTANCE},
            discard_objs={obj_id for obj_id in MVPSP.OBJECTS if obj_id not in obj_ids},
            discard_hands=True,
            discard_gaze=True,
        )

        # create index of all object instances
        self.index = []
        for sample_idx in range(len(self.dataset)):
            for obj_idx, _ in enumerate(self.dataset[sample_idx].get("objects", [])):
                self.index.append((sample_idx, obj_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        sample_idx, obj_idx = self.index[index]
        item = self.dataset[sample_idx].copy()
        item.update(item["objects"][obj_idx])
        item.pop("objects", None)
        for aux in self.auxs:
            item = aux(item, self)
        return item


class MultiViewObjectLoader(dataset_base_class):
    def __init__(
        self,
        dataset: MvpspMultiviewDataset,
        obj_ids: Sequence[int],
        auxs: Sequence[DatasetSampleAux],
    ):
        self.dataset = dataset
        self.auxs = auxs

        # remove not requested annotations
        self.dataset.apply_filter(
            require_rgb=True,
            require_depth=True,
            require_objs={MVPSP.ANY_INSTANCE},
            discard_objs={obj_id for obj_id in MVPSP.OBJECTS if obj_id not in obj_ids},
            discard_hands=True,
            discard_gaze=True,
        )

        # create index of all object instances
        self.index = []
        inst_equality_thres = 1.0  # instances of same class are considered to be equal iff position differs by less than 1mm
        for sample_idx, sample in enumerate(self.dataset):
            # get all object instance ids. each instance is stored as a tuple (world_position, {frame_idx: obj_idx})
            obj_insts = defaultdict(list)
            for frame_idx, frame in enumerate(sample):
                cam_c2w = np.linalg.inv(pose_from_Rt(frame["cam_R_w2c"], frame["cam_t_w2c"]))
                for obj_idx, obj in enumerate(frame.get("objects", [])):
                    obj_id = obj["obj_id"]
                    cam_m2c = pose_from_Rt(obj["cam_R_m2c"], obj["cam_t_m2c"])
                    obj_t_m2w = (cam_c2w @ cam_m2c)[:3, 3:]

                    # check if instance was seen already
                    new_instance = True
                    for inst_t_m2w, obj_idx_per_frame_idx in obj_insts.get(obj_id, []):
                        if np.linalg.norm(obj_t_m2w - inst_t_m2w) < inst_equality_thres:
                            obj_idx_per_frame_idx[frame_idx] = obj_idx
                            new_instance = False
                            break
                    if new_instance:
                        obj_insts[obj_id].append((obj_t_m2w, {frame_idx: obj_idx}))
            # add each unique object instance to the index
            for _, obj_insts in obj_insts.items():
                for _, obj_idx_per_frame_idx in obj_insts:
                    self.index.append((sample_idx, obj_idx_per_frame_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        sample_idx, obj_idx_per_frame_idx = self.index[index]
        # create copy of sample, keep only the selected object instance
        sample = [frame.copy() for frame in self.dataset[sample_idx]]
        for frame_idx, frame in enumerate(sample):
            obj_idx = obj_idx_per_frame_idx.get(frame_idx, None)
            if obj_idx is not None:
                frame.update(frame["objects"][obj_idx])
            frame.pop("objects", None)
        # apply auxiliary transforms
        for aux in self.auxs:
            sample = aux(sample, self)
        return sample
