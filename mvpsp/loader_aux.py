import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Set, Union, Tuple
import numpy as np
import cv2

from mvpsp.utils import pose_from_Rt, rle_to_mask

try:
    import torch

    dataset_base_class = torch.utils.data.Dataset
except ImportError:
    dataset_base_class = object


class DatasetSampleAux(object):
    def __init__(self):
        pass

    def __call__(self, sample: dict, dataset: dataset_base_class) -> dict:
        # ensure that the sample is a list of dicts. temporarily wrap single dicts in list.
        single_frame = isinstance(sample, dict)
        if single_frame:
            sample = [sample]
        sample = self.apply(sample, dataset)
        if single_frame:
            sample = sample[0]
        return sample

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        raise NotImplementedError("Method needs to be implemented in subclass!")


class RgbLoader(DatasetSampleAux):
    """
    Loads an RGB image.

    Input keys: rgb_path
    Output keys: rgb
    """

    def __init__(
        self, copy: bool = False, raise_error: bool = True, default_value: np.array = None
    ):
        super().__init__()
        self.copy = copy
        self.raise_error = raise_error
        self.default_value = default_value

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            try:
                fp = frame["rgb_path"]
                rgb = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                assert rgb is not None, f"Error reading image at {fp}"
                # BGR to RGB
                rgb = rgb[..., ::-1]
                frame["rgb"] = rgb.copy() if self.copy else rgb
            except Exception as e:
                if self.raise_error:
                    raise e
                else:
                    frame["rgb"] = (
                        self.default_value.copy()
                        if self.copy and hasattr(self.default_value, "copy")
                        else self.default_value
                    )
        return sample


class DepthLoader(DatasetSampleAux):
    """
    Loads a depth image.

    Input keys: depth_path
    Output keys: depth
    """

    def __init__(
        self, copy: bool = False, raise_error: bool = True, default_value: np.array = None
    ):
        super().__init__()
        self.copy = copy
        self.raise_error = raise_error
        self.default_value = default_value

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            try:
                fp = frame["depth_path"]
                depth = cv2.imread(str(fp), cv2.IMREAD_ANYDEPTH)
                assert depth is not None, f"Error reading image at {fp}"
                depth = frame.get("depth_scale", 1.0) * depth.astype(float)
                frame["depth"] = depth.copy() if self.copy else depth
            except Exception as e:
                if self.raise_error:
                    raise e
                else:
                    frame["depth"] = (
                        self.default_value.copy()
                        if self.copy and hasattr(self.default_value, "copy")
                        else self.default_value
                    )
        return sample


class ObjectMaskLoader(DatasetSampleAux):
    """
    Input keys: obj_mask_path or obj_mask_visib_path, depending on mask_type
    Output keys: mask or mask_visib, depending on mask_type
    """

    def __init__(
        self,
        mask_type: str = "mask_visib",
        copy: bool = False,
        raise_error: bool = True,
        default_value: np.array = None,
    ):
        super().__init__()
        self.mask_type = mask_type
        self.mask_key = f"obj_{mask_type}_path"
        self.copy = copy
        self.raise_error = raise_error
        self.default_value = default_value

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            try:
                fp = frame[self.mask_key]
                mask = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                assert mask is not None, f"Error reading image at {fp}"
                frame[self.mask_type] = mask.copy() if self.copy else mask
            except Exception as e:
                if self.raise_error:
                    raise e
                else:
                    frame[self.mask_type] = (
                        self.default_value.copy()
                        if self.copy and hasattr(self.default_value, "copy")
                        else self.default_value
                    )
        return sample


class ObjectBboxCrop(DatasetSampleAux):
    """
    Input keys: obj_bbox, obj_bbox_visib, cam_K
    Output keys: input keys with optional suffix defined by crop_suffix
    """

    def __init__(
        self,
        crop_res: int,
        im_keys: Set[str],
        crop_scale: float = 1.0,
        crop_suffix: str = "_crop",
        max_bbox_offset_scale: float = 0.05,
        max_inplane_rotation: float = 0.0,
        raise_error: bool = True,
    ):
        super().__init__()
        self.crop_res = crop_res
        self.crop_scale = crop_scale
        self.im_keys = im_keys
        self.crop_suffix = crop_suffix
        self.max_bbox_offset_scale = max_bbox_offset_scale
        self.max_inplane_rotation = max_inplane_rotation
        self.raise_error = raise_error

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            try:
                left, top, width, height = frame["obj_bbox"]
                theta = np.random.uniform(-self.max_inplane_rotation, self.max_inplane_rotation)
                S, C = np.sin(theta), np.cos(theta)
                R = np.array([[C, -S], [S, C]])
                cy, cx = top + height / 2, left + width / 2

                size = self.crop_res / max(width, height, 1) / self.crop_scale
                size = size * np.random.uniform(
                    1 - self.max_bbox_offset_scale, 1 + self.max_bbox_offset_scale
                )
                r = self.crop_res
                M = np.concatenate((R, [[-cx], [-cy]]), axis=1) * size
                M[:, 2] += r / 2

                offset = (r - r / self.crop_scale) / 2.0 * self.max_bbox_offset_scale
                M[:, 2] += np.random.uniform(-offset, offset, 2)
                Ms = np.concatenate((M, [[0, 0, 1]]))

                frame[f"M{self.crop_suffix}"] = M
                frame[f"cam_K{self.crop_suffix}"] = Ms @ frame["cam_K"]
            except Exception as e:
                if self.raise_error:
                    raise e
                continue

            for key in self.im_keys:
                try:
                    im = frame[key]
                    interp = (
                        cv2.INTER_LINEAR
                        if im.ndim == 2
                        else np.random.choice(
                            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
                        )
                    )
                    frame[f"{key}{self.crop_suffix}"] = cv2.warpAffine(
                        im, M, (self.crop_res, self.crop_res), flags=interp
                    )
                except Exception as e:
                    if self.raise_error:
                        raise e
        return sample


class DefaultBboxAnnotation(DatasetSampleAux):

    def __init__(self):
        super().__init__()

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            # missing annotation implies that the object is outside the field of view
            if "obj_bbox" not in frame:
                frame["obj_bbox"] = np.array([-1, -1, -1, -1])
            if "obj_bbox_visib" not in frame:
                frame["obj_bbox_visib"] = np.array([-1, -1, -1, -1])
            if "px_count_valid" not in frame:
                frame["px_count_valid"] = -1
            if "px_count_visib" not in frame:
                frame["px_count_visib"] = -1
            if "visib_fract" not in frame:
                frame["visib_fract"] = -1.0
        return sample


class ComputeOOVObjectAnnotation(DatasetSampleAux):
    """
    Compute annotations for out-of-view objects based on their annotations in other cameras.

    Input keys: cam_R_w2c, cam_t_w2c, cam_K, obj_bbox, obj_bbox_visib, cam_K
    Output keys: obj_id, inst_id, cam_R_m2c, cam_t_m2c, obj_bbox, obj_bbox_visib, px_count_valid, px_count_visib, visib_fract
    """

    def __init__(self):
        super().__init__()

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        annot_frame = None
        unannotated_frames = []
        for frame in sample:
            if "obj_id" not in frame:
                unannotated_frames.append(frame)
            elif annot_frame is None:
                annot_frame = frame

        annot_frame_c2w = np.linalg.inv(
            pose_from_Rt(annot_frame["cam_R_w2c"], annot_frame["cam_t_w2c"])
        )
        obj_m2w = annot_frame_c2w @ pose_from_Rt(annot_frame["cam_R_m2c"], annot_frame["cam_t_m2c"])
        for frame in unannotated_frames:
            frame["obj_id"] = annot_frame["obj_id"]
            frame["inst_id"] = annot_frame["inst_id"]
            cam_m2c = pose_from_Rt(frame["cam_R_w2c"], frame["cam_t_w2c"]) @ obj_m2w
            frame["cam_R_m2c"] = cam_m2c[:3, :3]
            frame["cam_t_m2c"] = cam_m2c[:3, 3:]

            # missing annotation implies that the object is outside the field of view
            frame["obj_bbox"] = np.array([-1, -1, -1, -1])
            frame["obj_bbox_visib"] = np.array([-1, -1, -1, -1])
            frame["px_count_valid"] = 0
            frame["px_count_visib"] = 0
            frame["visib_fract"] = 0.0
        return sample


class ViewSelector(DatasetSampleAux):
    """
    Selects the first frame with the given camera id from a multi-frame sample.
    """

    def __init__(self, cam_id):
        super().__init__()
        self.cam_id = cam_id

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            if frame["cam_id"] == self.cam_id:
                return [frame]
        return []


class CNOSMaskLoader(DatasetSampleAux):
    """
    Input keys: scene_id, im_id, obj_id
    Output keys: obj_bbox, <mask_type>, <mask_type>_score, <mask_type>_time
    """

    def __init__(
        self,
        mask_json_paths: Union[Set[Union[Path, str]], Path, str],
        conf_threshold: float = 0.5,
        mask_type: str = "mask",
        copy: bool = False,
        raise_error: bool = True,
        default_value: np.array = None,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.mask_type = mask_type
        self.copy = copy
        self.raise_error = raise_error
        self.default_value = default_value

        if not isinstance(mask_json_paths, Set):
            mask_json_paths = {mask_json_paths}

        # index masks by scene_id, img_id, obj_id
        self.detections = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for p in mask_json_paths:
            if not isinstance(p, Path):
                p = Path(p)
            with p.open() as f:
                dets = json.load(f)
            for det in dets:
                if float(det["score"]) >= self.conf_threshold:
                    scene_id = int(det["scene_id"])
                    img_id = int(det["image_id"])
                    obj_id = int(det["category_id"])
                    frame_dets = self.detections[scene_id][img_id][obj_id]
                    assert (
                        len(frame_dets) == 0
                    ), "Multiple detections of the same class in the same image not supported yet."
                    frame_dets.append(det)

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            try:
                det = self.detections[frame["scene_id"]][frame["im_id"]][frame["obj_id"]][0]
                frame[self.mask_type] = rle_to_mask(det["segmentation"]).astype(np.uint8)
                bbox = np.array(det["bbox"], dtype=int)  # in xyxy format
                bbox[2:] = bbox[2:] - bbox[:2]  # convert xyxy to xywh format
                frame["obj_bbox"] = bbox
                frame[f"{self.mask_type}_score"] = float(det["score"])
                frame[f"{self.mask_type}_time"] = float(det["time"])
            except Exception as e:
                if self.raise_error:
                    raise e
                else:
                    frame[self.mask_type] = (
                        self.default_value.copy()
                        if self.copy and hasattr(self.default_value, "copy")
                        else self.default_value
                    )
        return sample


class Resize(DatasetSampleAux):
    def __init__(
        self,
        im_keys: Set[str],
        out_res: Union[int, Tuple[int, int]],
        keep_aspect_ratio: bool = True,
        out_suffix: str = "_resize",
    ):
        super().__init__()
        if isinstance(out_res, int):
            out_res = (out_res, out_res)
        self.im_keys = im_keys
        self.out_res = out_res
        self.keep_aspect_ratio = keep_aspect_ratio
        self.out_suffix = out_suffix

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        M = None
        for frame in sample:
            for key in self.im_keys:
                if M is None:
                    in_res = frame[key].shape[:2]
                    M = np.zeros((2, 3))
                    scale_x = self.out_res[0] / in_res[1]
                    scale_y = self.out_res[1] / in_res[0]
                    if self.keep_aspect_ratio:
                        scale = min(scale_x, scale_y)
                        M[0, 0] = scale
                        M[1, 1] = scale
                        M[0, 2] = (self.out_res[0] - scale * in_res[1]) / 2
                        M[1, 2] = (self.out_res[1] - scale * in_res[0]) / 2
                    else:
                        M[0, 0] = scale_x
                        M[1, 1] = scale_y
                    Ms = np.concatenate((M, [[0, 0, 1]]))

                im = frame[key]
                interp = (
                    cv2.INTER_LINEAR
                    if im.ndim == 2
                    else np.random.choice(
                        [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
                    )
                )
                frame[f"{key}{self.out_suffix}"] = cv2.warpAffine(im, M, self.out_res, flags=interp)
                frame[f"M{self.out_suffix}"] = M
                frame[f"cam_K{self.out_suffix}"] = Ms @ frame["cam_K"]
        return sample


class FilterKeys(DatasetSampleAux):
    def __init__(
        self,
        keep_keys: Set[str] = None,
        discard_keys: Set[str] = None,
    ):
        super().__init__()
        if keep_keys is not None and len(keep_keys) == 0:
            keep_keys = None
        if discard_keys is not None and len(discard_keys) == 0:
            discard_keys = None
        if keep_keys is None and discard_keys is None:
            print("No keys to filter!")
        elif keep_keys is not None and discard_keys is not None:
            print(
                f"Cannot define both keep_keys and discard_keys. Keeping only keys in keep_keys while discarding all others."
            )
            self.discard_keys = None
        self.keep_keys = keep_keys
        self.discard_keys = discard_keys

    def apply(self, sample: Sequence[dict], dataset: dataset_base_class) -> Sequence[dict]:
        for frame in sample:
            if self.keep_keys is not None:
                for key in frame.keys():
                    if key not in self.keep_keys:
                        frame.pop(key)
            elif self.discard_keys is not None:
                for key in self.discard_keys:
                    frame.pop(key, None)
        return sample
