from collections import defaultdict
from typing import Mapping, Dict, Any
from pathlib import Path
import json
import numpy as np


def NestedDict():
    return defaultdict(NestedDict)


def nested_dict_update(d: Mapping, u: Mapping) -> Mapping:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def pose_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    assert R.shape == (3, 3)
    assert t.size == 3
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.reshape(3)
    return pose


def load_targets(targets_path: Path) -> NestedDict:
    targets_raw = json.load(targets_path.open())
    targets = NestedDict()
    for t in targets_raw:
        targets[t["scene_id"]][t["im_id"]][t["obj_id"]] = t
    return targets


def load_estimates(estimates_path: Path, prefix="", t_unit="mm") -> NestedDict:
    """

    :param estimates_path: Path to the .csv file containing the pose estimates.
    :param prefix: Optional prefix to prepend to the score, R, t, and time keys.
    :return: NestedDict mapping (scene_id, im_id, obj_id, est_id) to a pose estimate.
    """
    assert t_unit in ["mm", "cm", "m"], f"Error: Unknown unit {t_unit} for translation!"
    t_scale = 1.0
    if t_unit == "cm":
        t_scale = 10.0
    elif t_unit == "m":
        t_scale = 1000.0
    with estimates_path.open() as f:
        estimates_raw = f.readlines()
    estimates = NestedDict()
    header = "scene_id,im_id,obj_id,score,R,t,time"
    for line in estimates_raw:
        if header == line.strip():
            continue
        items = line.split(",")
        if len(items) != 7:
            raise ValueError("A line does not have 7 comma-sep. elements: {}".format(line))
        e = {
            f"scene_id": int(items[0]),
            f"im_id": int(items[1]),
            f"obj_id": int(items[2]),
            f"{prefix}score": float(items[3]),
            f"{prefix}R": np.array(list(map(float, items[4].split())), np.float64).reshape((3, 3)),
            f"{prefix}t": np.array(list(map(float, items[5].split())), np.float64).reshape((3, 1))
            * t_scale,
            f"{prefix}time": float(items[6]),
        }
        est_id = len(estimates[e["scene_id"]][e["im_id"]].get(e["obj_id"], []))
        estimates[e["scene_id"]][e["im_id"]][e["obj_id"]][est_id] = e
    return estimates


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order
