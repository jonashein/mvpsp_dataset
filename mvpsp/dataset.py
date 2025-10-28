from typing import (
    Union,
    Callable,
    Sequence,
    Dict,
    List,
    Container,
    Set,
)
from pathlib import Path
import json
import numpy as np
import trimesh
from .utils import NestedDict, nested_dict_update
from multiprocessing import Pool, cpu_count
from itertools import chain


class MVPSP(object):
    TRAIN_SUBSET_WETLAB = "train_wetlab"
    TRAIN_SUBSET_SYNTH = "train_synth"
    TRAIN_SUBSET_PBR = "train_pbr"
    TRAIN_SUBSETS = [TRAIN_SUBSET_WETLAB, TRAIN_SUBSET_SYNTH, TRAIN_SUBSET_PBR]
    TEST_SUBSET_WETLAB = "test_wetlab"
    TEST_SUBSET_ORX = "test_orx"
    TEST_SUBSETS = [TEST_SUBSET_WETLAB, TEST_SUBSET_ORX]
    CAMERA_LEFT = 0
    CAMERA_OPPOSITE_LEFT = 1
    CAMERA_OPPOSITE_RIGHT = 2
    CAMERA_RIGHT = 3
    CAMERA_CEILING = 4
    CAMERA_SURGEON = 5
    CAMERA_ASSISTANT = 6
    CAMERA_FAR = 7
    CAMERA_NAMES = {
        CAMERA_LEFT: "Left",
        CAMERA_OPPOSITE_LEFT: "Opposite Left",
        CAMERA_OPPOSITE_RIGHT: "Opposite Right",
        CAMERA_RIGHT: "Right",
        CAMERA_CEILING: "Ceiling",
        CAMERA_SURGEON: "Surgeon",
        CAMERA_ASSISTANT: "Assistant",
        CAMERA_FAR: "Far",
    }
    CAMERAS = list(CAMERA_NAMES.keys())
    CAMERAS_WETLAB = [
        CAMERA_LEFT,
        CAMERA_OPPOSITE_LEFT,
        CAMERA_OPPOSITE_RIGHT,
        CAMERA_RIGHT,
        CAMERA_CEILING,
        CAMERA_SURGEON,
        CAMERA_ASSISTANT,
    ]
    CAMERAS_ORX = [
        CAMERA_LEFT,
        CAMERA_OPPOSITE_LEFT,
        CAMERA_OPPOSITE_RIGHT,
        CAMERA_RIGHT,
        CAMERA_FAR,
    ]
    STAFF_SURGEON = 0
    STAFF_ASSISTANT = 1
    STAFF_NAMES = {
        STAFF_SURGEON: "Surgeon",
        STAFF_ASSISTANT: "Assistant",
    }
    STAFF = list(STAFF_NAMES.keys())
    HANDEDNESS_LEFT = 0
    HANDEDNESS_RIGHT = 1
    HANDEDNESS_NAMES = {
        HANDEDNESS_LEFT: "Left",
        HANDEDNESS_RIGHT: "Right",
    }
    HANDEDNESS = list(HANDEDNESS_NAMES.keys())
    OBJECT_DRILL = 1
    OBJECT_SCREWDRIVER = 2
    OBJECT_HOLOLENS = 3
    OBJECT_SPECIMEN_1 = 4
    OBJECT_SPECIMEN_2 = 5
    OBJECT_SPECIMEN_3 = 6
    OBJECT_NAMES = {
        OBJECT_DRILL: "Drill",
        OBJECT_SCREWDRIVER: "Screwdriver",
        OBJECT_HOLOLENS: "HoloLens",
        OBJECT_SPECIMEN_1: "Specimen1",
        OBJECT_SPECIMEN_2: "Specimen2",
        OBJECT_SPECIMEN_3: "Specimen3",
    }
    OBJECTS = list(OBJECT_NAMES.keys())
    SYMMETRIC_OBJECTS = [OBJECT_SCREWDRIVER]
    ANY_INSTANCE = -99
    # Hand joint information is in https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter
    HAND_CONNECTIVITY = [
        # [0, 1],
        # Thumb
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        # Index
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        # Middle
        [1, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],
        # Ring
        [1, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20],
        # Pinky
        [1, 21],
        [21, 22],
        [22, 23],
        [23, 24],
        [24, 25],
    ]


def _verify_require_discard(
    require_ids: Set[int] = None,
    discard_ids: Union[bool, Set[int]] = False,
    error_msg_fstr: str = None,
):
    if isinstance(discard_ids, set) and require_ids is not None:
        shared_invalid = require_ids & discard_ids
        if len(shared_invalid) > 0 and error_msg_fstr is not None:
            print(error_msg_fstr.format(ids=", ".join(map(str, shared_invalid))))
            discard_ids.difference_update(shared_invalid)


def _process_instance_request(
    instances: List[Dict],
    id_key,
    id_choices: Container,
    require_ids: Set[int] = None,
    discard_ids: Union[bool, Set[int]] = False,
    discard_unknown_ids: bool = True,
    track_requirements: bool = True,
):
    """
    Updates an instance list based on given required ids ones ot be discarded.

    Options for require:
        Non-empty set: Specified ids required. The special value MVPSP.ANY_INSTANCE matches can be used to require any instance to be present.
        False, None, empty set: No ids required

    Options for discard:
        True: All ids discarded, except for required ones
        Non-empty set: Specified ids discarded, except if specifically required (will print a warning).
        False, None, empty set: No ids discarded

    :param instances:
    :param id_key:
    :param id_choices:
    :param require_ids:
    :param discard_ids:
    :param discard_unknown_ids:
    :param track_requirements: If True, filfilled requirements are removed from the given require set.
    :return: True if the frame containing the instances should be kept, False if it needs to be discarded
    """
    # check arguments

    if require_ids is None:
        require_ids = set()
    if discard_ids is None:
        discard_ids = False
    if isinstance(discard_ids, set):
        discard_ids = discard_ids - require_ids
    # process require and discard ids
    fulfilled = set()
    for i in range(len(instances) - 1, -1, -1):
        inst_id = instances[i].get(id_key, None)
        if discard_unknown_ids and inst_id not in id_choices:
            # Unknown id
            instances.pop(i)
        elif inst_id in require_ids:
            # specifically required id
            fulfilled.add(inst_id)
        elif isinstance(discard_ids, set) and inst_id in discard_ids:
            # specifically ignored id
            instances.pop(i)
        elif MVPSP.ANY_INSTANCE in require_ids:
            # any id requested, which is fulfilled with the current one
            fulfilled.add(MVPSP.ANY_INSTANCE)
        elif isinstance(discard_ids, bool) and discard_ids:
            # all ids ignored if not required by any criteria above
            instances.pop(i)
    # check for any unfulfilled requirements
    unfulfilled = require_ids - fulfilled
    if track_requirements:
        require_ids.difference_update(fulfilled)
    return len(unfulfilled) == 0


def _frame_passes_filters(
    frame,
    require_rgb: bool = False,
    discard_rgb: bool = False,
    require_depth: bool = False,
    discard_depth: bool = False,
    require_objs: Set[int] = None,
    discard_objs: Union[bool, Set[int]] = False,
    require_hands: Set[int] = None,
    discard_hands: Union[bool, Set[int]] = False,
    require_handedness: Set[int] = None,
    discard_handedness: Union[bool, Set[int]] = False,
    require_gaze: Set[int] = None,
    discard_gaze: Union[bool, Set[int]] = False,
    check_all_paths: bool = False,
    track_requirements: bool = True,
):
    """
    Checks whether a single frame fulfils the requirements, and removes modalities and annotations requested to be ignored.

    Options for required objects, hands, handedness, and gaze:
        Non-empty set: Specified ids required. The special value MVPSP.ANY_INSTANCE matches can be used to require any instance to be present.
        False, None, empty set: No ids required

    Options for discarding objects, hands, handedness, and gaze:
        True: All ids discarded, except for required ones
        Non-empty set: Specified ids discarded, except if specifically required (will print a warning).
        False, None, empty set: No ids discarded

    :param frame:
    :param require_rgb:
    :param discard_rgb:
    :param require_depth:
    :param discard_depth:
    :param require_objs:
    :param discard_objs:
    :param require_hands:
    :param discard_hands:
    :param require_handedness:
    :param discard_handedness:
    :param require_gaze:
    :param discard_gaze:
    :param check_all_paths:
    :param track_requirements: If True, fulfilled requirements are removed from the given require_* sets.
    :return: True if the frame should be kept, False if it needs to be discarded
    """
    if require_rgb:
        discard_rgb = False
    if require_depth:
        discard_depth = False

    # check RGB modality
    if require_rgb and (
        "rgb_path" not in frame or (check_all_paths and not frame.get("rgb_path", Path()).is_file())
    ):
        return False
    elif discard_rgb:
        frame.pop("rgb_path", None)
        frame.pop("rgb", None)
    # check depth modality
    if require_depth and (
        "depth_path" not in frame
        or (check_all_paths and not frame.get("depth_path", Path()).is_file())
    ):
        return False
    elif discard_depth:
        frame.pop("depth_path", None)
        frame.pop("depth", None)
    # check object annotations
    frame_objs = frame.get("objects", [])
    if not _process_instance_request(
        frame_objs,
        "obj_id",
        MVPSP.OBJECT_NAMES.keys(),
        require_objs,
        discard_objs,
        track_requirements,
    ):
        return False
    # check staff hand requests
    frame_hands = frame.get("hands", [])
    if not _process_instance_request(
        frame_hands,
        "staff_id",
        MVPSP.STAFF_NAMES.keys(),
        require_hands,
        discard_hands,
        track_requirements,
    ):
        return False
    # check handedness requests
    if not _process_instance_request(
        frame_hands,
        "handedness",
        MVPSP.HANDEDNESS_NAMES.keys(),
        require_handedness,
        discard_handedness,
        track_requirements,
    ):
        return False
    # check eye gaze requests
    frame_gaze = frame.get("eye_gaze", None)
    frame_gaze = [frame_gaze] if frame_gaze is not None else []
    if not _process_instance_request(
        frame_gaze,
        "staff_id",
        MVPSP.STAFF_NAMES.keys(),
        require_gaze,
        discard_gaze,
        track_requirements,
    ):
        return False
    # update eye gaze dict
    if len(frame_gaze) == 0:
        frame.pop("eye_gaze", None)
    else:
        frame["eye_gaze"] = frame_gaze[0]
    return True


def _load_metadata_scene(args):
    # scene_dir: Path, rgb_ext="jpg", verify_file_existence=False
    scene_dir, rgb_ext, verify_file_existence = args

    rec_id = int(scene_dir.stem[:3])
    cam_id = int(scene_dir.stem[3:])
    frames = []
    try:
        scene_camera = json.load((scene_dir / "scene_camera.json").open())
        scene_gt = json.load((scene_dir / "scene_gt.json").open())
        scene_gt_info = json.load((scene_dir / "scene_gt_info.json").open())
        if (scene_dir / "scene_hand_eye.json").is_file():
            scene_hand_eye = json.load((scene_dir / "scene_hand_eye.json").open())
        else:
            scene_hand_eye = {}
    except Exception as e:
        print(f"WARNING: Could not load meta data in directory {scene_dir}: {e}")
        return frames

    prev_time_us = None
    for im_id, im_cam_info in scene_camera.items():
        assert "cam_K" in im_cam_info, f"Missing cam_K in {scene_dir / 'scene_camera.json'}"
        cam_R_w2c = (
            np.array(im_cam_info["cam_R_w2c"], dtype=float).reshape(3, 3)
            if "cam_R_w2c" in im_cam_info
            else None
        )
        cam_t_w2c = (
            np.array(im_cam_info["cam_t_w2c"], dtype=float).reshape(3, 1)
            if "cam_t_w2c" in im_cam_info
            else None
        )
        time_us = int(im_cam_info["time_us"]) if "time_us" in im_cam_info else None
        assert prev_time_us is None or prev_time_us < time_us
        frame = {
            "rec_id": rec_id,
            "cam_id": cam_id,
            "scene_id": int(scene_dir.stem),
            "im_id": int(im_id),
            "cam_K": np.array(im_cam_info["cam_K"], dtype=float).reshape(3, 3),
        }
        if cam_R_w2c is not None:
            frame["cam_R_w2c"] = cam_R_w2c
        if cam_t_w2c is not None:
            frame["cam_t_w2c"] = cam_t_w2c
        if time_us is not None:
            frame["time_us"] = time_us
            prev_time_us = time_us

        rgb_path = scene_dir / "rgb" / f"{int(im_id):06d}.{rgb_ext}"
        if verify_file_existence and not rgb_path.is_file():
            print(f"WARNING: Skipping frame with missing RGB image {rgb_path}")
            continue
        else:
            frame["rgb_path"] = rgb_path

        depth_path = scene_dir / "depth" / f"{int(im_id):06d}.png"
        frame_has_depth = "depth_scale" in im_cam_info
        if frame_has_depth:
            # assert (
            #    "depth_scale" in im_cam_info
            # ), f"Missing depth_scale for frame {im_id} in {scene_dir / 'scene_camera.json'}"
            frame["depth_scale"] = float(im_cam_info["depth_scale"])
            if verify_file_existence:
                assert depth_path.is_file(), f"Missing depth image {depth_path}"
            frame["depth_path"] = depth_path

        im_gt = scene_gt.get(im_id, [])
        im_gt_info = scene_gt_info.get(im_id, [])

        frame_objects = []
        for inst_id, inst_info in enumerate(im_gt):
            assert inst_id < len(im_gt_info)
            inst_gt_info = im_gt_info[inst_id]
            obj_mask_path = scene_dir / "mask" / f"{int(im_id):06d}_{int(inst_id):06d}.png"
            if verify_file_existence:
                assert obj_mask_path.is_file(), f"Missing mask image {depth_path}"

            inst = {
                "obj_id": int(inst_info["obj_id"]),
                "inst_id": int(inst_id),
                "cam_R_m2c": np.array(inst_info["cam_R_m2c"], dtype=float).reshape(3, 3),
                "cam_t_m2c": np.array(inst_info["cam_t_m2c"], dtype=float).reshape(3, 1),
                "obj_mask_path": obj_mask_path,
                "obj_bbox": [int(v) for v in inst_gt_info["bbox_obj"]],
            }
            if frame_has_depth:
                assert (
                    "bbox_visib" in inst_gt_info
                ), f"Missing bbox_visib for frame {im_id} in {scene_dir / 'scene_gt_info.json'}"
                assert (
                    "px_count_valid" in inst_gt_info
                ), f"Missing px_count_valid for frame {im_id} in {scene_dir / 'scene_gt_info.json'}"
                assert (
                    "px_count_visib" in inst_gt_info
                ), f"Missing px_count_visib for frame {im_id} in {scene_dir / 'scene_gt_info.json'}"
                assert (
                    "visib_fract" in inst_gt_info
                ), f"Missing visib_fract for frame {im_id} in {scene_dir / 'scene_gt_info.json'}"
                obj_mask_visib_path = (
                    scene_dir / "mask_visib" / f"{int(im_id):06d}_{int(inst_id):06d}.png"
                )
                if verify_file_existence:
                    assert obj_mask_visib_path.is_file(), f"Missing visible mask image {depth_path}"

                inst["obj_bbox_visib"] = [int(v) for v in inst_gt_info["bbox_visib"]]
                inst["obj_mask_visib_path"] = obj_mask_visib_path
                inst["px_count_valid"] = int(inst_gt_info["px_count_valid"])
                inst["px_count_visib"] = int(inst_gt_info["px_count_visib"])
                inst["visib_fract"] = float(inst_gt_info["visib_fract"])
            frame_objects.append(inst)

        if len(frame_objects) > 0:
            frame["objects"] = frame_objects

        im_hand_eye = scene_hand_eye.get(im_id, {})
        frame_hands = []
        if "left_hand_joints" in im_hand_eye:
            frame_hands.append(
                {
                    "handedness": MVPSP.HANDEDNESS_LEFT,
                    "staff_id": (
                        MVPSP.STAFF_SURGEON
                        if cam_id == MVPSP.CAMERA_SURGEON
                        else MVPSP.STAFF_ASSISTANT
                    ),
                    "joint_positions": np.array(
                        im_hand_eye["left_hand_joints"], dtype=float
                    ).reshape(26, 3),
                }
            )
        if "right_hand_joints" in im_hand_eye:
            frame_hands.append(
                {
                    "handedness": 1,
                    "staff_id": (
                        MVPSP.STAFF_SURGEON
                        if cam_id == MVPSP.CAMERA_SURGEON
                        else MVPSP.STAFF_ASSISTANT
                    ),
                    "joint_positions": np.array(
                        im_hand_eye["right_hand_joints"], dtype=float
                    ).reshape(26, 3),
                }
            )
        if len(frame_hands) > 0:
            frame["hands"] = frame_hands
        if "eye_gaze" in im_hand_eye:
            frame["eye_gaze"] = {
                "origin": np.array(im_hand_eye["eye_gaze"]["origin"], dtype=float).reshape(3, 1),
                "direction": np.array(im_hand_eye["eye_gaze"]["direction"], dtype=float).reshape(
                    3, 1
                ),
                "staff_id": (
                    MVPSP.STAFF_SURGEON if cam_id == MVPSP.CAMERA_SURGEON else MVPSP.STAFF_ASSISTANT
                ),
            }
        frames.append(frame)
    return frames


class MvpspSingleviewDataset(MVPSP):
    def __init__(
        self,
        root_dir: Union[Path, str],
        is_test: bool = False,
        include_subsets: Union[list, str] = None,
        include_rec_ids: list = None,
        include_cam_ids: list = None,
        models_subdir: str = "models_eval",
        cache_dir: Union[Path, str] = None,
        num_workers=cpu_count(),
    ):
        """

        :param root_dir: Path to the MVPSP dataset root directory.
        :param is_test: Indicate whether the dataset is used for training or testing.
        :param include_subsets: A string or list of strings indicating the subsets to load, or None to include all subsets based on is_test.
        :param include_rec_ids: A list of recording ids to load, or None to include all.
        :param include_cam_ids: A list of camera ids to load, or None to include all.
        :param cache_dir: Path to an optional cache directory. Set to None to disable caching.
        """

        # parse root_dir
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        assert root_dir.is_dir(), f"MVPSP root directory not found: {root_dir}"
        self.root_dir = root_dir
        obj_model_dir = root_dir / models_subdir
        assert obj_model_dir.is_dir(), f"MVPSP model directory not found: {obj_model_dir}"
        self.is_test = is_test
        # parse subsets
        if isinstance(include_subsets, str):
            include_subsets = [include_subsets]
        elif include_subsets is None or len(include_subsets) == 0:
            if is_test:
                include_subsets = MVPSP.TEST_SUBSETS
            else:
                include_subsets = MVPSP.TRAIN_SUBSETS
        self.include_subsets = []
        for s in include_subsets:
            if s not in MVPSP.TRAIN_SUBSETS + MVPSP.TEST_SUBSETS:
                print(f"Ignoring unknown subsets: {s}")
            else:
                if is_test and s in MVPSP.TRAIN_SUBSETS:
                    print(f"WARNING: LOADING TRAINING SUBSET {s} AT TEST TIME!")
                elif not is_test and s in MVPSP.TEST_SUBSETS:
                    print(f"WARNING: LOADING TEST SUBSET {s} AT TRAINING TIME!")
                self.include_subsets.append(s)
        assert len(self.include_subsets) > 0, f"No subsets given"

        self.include_cam_ids = include_cam_ids
        self.include_rec_ids = include_rec_ids

        # Try to load metadata from cache
        if cache_dir is not None:
            if not self._load_metadata_from_cache(cache_dir):
                self._load_metadata(num_workers=num_workers)
                self._save_metadata_to_cache(cache_dir)
        else:
            self._load_metadata(num_workers=num_workers)
        self._load_objects(models_subdir)

    def _load_metadata(self, num_workers=1):
        """
        This method loads the metadata information for all frames of the MVPSP dataset.
        Each frame instance is stored in a dict and should have the following format:
        {
            'rec_id',
            'cam_id',
            'cam_K',
            'cam_R_w2c',
            'cam_t_w2c',
            'time_us',
            'depth_scale',
            'rgb_path',
            'depth_path',
            'objects': [
                {
                    'obj_id': 0,
                    'inst_id': 0,
                    'obj_R_m2c': ,
                    'obj_t_m2c': ,
                    'obj_bbox': [],
                    'obj_bbox_visib': [],
                    'px_count_all': 0,
                    'px_count_valid': 0,
                    'px_count_visib': 0,
                    'visib_fract': 0.0,
                    'obj_mask_path': ,
                    'obj_mask_visib_path': ,
                }
            ],
            'hands': [
                {
                    'handedness': HANDEDNESS,
                    'staff_id': STAFF_IDS,
                    'joint_positions': [],
                },
            ],
            'gaze': {
                'staff_id': STAFF_IDS,
                'origin': [],
                'direction': [],
            }
        }
        """

        print(f"Loading frame metadata from {self.root_dir}")
        self.frames = []
        worker_args = []
        for subset in self.include_subsets:
            subset_dir = self.root_dir / subset
            if not subset_dir.is_dir():
                print(f"MVPSP subset directory not found: {subset_dir}")
                continue

            rgb_ext = "png" if subset == MVPSP.TRAIN_SUBSET_SYNTH else "jpg"

            for scene_dir in subset_dir.iterdir():
                if not (
                    scene_dir.is_dir()
                    and len(scene_dir.stem) == 6
                    and scene_dir.stem.isdecimal()
                    and (scene_dir / "scene_camera.json").is_file()
                    and (scene_dir / "scene_gt.json").is_file()
                    and (scene_dir / "scene_gt_info.json").is_file()
                    and ((scene_dir / "rgb").is_dir())
                    and ((scene_dir / "depth").is_dir())
                    and ((scene_dir / "mask").is_dir())
                    and ((scene_dir / "mask_visib").is_dir())
                ):
                    continue

                rec_id = int(scene_dir.stem[:3])
                cam_id = int(scene_dir.stem[3:])
                if (
                    self.include_rec_ids is not None
                    and rec_id not in self.include_rec_ids
                    or self.include_cam_ids is not None
                    and cam_id not in self.include_cam_ids
                ):
                    continue
                worker_args.append(
                    (
                        scene_dir,
                        rgb_ext,
                        False,
                    )
                )
        num_workers = max(1, min(num_workers, len(worker_args)))
        with Pool(num_workers) as pool:
            if num_workers <= 1:
                scene_frames = [_load_metadata_scene(args) for args in worker_args]
            else:
                scene_frames = pool.imap(_load_metadata_scene, worker_args)
            self.frames = list(chain.from_iterable(scene_frames))

    def _load_objects(self, model_dir="models_eval"):
        self.objects = {}
        models_info = json.load((self.root_dir / model_dir / "models_info.json").open())
        for obj_id, info in models_info.items():
            obj_id = int(obj_id)
            obj = {
                "min_pos": np.array(
                    [info["min_x"], info["min_y"], info["min_z"]], dtype=float
                ).reshape(3, 1),
                "size": np.array(
                    [info["size_x"], info["size_y"], info["size_z"]], dtype=float
                ).reshape(3, 1),
                "diameter": float(info["diameter"]),
            }
            if "tip_pos" in info:
                obj["tip_pos"] = np.array(info["tip_pos"], dtype=float).reshape(3, 1)
            if "tip_dir" in info:
                obj["tip_dir"] = np.array(info["tip_dir"], dtype=float).reshape(3, 1)
            if "symmetries_discrete" in info:
                obj["symmetries_discrete"] = [
                    np.array(tf, dtype=float).reshape(4, 4) for tf in info["symmetries_discrete"]
                ]
            obj["mesh"] = trimesh.load(self.root_dir / model_dir / f"obj_{obj_id:06d}.ply")
            self.objects[obj_id] = obj

    def _load_metadata_from_cache(self, cache_dir: Union[Path, str]):
        """
        Tries to load the dataset metadata from a cache directory.
        :param cache_dir: Path to an optional cache directory. Set to None to disable caching.
        :return: True if metadata was successfully loaded from cache, False otherwise.
        """
        if cache_dir is None:
            return False
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            return False
        index_file = cache_dir / "index.json"
        if not index_file.is_file():
            return False
        cache_index = json.loads(index_file.read_text())
        dataset_key = self.__class__.__name__
        subset_key = ",".join(sorted(self.include_subsets))
        rec_ids_key = (
            ",".join(map(str, sorted(self.include_rec_ids)))
            if self.include_rec_ids is not None
            else "All"
        )
        cam_ids_key = ",".join(map(str, sorted(self.include_cam_ids)))
        cache_file = (
            cache_index.get(dataset_key, {})
            .get(subset_key, {})
            .get(rec_ids_key, {})
            .get(cam_ids_key, None)
        )
        if cache_file is None:
            return False
        cache_file = cache_dir / cache_file
        if not cache_file.is_file():
            return False
        self.frames = np.load(cache_file, allow_pickle=True)["frames"]
        print(f"Loaded frame metadata from cache: {cache_file}")
        return True

    def _save_metadata_to_cache(self, cache_dir: Union[Path, str]):
        if cache_dir is None:
            return False
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

        index_file = cache_dir / "index.json"
        if index_file.is_file():
            cache_index = json.loads(index_file.read_text())
        else:
            cache_index = {}
        dataset_key = self.__class__.__name__
        subset_key = ",".join(sorted(self.include_subsets))
        rec_ids_key = (
            ",".join(map(str, sorted(self.include_rec_ids)))
            if self.include_rec_ids is not None
            else "All"
        )
        cam_ids_key = ",".join(map(str, sorted(self.include_cam_ids)))
        cache_file = (
            cache_index.get(dataset_key, {})
            .get(subset_key, {})
            .get(rec_ids_key, {})
            .get(cam_ids_key, None)
        )
        if cache_file is None:
            cache_id = 1
            while (cache_dir / f"cache_{cache_id:04d}.npz").is_file():
                cache_id += 1
            cache_file = cache_dir / f"cache_{cache_id:04d}.npz"
        np.savez_compressed(cache_file, frames=self.frames)
        cache_update = {dataset_key: {subset_key: {rec_ids_key: {cam_ids_key: cache_file.name}}}}
        cache_index = nested_dict_update(cache_index, cache_update)
        index_file.write_text(json.dumps(cache_index))
        print(f"Saved frame metadata to cache: {cache_file}")
        return True

    def _filter_objects(self, discard_objs: Set[int]):
        for obj_id in MVPSP.OBJECT_NAMES.keys():
            if obj_id in discard_objs:
                self.objects.pop(obj_id, None)

    def apply_filter(
        self,
        require_rgb: bool = False,
        discard_rgb: bool = False,
        require_depth: bool = False,
        discard_depth: bool = False,
        require_objs: Set[int] = None,
        discard_objs: Union[bool, Set[int]] = False,
        require_hands: Set[int] = None,
        discard_hands: Union[bool, Set[int]] = False,
        require_handedness: Set[int] = None,
        discard_handedness: Union[bool, Set[int]] = False,
        require_gaze: Set[int] = None,
        discard_gaze: Union[bool, Set[int]] = False,
        check_all_paths: bool = False,
    ):
        """
        Filter dataset samples and discard the ones which do not contain the requested modalities or annotations.

        Options for required objects, hands, handedness, and gaze:
            Non-empty set: Specified ids required. The special value MVPSP.ANY_INSTANCE matches can be used to require any instance to be present.
            False, None, empty set: No ids required

        Options for discarding objects, hands, handedness, and gaze:
            True: All ids discarded, except for required ones
            Non-empty set: Specified ids discarded, except if specifically required (will print a warning).
            False, None, empty set: No ids discarded

        :param rgb: RGB image modality.
        :param depth: Depth map modality.
        :return:
        """
        # check arguments
        if require_rgb and discard_rgb:
            print(f"WARN: Cannot require and discard RGB at the same time! Defaulting to require.")
            discard_rgb = False
        if require_depth and discard_depth:
            print(
                f"WARN: Cannot require and discard depth at the same time! Defaulting to require."
            )
            discard_depth = False
        _verify_require_discard(
            require_objs,
            discard_objs,
            "Cannot simultaneously require and discard object IDs {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_hands,
            discard_hands,
            "Cannot simultaneously require and discard hands for staff IDs {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_handedness,
            discard_handedness,
            "Cannot simultaneously require and discard handedness {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_gaze,
            discard_gaze,
            "Cannot simultaneously require and discard gaze for staff IDs {ids}. Defaulting to require.",
        )

        # filter samples based on given criteria
        self.frames = [
            f
            for f in self.frames
            if _frame_passes_filters(
                f,
                require_rgb,
                discard_rgb,
                require_depth,
                discard_depth,
                require_objs,
                discard_objs,
                require_hands,
                discard_hands,
                require_handedness,
                discard_handedness,
                require_gaze,
                discard_gaze,
                check_all_paths,
                track_requirements=False,
            )
        ]
        if isinstance(discard_objs, Set):
            self._filter_objects(discard_objs)

    def apply_filter_fn(self, fn: Callable[[dict], bool]):
        """
        Applies a filter function to the samples in the dataset, keeping only samples where the filter returns True.

        :param fn: filter function
        :return:
        """
        self.frames = [v for v in self.frames if fn(v)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx].copy()

    def as_bop_targets(self, out_path: Path = None):
        targets = []
        for frame in self.frames:
            for obj in frame.get("objects", []):
                targets.append(
                    {
                        "scene_id": frame["rec_id"] * 1000 + frame["cam_id"],
                        "im_id": frame["im_id"],
                        "obj_id": obj["obj_id"],
                        "inst_count": 1,
                    }
                )
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(targets, out_path.open("w"))
        return targets

    def get_bop_target_samples(self, targets: NestedDict = None) -> NestedDict:
        samples = NestedDict()
        for frame in self.frames:
            if targets is not None and frame["scene_id"] not in targets:
                continue
            if targets is not None and frame["im_id"] not in targets[frame["scene_id"]]:
                continue
            for obj in frame.get("objects", []):
                if (
                    targets is not None
                    and obj["obj_id"] not in targets[frame["scene_id"]][frame["im_id"]]
                ):
                    continue
                s = frame.copy()
                s.update(obj)
                s.pop("objects", None)
                samples[frame["scene_id"]][frame["im_id"]][obj["obj_id"]][obj["inst_id"]] = s
        return samples


class MvpspMultiviewDataset(MVPSP):

    def __init__(
        self,
        root_dir: Union[Path, str],
        is_test: bool = False,
        include_subsets: Union[list, str] = None,
        include_rec_ids: list = None,
        include_cam_ids: list = None,
        models_subdir: str = "models_eval",
        cache_dir: Union[Path, str] = None,
        max_temporal_offset_us: int = 8e3,
        num_workers=cpu_count(),
    ):
        """

        :param root_dir: Path to the MVPSP dataset root directory.
        :param is_test: Indicate whether the dataset is used for training or testing.
        :param include_subsets: A string or list of strings indicating the subsets to load, or None to include all subsets based on is_test.
        :param include_rec_ids: A list of recording ids to include in the index, or None to include all.
        :param include_cam_ids: A list of camera ids to include in the index, or None to include all.
        :param cache_dir: Path to an optional cache directory. Set to None to disable caching.
        :param max_temporal_offset_us: Maximum temporal offset between any pair of frames in the sample,
                expressed in microseconds. Defaults to 8 milliseconds.
        """

        assert max_temporal_offset_us >= 0, f"max_temporal_offset_us cannot be negative."
        self.max_temporal_offset_us = max_temporal_offset_us

        # parse subsets
        if isinstance(include_subsets, str):
            include_subsets = [include_subsets]
        elif include_subsets is None or len(include_subsets) == 0:
            if is_test:
                include_subsets = MVPSP.TEST_SUBSETS
            else:
                include_subsets = [MVPSP.TRAIN_SUBSET_WETLAB]
        subsets_filtered = []
        for s in include_subsets:
            if s in [MVPSP.TRAIN_SUBSET_SYNTH, MVPSP.TRAIN_SUBSET_PBR]:
                print(
                    f"WARNING: Loading multi-view samples from synthetic subsets is not supported yet: {s}"
                )
            else:
                if is_test and s in MVPSP.TRAIN_SUBSETS:
                    print(f"WARNING: LOADING TRAINING SUBSET {s} AT TEST TIME!")
                elif not is_test and s in MVPSP.TEST_SUBSETS:
                    print(f"WARNING: LOADING TEST SUBSET {s} AT TRAINING TIME!")
                subsets_filtered.append(s)

        self.frames = MvpspSingleviewDataset(
            root_dir=root_dir,
            is_test=is_test,
            include_subsets=subsets_filtered,
            include_rec_ids=include_rec_ids,
            include_cam_ids=include_cam_ids,
            models_subdir=models_subdir,
            cache_dir=cache_dir,
            num_workers=num_workers,
        )

        # Try to load multiview index from cache
        if cache_dir is not None:
            if not self._load_multiview_index_from_cache(cache_dir):
                self._create_multiview_index(num_workers=num_workers)
                self._save_multiview_index_to_cache(cache_dir)
        else:
            self._create_multiview_index(num_workers=num_workers)

    def _locally_minimize_multiview_std(self, rec_frames, cam_indices):
        """

        :param rec_frames: dict that maps cam_id to a sorted list of frame indices within self.frames
        :param cam_indices: dict that maps cam_id to the current frame index within rec_frames
        :return: dict that maps each cam_id to the selected frame index within rec_frames
        """
        # Multiview extraction not possible if no more frames available for any camera
        if cam_indices is None:
            return None
        for cam_id, cam_idx in cam_indices.items():
            if cam_idx >= len(rec_frames[cam_id]):
                return None
        # collect timestamps
        mv_timestamps = {
            cam_id: self.frames[rec_frames[cam_id][cam_indices[cam_id]]]["time_us"]
            for cam_id in cam_indices
        }
        mv_std = np.std(list(mv_timestamps.values()))
        mv_start_time = np.min(list(mv_timestamps.values()))
        mv_end_time = np.max(list(mv_timestamps.values()))
        # initial multiview candidate must be valid
        if mv_end_time - mv_start_time > self.max_temporal_offset_us:
            return None
        # check if multiview contains closest possible frames, by testing replacement of frame t with t+1 for each camera
        candidate_improved = True
        while candidate_improved:
            candidate_improved = False
            for cam_id, cam_idx in cam_indices.items():
                if cam_idx + 1 >= len(rec_frames[cam_id]):
                    continue
                candidate = dict(cam_indices)
                candidate[cam_id] = cam_indices[cam_id] + 1
                replacement_timestamp = self.frames[rec_frames[cam_id][candidate[cam_id]]][
                    "time_us"
                ]
                candidate_start_time = min(mv_start_time, replacement_timestamp)
                candidate_end_time = max(mv_end_time, replacement_timestamp)
                if candidate_end_time - candidate_start_time > self.max_temporal_offset_us:
                    continue
                candidate_timestamps = dict(mv_timestamps)
                candidate_timestamps[cam_id] = replacement_timestamp
                candidate_std = np.std(list(candidate_timestamps.values()))
                if candidate_std >= mv_std:
                    continue
                cam_indices = candidate
                mv_timestamps = candidate_timestamps
                mv_start_time = candidate_start_time
                mv_end_time = candidate_end_time
                mv_std = candidate_std
                candidate_improved = True
        return cam_indices

    def _create_multiview_index_for_recording(self, rec_frames):
        indices = []
        cam_indices = {cam_id: 0 for cam_id in rec_frames.keys()}
        while cam_indices is not None:
            # initialize multiview candidate defined by cam_indices
            cam_timestamps = {}
            for cam_id in cam_indices:
                if cam_indices[cam_id] >= len(rec_frames[cam_id]):
                    cam_indices = None
                    break
                cam_timestamps[cam_id] = self.frames[rec_frames[cam_id][cam_indices[cam_id]]][
                    "time_us"
                ]
            if cam_indices is None:
                break
            min_time_cam_id = min(cam_timestamps, key=cam_timestamps.get)
            max_time_cam_id = max(cam_timestamps, key=cam_timestamps.get)

            # find valid multiview candidate
            while (
                cam_timestamps[max_time_cam_id] - cam_timestamps[min_time_cam_id]
                > self.max_temporal_offset_us
            ):
                cam_indices[min_time_cam_id] += 1
                if cam_indices[min_time_cam_id] >= len(rec_frames[min_time_cam_id]):
                    # no more valid multiview candidates possible
                    cam_indices = None
                    break
                cam_timestamps[min_time_cam_id] = self.frames[
                    rec_frames[min_time_cam_id][cam_indices[min_time_cam_id]]
                ]["time_us"]
                min_time_cam_id = min(cam_timestamps, key=cam_timestamps.get)
                max_time_cam_id = max(cam_timestamps, key=cam_timestamps.get)

            # locally minimize std of timestamps to find optimal multiview
            cam_indices = self._locally_minimize_multiview_std(rec_frames, cam_indices)
            if cam_indices is not None:
                indices.append(
                    [rec_frames[cam_id][cam_indices[cam_id]] for cam_id in rec_frames.keys()]
                )
                for cam_id in cam_indices:
                    cam_indices[cam_id] += 1
        return indices

    def _create_multiview_index(self, num_workers=1):
        print("Compute multi-view index...")
        # group frames by recording and camera
        scene_frames = {}
        all_cam_ids = set()
        for frame_idx, frame in enumerate(self.frames):
            rec_id = frame["rec_id"]
            cam_id = frame["cam_id"]
            all_cam_ids.add(cam_id)
            if rec_id not in scene_frames:
                scene_frames[rec_id] = {}
            if cam_id not in scene_frames[rec_id]:
                scene_frames[rec_id][cam_id] = []
            scene_frames[rec_id][cam_id].append(frame_idx)

        # verify that all cameras are included for each recording
        verified_recs = []
        for rec_id, rec_frames in scene_frames.items():
            # skip recordings with missing cameras
            skip_recording = False
            for cam_id in all_cam_ids:
                if cam_id not in rec_frames:
                    skip_recording = True
                    break
            if skip_recording:
                print(
                    f"Skipping recording id {rec_id} due to missing cameras (requested {','.join(all_cam_ids)})"
                )
                continue
            verified_recs.append(rec_frames)

        # compute multiview frames
        num_workers = max(1, min(num_workers, len(verified_recs)))
        with Pool(num_workers) as pool:
            if num_workers <= 1:
                result = [self._create_multiview_index_for_recording(f) for f in verified_recs]
            else:
                result = pool.imap(self._create_multiview_index_for_recording, verified_recs)
            self.index = list(chain.from_iterable(result))

    def _load_multiview_index_from_cache(self, cache_dir: Union[Path, str]):
        """
        Tries to load the dataset metadata from a cache directory.
        :param cache_dir: Path to an optional cache directory. Set to None to disable caching.
        :return: True if metadata was successfully loaded from cache, False otherwise.
        """
        if cache_dir is None:
            return False
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            return False
        index_file = cache_dir / "index.json"
        if not index_file.is_file():
            return False
        cache_index = json.loads(index_file.read_text())
        dataset_key = self.__class__.__name__
        subset_key = ",".join(sorted(self.frames.include_subsets))
        rec_ids_key = (
            ",".join(map(str, sorted(self.frames.include_rec_ids)))
            if self.frames.include_rec_ids is not None
            else "All"
        )
        cam_ids_key = ",".join(map(str, sorted(self.frames.include_cam_ids)))
        mto_key = str(self.max_temporal_offset_us)
        cache_file = (
            cache_index.get(dataset_key, {})
            .get(subset_key, {})
            .get(rec_ids_key, {})
            .get(cam_ids_key, {})
            .get(mto_key, None)
        )
        if cache_file is None:
            return False
        cache_file = cache_dir / cache_file
        if not cache_file.is_file():
            return False
        self.index = np.load(cache_file, allow_pickle=True)["index"].tolist()
        print(f"Loaded multiview index from cache: {cache_file}")
        return True

    def _save_multiview_index_to_cache(self, cache_dir: Union[Path, str]):
        if cache_dir is None:
            return False
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

        index_file = cache_dir / "index.json"
        if index_file.is_file():
            cache_index = json.loads(index_file.read_text())
        else:
            cache_index = {}
        dataset_key = self.__class__.__name__
        subset_key = ",".join(sorted(self.frames.include_subsets))
        rec_ids_key = (
            ",".join(map(str, sorted(self.frames.include_rec_ids)))
            if self.frames.include_rec_ids is not None
            else "All"
        )
        cam_ids_key = ",".join(map(str, sorted(self.frames.include_cam_ids)))
        mto_key = str(self.max_temporal_offset_us)
        cache_file = (
            cache_index.get(dataset_key, {})
            .get(subset_key, {})
            .get(rec_ids_key, {})
            .get(cam_ids_key, {})
            .get(mto_key, None)
        )
        if cache_file is None:
            cache_id = 1
            while (cache_dir / f"cache_{cache_id:04d}.npz").is_file():
                cache_id += 1
            cache_file = cache_dir / f"cache_{cache_id:04d}.npz"
        np.savez_compressed(cache_file, index=self.index)
        cache_update = {
            dataset_key: {subset_key: {rec_ids_key: {cam_ids_key: {mto_key: cache_file.name}}}}
        }
        cache_index = nested_dict_update(cache_index, cache_update)
        index_file.write_text(json.dumps(cache_index))
        print(f"Saved multiview index to cache: {cache_file}")
        return True

    def apply_filter(
        self,
        require_rgb: bool = False,
        discard_rgb: bool = False,
        require_depth: bool = False,
        discard_depth: bool = False,
        require_objs: Set[int] = None,
        discard_objs: Union[bool, Set[int]] = False,
        require_hands: Set[int] = None,
        discard_hands: Union[bool, Set[int]] = False,
        require_handedness: Set[int] = None,
        discard_handedness: Union[bool, Set[int]] = False,
        require_gaze: Set[int] = None,
        discard_gaze: Union[bool, Set[int]] = False,
        check_all_paths: bool = False,
    ):
        """
        Filter dataset samples and discard the ones which do not contain the requested modalities or annotations.

        Options for required objects, hands, handedness, and gaze:
            Non-empty set: Specified annotations required in at least one frame. The special value MVPSP.ANY_INSTANCE matches can be used to require any annotation to be present.
            False, None, empty set: No annotations required
        Options for ignored objects, hands, handedness, and gaze:
            True: All annotations ignored, except for required ones
            Non-empty set: Specified annotations ignored, except if specifically required (will print a warning).
            False, None, empty set: No annotations ignored

        :return:
        """

        # check arguments
        if require_rgb and discard_rgb:
            print("Cannot simultaneously require and discard RGB modality. Defaulting to require.")
            discard_rgb = False
        if require_depth and discard_depth:
            print(
                "Cannot simultaneously require and discard depth modality. Defaulting to require."
            )
            discard_depth = False
        _verify_require_discard(
            require_objs,
            discard_objs,
            "Cannot simultaneously require and discard object IDs {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_hands,
            discard_hands,
            "Cannot simultaneously require and discard hands for staff IDs {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_handedness,
            discard_handedness,
            "Cannot simultaneously require and discard handedness {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_gaze,
            discard_gaze,
            "Cannot simultaneously require and discard gaze for staff IDs {ids}. Defaulting to require.",
        )

        # filter samples based on given criteria
        for sample_idx in range(len(self)):
            sample = self[sample_idx]
            # Copy requirement sets as they will be altered
            remaining_require_objs = (
                require_objs.copy() if isinstance(require_objs, Set) else require_objs
            )
            remaining_require_hands = (
                require_hands.copy() if isinstance(require_hands, Set) else require_hands
            )
            remaining_require_handedness = (
                require_handedness.copy()
                if isinstance(require_handedness, Set)
                else require_handedness
            )
            remaining_require_gaze = (
                require_gaze.copy() if isinstance(require_gaze, Set) else require_gaze
            )

            for frame in sample:
                # rgb and depth modality criteria have to be passed by each frame independently
                if not _frame_passes_filters(
                    frame,
                    require_rgb,
                    discard_rgb,
                    require_depth,
                    discard_depth,
                    check_all_paths=check_all_paths,
                    track_requirements=False,
                ):
                    self.index[sample_idx] = None
                    break
                # Annotation criteria can be passed partially by each frame
                _frame_passes_filters(
                    frame,
                    require_objs=remaining_require_objs,
                    discard_objs=discard_objs,
                    require_hands=remaining_require_hands,
                    discard_hands=discard_hands,
                    require_handedness=remaining_require_handedness,
                    discard_handedness=discard_handedness,
                    require_gaze=remaining_require_gaze,
                    discard_gaze=discard_gaze,
                    track_requirements=True,
                )

            if (
                (remaining_require_objs is not None and len(remaining_require_objs) > 0)
                or (remaining_require_hands is not None and len(remaining_require_hands) > 0)
                or (
                    remaining_require_handedness is not None
                    and len(remaining_require_handedness) > 0
                )
                or (remaining_require_gaze is not None and len(remaining_require_gaze) > 0)
            ):
                self.index[sample_idx] = None
        self.index = [idx for idx in self.index if idx is not None]
        # TODO remove unreferenced frames from self.frames, to save on memory
        if isinstance(discard_objs, Set):
            self.frames._filter_objects(discard_objs)

    def apply_filter_fn(self, fn: Callable[[Sequence[dict]], bool]):
        """
        Applies a filter function to the samples in the dataset, keeping only samples where the filter returns True.

        :param fn: filter function
        :return:
        """
        self.index = [sample for i, sample in enumerate(self.index) if fn(self[i])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return [self.frames[i] for i in self.index[idx]]

    def as_bop_targets(self, out_path: Path = None, enforce_n_cams: int = 0):
        targets = []
        for i in range(len(self)):
            if enforce_n_cams > 0 and len(self[i]) != enforce_n_cams:
                logging.info(f"Skipping frame {i} with cam_ids {','.join(map(str,sorted([frame['cam_id'] for frame in self[i]])))} due to enforced camera count of {enforce_n_cams}")
                continue
            for frame in self[i]:
                for obj in frame.get("objects", []):
                    targets.append(
                        {
                            "scene_id": frame["rec_id"] * 1000 + frame["cam_id"],
                            "multiview_id": i,
                            "im_id": frame["im_id"],
                            "obj_id": obj["obj_id"],
                            "inst_count": 1,
                        }
                    )
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(targets, out_path.open("w"))
        return targets


class MvpspSingleviewSequenceDataset(MVPSP):
    def __init__(
        self,
        root_dir: Union[Path, str],
        is_test: bool = False,
        include_subsets: Union[list, str] = None,
        include_rec_ids: list = None,
        include_cam_ids: list = None,
        models_subdir: str = "models_eval",
        cache_dir: Union[Path, str] = None,
        sequence_length: int = 2,
        sequence_stride: int = 0,
        min_frame_gap_us: int = 0,
        max_frame_gap_us: int = 67e3,
    ):
        """

        :param root_dir: Path to the MVPSP dataset root directory.
        :param is_test: Indicate whether the dataset is used for training or testing.
        :param include_subsets: A string or list of strings indicating the subsets to load, or None to include all subsets based on is_test.
        :param include_rec_ids: A list of recording ids to include in the index, or None to include all.
        :param include_cam_ids: A list of camera ids to include in the index, or None to include all.
        :param cache_dir: Path to an optional cache directory. Set to None to disable caching.
        :param sequence_length: The number of consecutive frames in each sequence.
        :param sequence_stride: The default value of 0 indicates a stride of sequence_length, effectively returning non-overlapping sequences.
        :param min_frame_gap_us: The minimum temporal difference between any two consequtive frames in each sequence, expressesd in microseconds.
                Defaults to 0.
        :param max_frame_gap_us: The maximum temporal difference between any two consequtive frames in each sequence, expressesd in microseconds.
                Defaults to 67 milliseconds.
        """
        assert sequence_length >= 1
        self.sequence_length = sequence_length
        if sequence_stride <= 0:
            sequence_stride = sequence_length
            assert sequence_stride > 0
        self.sequence_stride = sequence_stride
        assert min_frame_gap_us >= 0
        assert max_frame_gap_us > 0
        assert max_frame_gap_us >= min_frame_gap_us
        self.min_frame_gap_us = min_frame_gap_us
        self.max_frame_gap_us = max_frame_gap_us

        # parse subsets
        if isinstance(include_subsets, str):
            include_subsets = [include_subsets]
        elif include_subsets is None or len(include_subsets) == 0:
            if is_test:
                include_subsets = MVPSP.TEST_SUBSETS
            else:
                include_subsets = [MVPSP.TRAIN_SUBSET_WETLAB]
        subsets_filtered = []
        for s in include_subsets:
            if s in [MVPSP.TRAIN_SUBSET_SYNTH, MVPSP.TRAIN_SUBSET_PBR]:
                print(
                    f"WARNING: Loading sequence samples from synthetic subsets is not supported yet: {s}"
                )
            else:
                if is_test and s in MVPSP.TRAIN_SUBSETS:
                    print(f"WARNING: LOADING TRAINING SUBSET {s} AT TEST TIME!")
                elif not is_test and s in MVPSP.TEST_SUBSETS:
                    print(f"WARNING: LOADING TEST SUBSET {s} AT TRAINING TIME!")
                subsets_filtered.append(s)

        self.frames = MvpspSingleviewDataset(
            root_dir=root_dir,
            is_test=is_test,
            include_subsets=subsets_filtered,
            include_rec_ids=include_rec_ids,
            include_cam_ids=include_cam_ids,
            cache_dir=cache_dir,
            models_subdir=models_subdir,
        )

        # Try to load sequence index from cache
        if cache_dir is not None:
            if not self._load_sequence_index_from_cache(cache_dir):
                self._create_sequence_index()
                self._save_sequence_index_to_cache(cache_dir)
        else:
            self._create_sequence_index()

    def _create_sequence_index(self):
        sequences = {}
        # collect per-scene sequences that satisfy the sequence_min_frame_gap_us and sequence_max_frame_gap_us constraints
        for frame_idx in range(len(self.frames)):
            frame = self.frames[frame_idx]
            rec_id = frame["rec_id"]
            cam_id = frame["cam_id"]
            if rec_id not in sequences:
                sequences[rec_id] = {}
            if cam_id not in sequences[rec_id]:
                sequences[rec_id][cam_id] = []
            if (
                len(sequences[rec_id][cam_id]) > 0
                and abs(
                    self.frames[sequences[rec_id][cam_id][-1][-1]]["time_us"] - frame["time_us"]
                )
                < self.min_frame_gap_us
            ):
                # skip frames that are temporally closer than sequence_min_frame_gap_us to the last frame
                continue
            elif (
                len(sequences[rec_id][cam_id]) > 0
                and abs(
                    self.frames[sequences[rec_id][cam_id][-1][-1]]["time_us"] - frame["time_us"]
                )
                < self.max_frame_gap_us
            ):
                # append frames within sequence_min_frame_gap_us to sequence_max_frame_gap_us to the current sequence
                sequences[rec_id][cam_id][-1].append(frame_idx)
            else:
                # start a new sequence when the next frame is more than sequence_max_frame_gap_us later than the last on of the previous sequence
                sequences[rec_id][cam_id].append([frame_idx])

        # split long sequences into multiple sequences of sequence_length while satisfying sequence_stride
        self.index = []
        for rec_id, cam_sequences in sequences.items():
            for cam_id, long_sequences in cam_sequences.items():
                for long_sequence in long_sequences:
                    for i in range(
                        0, len(long_sequence) - self.sequence_length + 1, self.sequence_stride
                    ):
                        self.index.append(long_sequence[i : i + self.sequence_length])

    def _load_sequence_index_from_cache(self, cache_dir: Union[Path, str]):
        """
        Tries to load the dataset metadata from a cache directory.
        :param cache_dir: Path to an optional cache directory. Set to None to disable caching.
        :return: True if metadata was successfully loaded from cache, False otherwise.
        """
        if cache_dir is None:
            return False
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            return False
        index_file = cache_dir / "index.json"
        if not index_file.is_file():
            return False
        cache_index = json.loads(index_file.read_text())
        dataset_key = self.__class__.__name__
        subset_key = ",".join(sorted(self.frames.include_subsets))
        rec_ids_key = (
            ",".join(map(str, sorted(self.frames.include_rec_ids)))
            if self.frames.include_rec_ids is not None
            else "All"
        )
        cam_ids_key = ",".join(map(str, sorted(self.frames.include_cam_ids)))
        seq_length_key = str(self.sequence_length)
        seq_stride_key = str(self.sequence_stride)
        seq_mfgap_key = str(self.max_frame_gap_us)
        cache_file = (
            cache_index.get(dataset_key, {})
            .get(subset_key, {})
            .get(rec_ids_key, {})
            .get(cam_ids_key, {})
            .get(seq_length_key, {})
            .get(seq_stride_key, {})
            .get(seq_mfgap_key, None)
        )
        if cache_file is None:
            return False
        cache_file = cache_dir / cache_file
        if not cache_file.is_file():
            return False
        self.index = np.load(cache_file, allow_pickle=True)["index"].tolist()
        print(f"Loaded sequence index from cache: {cache_file}")
        return True

    def _save_sequence_index_to_cache(self, cache_dir: Union[Path, str]):
        if cache_dir is None:
            return False
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)

        index_file = cache_dir / "index.json"
        if index_file.is_file():
            cache_index = json.loads(index_file.read_text())
        else:
            cache_index = {}
        dataset_key = self.__class__.__name__
        subset_key = ",".join(sorted(self.frames.include_subsets))
        rec_ids_key = (
            ",".join(map(str, sorted(self.frames.include_rec_ids)))
            if self.frames.include_rec_ids is not None
            else "All"
        )
        cam_ids_key = ",".join(map(str, sorted(self.frames.include_cam_ids)))
        seq_length_key = str(self.sequence_length)
        seq_stride_key = str(self.sequence_stride)
        seq_mfgap_key = str(self.max_frame_gap_us)
        cache_file = (
            cache_index.get(dataset_key, {})
            .get(subset_key, {})
            .get(rec_ids_key, {})
            .get(cam_ids_key, {})
            .get(seq_length_key, {})
            .get(seq_stride_key, {})
            .get(seq_mfgap_key, None)
        )
        if cache_file is None:
            cache_id = 1
            while (cache_dir / f"cache_{cache_id:04d}.npz").is_file():
                cache_id += 1
            cache_file = cache_dir / f"cache_{cache_id:04d}.npz"
        np.savez_compressed(cache_file, index=self.index)
        cache_update = {
            dataset_key: {
                subset_key: {
                    rec_ids_key: {
                        cam_ids_key: {
                            seq_length_key: {seq_stride_key: {seq_mfgap_key: cache_file.name}}
                        }
                    }
                }
            }
        }
        cache_index = nested_dict_update(cache_index, cache_update)
        index_file.write_text(json.dumps(cache_index))
        print(f"Saved sequence index to cache: {cache_file}")
        return True

    def apply_filter(
        self,
        require_rgb: bool = False,
        discard_rgb: bool = False,
        require_depth: bool = False,
        discard_depth: bool = False,
        require_objs: Set[int] = None,
        discard_objs: Union[bool, Set[int]] = False,
        require_hands: Set[int] = None,
        discard_hands: Union[bool, Set[int]] = False,
        require_handedness: Set[int] = None,
        discard_handedness: Union[bool, Set[int]] = False,
        require_gaze: Set[int] = None,
        discard_gaze: Union[bool, Set[int]] = False,
        evaluate_first_frame_only: bool = False,
        evaluate_last_frame_only: bool = False,
        check_all_paths: bool = False,
    ):
        """
        Filter dataset samples and discard the ones which do not contain the requested modalities or annotations.

        :return:
        """

        # check arguments
        if require_rgb and discard_rgb:
            print("Cannot simultaneously require and discard RGB modality. Defaulting to require.")
            discard_rgb = False
        if require_depth and discard_depth:
            print(
                "Cannot simultaneously require and discard depth modality. Defaulting to require."
            )
            discard_depth = False
        _verify_require_discard(
            require_objs,
            discard_objs,
            "Cannot simultaneously require and discard object IDs {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_hands,
            discard_hands,
            "Cannot simultaneously require and discard hands for staff IDs {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_handedness,
            discard_handedness,
            "Cannot simultaneously require and discard handedness {ids}. Defaulting to require.",
        )
        _verify_require_discard(
            require_gaze,
            discard_gaze,
            "Cannot simultaneously require and discard gaze for staff IDs {ids}. Defaulting to require.",
        )

        # filter samples based on given criteria
        for sample_idx in range(len(self)):
            # Copy requirement sets as they will be altered
            remaining_require_objs = (
                require_objs.copy() if isinstance(require_objs, Set) else require_objs
            )
            remaining_require_hands = (
                require_hands.copy() if isinstance(require_hands, Set) else require_hands
            )
            remaining_require_handedness = (
                require_handedness.copy()
                if isinstance(require_handedness, Set)
                else require_handedness
            )
            remaining_require_gaze = (
                require_gaze.copy() if isinstance(require_gaze, Set) else require_gaze
            )
            # get sample and frames to be evaluated
            sample = self[sample_idx]
            if evaluate_first_frame_only or evaluate_last_frame_only:
                orig_sample = sample
                sample = []
                if evaluate_first_frame_only:
                    sample.append(orig_sample[0])
                if evaluate_last_frame_only:
                    sample.append(orig_sample[-1])
            for frame in sample:
                # rgb and depth modality criteria have to be passed by each frame independently
                if not _frame_passes_filters(
                    frame,
                    require_rgb,
                    discard_rgb,
                    require_depth,
                    discard_depth,
                    check_all_paths=check_all_paths,
                    track_requirements=False,
                ):
                    self.index[sample_idx] = None
                    break
                # Annotation criteria can be passed partially by each frame
                _frame_passes_filters(
                    remaining_require_objs,
                    discard_objs,
                    remaining_require_hands,
                    discard_hands,
                    remaining_require_handedness,
                    discard_handedness,
                    remaining_require_gaze,
                    discard_gaze,
                    track_requirements=True,
                )

            if (
                len(remaining_require_objs) > 0
                or len(remaining_require_hands) > 0
                or len(remaining_require_handedness) > 0
                or len(remaining_require_gaze) > 0
            ):
                self.index[sample_idx] = None
        self.index = [idx for idx in self.index if idx is not None]
        # TODO remove unreferenced frames from self.frames, to save on memory
        if isinstance(discard_objs, Set):
            self.frames._filter_objects(discard_objs)

    def apply_filter_fn(self, fn: Callable[[Sequence[dict]], bool]):
        """
        Applies a filter function to the samples in the dataset, keeping only samples where the filter returns True.

        :param fn: filter function
        :return:
        """
        if self.index is not None:
            index = self.index
        else:
            index = range(len(self.frames))
        self.index = [i for i in index if fn(self[i])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return [self.frames[i] for i in self.index[idx]]
