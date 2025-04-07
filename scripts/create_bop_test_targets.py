from pathlib import Path
from mvpsp.dataset import MVPSP, MvpspMultiviewDataset

dataset_root = Path("/media/heinj/OR-X_DATA/datasets/bop/mvpsp/")
assert dataset_root.exists(), f"Please place (a link to) the dataset at {dataset_root.absolute()}."

for subset in MVPSP.TEST_SUBSETS:
    # Load the dataset as single views, list the number of samples and select a random sample
    dataset = MvpspMultiviewDataset(
        dataset_root, is_test=True, include_subsets=subset, cache_dir=None
    )
    dataset.apply_filter(
        require_rgb=True,
        require_depth=True,
        require_objs={MVPSP.ANY_INSTANCE},
        discard_objs={
            obj_id
            for obj_id in MVPSP.OBJECTS
            if obj_id not in [MVPSP.OBJECT_DRILL, MVPSP.OBJECT_SCREWDRIVER]
        },
    )
    print(f"{subset} has {len(dataset)} RGB frames with drill or screwdriver annotation")
    targets_file = dataset_root / subset / "test_targets.json"
    targets = dataset.as_bop_targets(targets_file)
    print(f"Saved {len(targets)} targets to {targets_file}")
