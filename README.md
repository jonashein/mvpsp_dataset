# MVPSP Dataset Tools
Tools to download and visualize the MVPSP dataset.

# Download
After cloning this repo, run the `1_download.py <download_dir>` script to automatically download the dataset to the location of your choice.

# Structure
The dataset generally follows the [BOP dataset format](https://github.com/thodan/bop_toolkit/blob/8facae674f752f9680c4d2a75bc951a6fd947f1e/docs/bop_datasets_format.md).
There are some minor changes to account for the multi-camera video content, which are described below:
- The `scene_id` consists of a 3-digit `recording_id` followed by a 3-digit `camera_id`. For example, the scene_id 003005 contains images from recording 3 and camera 5. camera_ids between 0 to 4 are assigned to static cameras, while camera_ids 5 and 6 are assigned to the two dynamic cameras.
- Each scene has a `scene_timings.json` file, which contains the microsecond timestamp of each frame. These timestamps are synchronized across all cameras of the same recording, and can be used to compute the time span between the exposure window of any two frames.
- Since the captured scenes are dynamic, a meaningful input for multi-view models is a set of frames with the same recording_id and similar timestamps (as per `scene_timings.json`).
As described in the paper, we tested all baselines on selected sets of frames with a maximum pairwise temporal offset of 8 milliseconds between the exposure windows. 
For convenience reasons, we provide frame correspondences for the test sets in form of an additional attribute `multiview_id` inside the `scene_camera.json`. Frames with at most 8 milliseconds offset are assigned the same multiview_id.
Thus, the multiview_id identifies synchronized frames across all cameras of the same recording.
- The camera intrinsics for the surgical wet lab subsets are stored in `camera_kinect.json` (static) and `camera_hololens.json` (dynamic); the intrinsics for the OR-X subset are stored in `camera_kinect_4K.json`

## Notes
- The dataset contains images of three different resolutions. Some implementations assume a constant image resolution and may break when loading this dataset. Make sure to load the correct intrinsics.

# Visualizations
