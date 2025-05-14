# Multi-View Pedicle Screw Placement (MVPSP) Dataset
The MVPSP dataset is a multi-view RGB-D video dataset showing the pedicle screw placement step of spinal surgery.
The dataset comprises five static cameras and two egocentric cameras, worn by the surgeon and an assistant.
We provide a total of 23 recordings, comprising 17 training and 4 test sequences captured in a surgical wet lab, and 2 test sequences captured in a real operating room. 

This repo contains scripts to download the MVPSP dataset, as well as examples on how to read and visualize it.

# Download
After cloning this repo, run `download_mvpsp.sh -d -u -r datasets/mvpsp/` to automatically download (`-d`) and unpack (`-u`) the dataset, and to remove (`-r`) the compressed archives.
The uncompressed dataset has a size of about 2.9 TiB. We provide an overview of the subset sizes below.

| Subset       | Compressed | Uncompressed |
|--------------|-----------:|-------------:|
| train_wetlab |    1.7 TiB |      2.0 TiB |
| train_synth  |    2.2 GiB |      3.2 GiB |
| train_pbr    |   20.7 GiB |     22.9 GiB |
| test_wetlab  |  677.3 GiB |    803.6 GiB |
| test_orx     |   79.6 GiB |     96.0 GiB |
| TOTAL        |    2.4 TiB |      2.9 TiB |

# Demo
In `scripts/demo_visualization.ipynb` we provide a simple visualization demo for all modalities.
It contains reference implementations showcasing how to load monocular and multi-view samples, as well as monocular video sequences.

# Structure
This dataset generally follows the [BOP dataset scenewise format](https://github.com/thodan/bop_toolkit/blob/8facae674f752f9680c4d2a75bc951a6fd947f1e/docs/bop_datasets_format.md). 
After the download and extraction is completed, the dataset should have this structure:
```
README.md (this file)
scripts
├─ download.sh  
├─ demo_visualization.ipynb  
datasets
├─ mvpsp
│  ├─ camera_kinect.json
│  ├─ camera_kinect_4K.json
│  ├─ camera_hololens2.json
│  ├─ models[_eval]
│  │  ├─ models_info.json
│  │  ├─ obj_<obj_id>.ply
│  ├─ train|test[_SUBSET]
│  │  ├─ <recording_id><camera_id>
│  │  │  ├─ scene_camera.json
│  │  │  ├─ scene_gt.json
│  │  │  ├─ scene_gt_info.json
│  │  │  ├─ [scene_hand_eye.json]
│  │  │  ├─ depth
│  │  │  ├─ mask
│  │  │  ├─ mask_visib
│  │  │  ├─ rgb
```
 
There are some minor changes to account for the multi-camera video content, which are summarized below:

- The `scene_id` consists of a 3-digit `recording_id` followed by a 3-digit `camera_id`. For example, the scene_id 003005 contains images from recording 3 and camera 5.
- The `scene_camera.json` contains a timestamp `time_us` for each frame. 
The timestamps are synchronized across all scenes with the same `recording_id`, and can be used to compute the time span between the exposure windows of any two frames.
- The camera intrinsics for the surgical wet lab subsets are stored in `camera_kinect.json` (static) and `camera_hololens.json` (dynamic); the intrinsics for the OR-X subset are stored in `camera_kinect_4K.json`
- For egocentric cameras we provide hand pose and eye gaze information in `scene_hand_eye.json`. Detailed information can be found below.

### Data Formats in Detail

#### Units
This dataset consistently uses millimeters for translations, depth maps, and 3D models. Timestamps are expressed in microseconds.

#### scene_camera.json:
```
{
    <frame_id>: {
        cam_K: <3x3 camera intrinsics, flattened in row-major format>, 
        cam_R_w2c: <3x3 rotation matrix from world to camera coordinate frame, flattened in row-major format>,
        cam_t_w2c: <3x1 translation vector from world to camera coordinate frame, flattened in row-major format>,
        time_us: <frame timestamp in microseconds>,
        depth_scale: <scaling factor to convert depth maps to mm>,
    },
    ...
}
```

#### scene_hand_eye.json
For scenes captured from the egocentric perspective (i.e. `camera_id`'s 5 and 6) the `scene_hand_eye.json` provides the 3D hand joint positions as well as the eye gaze origin and direction. 
All positions and directions are expressed in the camera coordinate frame.
```
{ 
    <frame_id>: {
        [left_hand_joints]: <26x3 matrix containing the 3D position of 26 left hand joints, flattened in row-major format>, 
        [right_hand_joints]: <26x3 matrix containing the 3D position of 26 right hand joints, flattened in row-major format>, 
        [eye_gaze]: {
            origin: <3D position of the eye gaze origin>,
            direction: <unit 3D direction vector>,
        }
    },
    ...
}
```
Please note that the hand pose and eye gaze information is provided as recognized by the Hololens 2 and without further refinement.

# Acknowledgements
If our dataset is relevant for your research, please consider citing our paper:
```
@article{hein_next-generation_2025,
	title = {Next-generation surgical navigation: Marker-less multi-view 6DoF pose estimation of surgical instruments},
	issn = {1361-8415},
	url = {https://www.sciencedirect.com/science/article/pii/S1361841525001604},
	doi = {10.1016/j.media.2025.103613},
	shorttitle = {Next-generation surgical navigation},
	abstract = {State-of-the-art research of traditional computer vision is increasingly leveraged in the surgical domain. A particular focus in computer-assisted surgery is to replace marker-based tracking systems for instrument localization with pure image-based 6DoF pose estimation using deep-learning methods. However, state-of-the-art single-view pose estimation methods do not yet meet the accuracy required for surgical navigation. In this context, we investigate the benefits of multi-view setups for highly accurate and occlusion-robust 6DoF pose estimation of surgical instruments and derive recommendations for an ideal camera system that addresses the challenges in the operating room. Our contributions are threefold. First, we present a multi-view {RGB}-D video dataset of ex-vivo spine surgeries, captured with static and head-mounted cameras and including rich annotations for surgeon, instruments, and patient anatomy. Second, we perform an extensive evaluation of three state-of-the-art single-view and multi-view pose estimation methods, analyzing the impact of camera quantities and positioning, limited real-world data, and static, hybrid, or fully mobile camera setups on the pose accuracy, occlusion robustness, and generalizability. Third, we design a multi-camera system for marker-less surgical instrument tracking, achieving an average position error of 1.01mm and orientation error of 0.89° for a surgical drill, and 2.79mm and 3.33° for a screwdriver under optimal conditions. Our results demonstrate that marker-less tracking of surgical instruments is becoming a feasible alternative to existing marker-based systems.},
	pages = {103613},
	journaltitle = {Medical Image Analysis},
	author = {Hein, Jonas and Cavalcanti, Nicola and Suter, Daniel and Zingg, Lukas and Carrillo, Fabio and Calvet, Lilian and Farshad, Mazda and Navab, Nassir and Pollefeys, Marc and Fürnstahl, Philipp},
	keywords = {Deep Learning, Marker-less tracking, Multi-view {RGB}-D video dataset, Object pose estimation, Surgical instruments, Surgical navigation},
}
```