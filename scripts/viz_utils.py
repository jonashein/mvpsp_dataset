from typing import Union
import numpy as np
from PIL import Image, ImageDraw
from mvpsp.dataset import MVPSP
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

MASK_COLORS = [tuple([int(255 * v) for v in mcolors.to_rgb(c)]) for c in mcolors.TABLEAU_COLORS][1:]


# utility function to draw hands
def draw_hands(img, img_frame, hand_frames=None, color=(0, 0, 255), thickness=6):
    img_cam_Rt = np.eye(4)
    if "cam_R_w2c" in img_frame and "cam_t_w2c" in img_frame:
        img_cam_Rt[:3, :3] = img_frame["cam_R_w2c"]
        img_cam_Rt[:3, 3:] = img_frame["cam_t_w2c"]
    if hand_frames is None:
        hand_frames = [img_frame]
    for hand_frame in hand_frames:
        if len(hand_frame.get("hands", [])) == 0:
            continue
        hand_cam_Rt = np.eye(4)
        if "cam_R_w2c" in hand_frame and "cam_t_w2c" in hand_frame:
            hand_cam_Rt[:3, :3] = hand_frame["cam_R_w2c"]
            hand_cam_Rt[:3, 3:] = hand_frame["cam_t_w2c"]
        # invert hand camera pose from w2c to c2w
        hand_cam_Rt = np.linalg.inv(hand_cam_Rt)
        for hand in hand_frame.get("hands", []):
            # project hand into image camera
            hand_joints_hom = np.hstack([hand["joint_positions"], np.ones((26, 1))])
            hand_pts_2d = img_frame["cam_K"] @ img_cam_Rt[:3, :] @ hand_cam_Rt @ hand_joints_hom.T
            hand_pts_2d = hand_pts_2d[:2, :] / hand_pts_2d[2:, :]
            hand_bones = (
                np.array(
                    [
                        hand_pts_2d[:, b[0]].tolist() + hand_pts_2d[:, b[1]].tolist()
                        for b in MVPSP.HAND_CONNECTIVITY
                    ]
                )
                .flatten()
                .tolist()
            )
            d = ImageDraw.Draw(img)
            d.line(hand_bones, fill=color, width=thickness)
            d.point(hand_pts_2d.flatten().tolist(), fill=color)
            # draw label
            top_left_pt = np.min(hand_pts_2d, axis=0).flatten().tolist()
            label = f"{MVPSP.STAFF_NAMES[hand['staff_id']]} {MVPSP.HANDEDNESS_NAMES[hand['handedness']]}"
            d.text(
                hand_pts_2d[:, 1],
                label,
                fill=(255, 255, 255),
                anchor="mt",
                stroke_width=1,
                font_size=int(0.05 * img.size[1]),
            )


# utility function to draw eye gazes
def draw_gazes(
    img,
    img_frame,
    gaze_frames=None,
    color=(255, 0, 0),
    thickness=8,
    gaze_length_mm=700,
    gaze_target_radius_px=20,
):
    img_cam_Rt = np.eye(4)
    if "cam_R_w2c" in img_frame and "cam_t_w2c" in img_frame:
        img_cam_Rt[:3, :3] = img_frame["cam_R_w2c"]
        img_cam_Rt[:3, 3:] = img_frame["cam_t_w2c"]
    if gaze_frames is None:
        gaze_frames = [img_frame]
    for gaze_frame in gaze_frames:
        if "eye_gaze" not in gaze_frame:
            continue
        gaze = gaze_frame["eye_gaze"]
        # project gaze into camera
        gaze_pts_hom = np.ones((4, 2))
        gaze_pts_hom[:3, 0:1] = gaze["origin"]
        gaze_unit_dir = gaze["direction"] / np.linalg.norm(gaze["direction"])
        gaze_pts_hom[:3, 1:2] = gaze["origin"] + gaze_length_mm * gaze_unit_dir
        gaze_cam_Rt = np.eye(4)
        if "cam_R_w2c" in gaze_frame and "cam_t_w2c" in gaze_frame:
            gaze_cam_Rt[:3, :3] = gaze_frame["cam_R_w2c"]
            gaze_cam_Rt[:3, 3:] = gaze_frame["cam_t_w2c"]
        # invert gaze camera pose from w2c to c2w
        gaze_cam_Rt = np.linalg.inv(gaze_cam_Rt)
        gaze_pts_2d = img_frame["cam_K"] @ img_cam_Rt[:3, :] @ gaze_cam_Rt @ gaze_pts_hom
        gaze_pts_2d = gaze_pts_2d[:2, :] / gaze_pts_2d[2:, :]
        gaze_src = (gaze_pts_2d[:, 0]).astype(int)
        gaze_dst = (gaze_pts_2d[:, 1]).astype(int)
        d = ImageDraw.Draw(img)
        d.line([tuple(gaze_src), tuple(gaze_dst)], fill=color, width=thickness)
        dst_radius = np.array([gaze_target_radius_px, gaze_target_radius_px])
        gaze_dst_bbox = [tuple(gaze_dst - dst_radius), tuple(gaze_dst + dst_radius)]
        d.ellipse(gaze_dst_bbox, outline=color, width=thickness)
        label = f"{MVPSP.STAFF_NAMES[gaze['staff_id']]}"
        d.text(
            gaze_dst,
            label,
            fill=(255, 255, 255),
            anchor="lt",
            stroke_width=1,
            font_size=int(0.05 * img.size[1]),
        )


# utility function to visualize a sample comprising one or multiple views, or a sequence
def show_sample(
    sample: Union[dict, list], show_hands_gaze_in_all_views: bool = False, y_labels: dict = {}
):
    if isinstance(sample, dict):
        sample = [sample]
    rows = len(sample)
    sample_has_hands_gaze = False
    for frame in sample:
        if "eye_gaze" in frame or len(frame.get("hands", [])) > 0:
            sample_has_hands_gaze = True
            break
    if not sample_has_hands_gaze:
        show_hands_gaze_in_all_views = False
    cols = 5 if sample_has_hands_gaze else 4
    plt.rcParams["figure.figsize"] = (cols * 5, rows * 5)
    plt.axis("off")
    for j, frame in enumerate(sample):
        im_rgb = Image.open(frame["rgb_path"]).convert("RGBA")
        if "depth_path" in frame:
            im_depth = Image.open(frame["depth_path"])
        else:
            im_depth = None
        im_mask_overlay = im_rgb.copy()
        im_mask_visib_overlay = im_rgb.copy()
        for i, obj in enumerate(frame.get("objects", [])):
            im_obj_mask = Image.open(obj["obj_mask_path"]).convert("L").point(lambda i: 0.66 * i)
            im_mask_overlay.paste(MASK_COLORS[i % len(MASK_COLORS)], mask=im_obj_mask)
            if "obj_mask_visib_path" in obj:
                im_obj_mask_visib = (
                    Image.open(obj["obj_mask_visib_path"]).convert("L").point(lambda i: 0.66 * i)
                )
                im_mask_visib_overlay.paste(
                    MASK_COLORS[i % len(MASK_COLORS)], mask=im_obj_mask_visib
                )
        # add bounding boxes and class labels
        d = ImageDraw.Draw(im_mask_overlay)
        for i, obj in enumerate(frame.get("objects", [])):
            # stored bbox has format (x, y, width, height); convert to (x0, y0, x1, y1)
            color = MASK_COLORS[i % len(MASK_COLORS)]
            bbox = list(obj["obj_bbox"])
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            d.rectangle(bbox, outline=color, width=8)
            d.text(
                bbox[:2],
                f"{MVPSP.OBJECT_NAMES[obj['obj_id']]} ID: {obj['inst_id']}",
                fill=(255, 255, 255),
                anchor="lb",
                stroke_width=2,
                font_size=int(0.05 * im_rgb.size[1]),
            )
        # show hands and eye gaze
        has_hands_or_gaze = (
            show_hands_gaze_in_all_views or "eye_gaze" in frame or len(frame.get("hands", [])) > 0
        )
        if has_hands_or_gaze:
            im_hands_gaze = im_rgb.copy()
            draw_gazes(
                im_hands_gaze, frame, gaze_frames=sample if show_hands_gaze_in_all_views else None
            )
            draw_hands(
                im_hands_gaze, frame, hand_frames=sample if show_hands_gaze_in_all_views else None
            )
        else:
            im_hands_gaze = None

        plt.subplot(rows, cols, cols * j + 1)
        if j == 0:
            plt.title("RGB")
        plt.ylabel(y_labels.get(j, f"Frame {j}"))
        plt.imshow(im_rgb)
        if j == 0:
            plt.subplot(rows, cols, cols * j + 2)
            plt.title("Depth")
        if im_depth is not None:
            plt.subplot(rows, cols, cols * j + 2)
            plt.imshow(im_depth)
        plt.subplot(rows, cols, cols * j + 3)
        if j == 0:
            plt.title("Mask Overlay")
        plt.imshow(im_mask_overlay)
        plt.subplot(rows, cols, cols * j + 4)
        if j == 0:
            plt.title("Visible Mask Overlay")
        plt.imshow(im_mask_visib_overlay)
        if j == 0 and cols >= 5:
            plt.subplot(rows, cols, cols * j + 5)
            plt.title("Hands / Eye Gaze")
        if has_hands_or_gaze:
            plt.subplot(rows, cols, cols * j + 5)
            plt.imshow(im_hands_gaze)

    # disable axes ticks
    for i in range(1, rows * cols + 1):
        ax = plt.subplot(rows, cols, i)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    plt.show()
