# ComfyUI_DWposeDeluxe/dwpose/util.py

import math
import numpy as np
import matplotlib.colors
import cv2

eps = 0.01


def smart_resize(x, s): Ht, Wt = s; Ho, Wo = x.shape[:2]; k = float(Ht + Wt) / float(Ho + Wo); return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]; w = img.shape[1]
    pad = [0, 0, 0 if (h % stride == 0) else stride - (h % stride), 0 if (w % stride == 0) else stride - (w % stride)]
    img_padded = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=padValue)
    return img_padded, pad


def draw_bodypose_with_feet(canvas, candidate, subset, options={}):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    thickness_mod = options.get('body_line_thickness_modifier', 0)
    dot_size_mod = options.get('body_dot_size_modifier', 0)
    stickwidth = max(1, 4 + thickness_mod)
    dot_radius = max(1, 4 + dot_size_mod)

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18],
        
        [11, 24], [14, 21], [14, 19], [19, 20], [11, 22], [22, 23] # Feet
    ]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85], [85, 0, 255], [85, 0, 255],
        [85, 0, 255], [0, 170, 255], [0, 170, 255], [0, 170, 255],
    ]

    # Define ankle and foot point indices for color mapping (0-based)
    R_ANKLE_IDX = 10
    L_ANKLE_IDX = 13
    # Joint indices for left foot (including LAnkle)
    LEFT_FOOT_JOINTS = {L_ANKLE_IDX, 18, 19, 20} # LAnkle, LBigToe, LSmToe, LHeel
    # Joint indices for right foot (including RAnkle)
    RIGHT_FOOT_JOINTS = {R_ANKLE_IDX, 21, 22, 23} # RAnkle, RBigToe, RSmToe, RHeel

    # Draw bones
    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        limb_indices = np.array(limb) - 1
        p1_idx, p2_idx = limb_indices[0], limb_indices[1]

        # Determine bone color
        color_to_use = None
        if (p1_idx in LEFT_FOOT_JOINTS or p2_idx in LEFT_FOOT_JOINTS):
            color_to_use = colors[L_ANKLE_IDX]
        elif (p1_idx in RIGHT_FOOT_JOINTS or p2_idx in RIGHT_FOOT_JOINTS):
            color_to_use = colors[R_ANKLE_IDX]
        else:
            # For non-foot bones, use the sequential color
            color_to_use = colors[i]

        cv_color = (int(color_to_use[0]), int(color_to_use[1]), int(color_to_use[2]))

        for n in range(len(subset)):
            if np.any(limb_indices >= subset.shape[1]): continue
            index = subset[n][limb_indices]
            if -1 in index or np.any(index >= len(candidate)): continue

            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)): continue

            mX, mY = np.mean(X), np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if length < 1: continue

            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, cv_color)

    canvas = (canvas * 0.6).astype(np.uint8)

    # Draw Dots
    num_total_points = 24
    for i in range(num_total_points):
        # Determine dot color
        color_to_use = None
        if i in LEFT_FOOT_JOINTS:
            color_to_use = colors[L_ANKLE_IDX]
        elif i in RIGHT_FOOT_JOINTS:
            color_to_use = colors[R_ANKLE_IDX]
        else:
            # For non-foot points, use the sequential color
            color_to_use = colors[i]

        cv_color = (int(color_to_use[0]), int(color_to_use[1]), int(color_to_use[2]))

        for n in range(len(subset)):
            if i >= subset.shape[1]: continue
            index = int(subset[n][i])
            if index == -1 or index >= len(candidate): continue
            x, y = candidate[index][0:2]
            x, y = int(x * W), int(y * H)
            cv2.circle(canvas, (int(x), int(y)), dot_radius, cv_color, thickness=-1)
    return canvas


def draw_bodypose_without_feet(canvas, candidate, subset, options={}):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    thickness_mod = options.get('body_line_thickness_modifier', 0)
    dot_size_mod = options.get('body_dot_size_modifier', 0)
    stickwidth = max(1, 4 + thickness_mod)
    dot_radius = max(1, 4 + dot_size_mod)

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), dot_radius, colors[i], thickness=-1)
    return canvas

def draw_bodypose(canvas, candidate, subset, show_feet, options={}):
    if show_feet:
        return draw_bodypose_with_feet(canvas, candidate, subset, options)
    else:
        return draw_bodypose_without_feet(canvas, candidate, subset, options)


def draw_handpose(canvas, all_hand_peaks, options={}):
    H, W, C = canvas.shape

    thickness_mod = options.get('hand_line_thickness_modifier', 0); dot_size_mod = options.get('hand_dot_size_modifier', 0)
    line_thickness = max(1, 2 + thickness_mod); dot_radius = max(1, 4 + dot_size_mod)
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for hand_idx, peaks in enumerate(all_hand_peaks):
        peaks = np.array(peaks)

        if peaks.ndim != 2 or peaks.shape[1] != 2: continue
        num_lines_drawn = 0; num_dots_drawn = 0
        for ie, e in enumerate(edges):
             if np.any(np.array(e) >= len(peaks)): continue
             p1_idx, p2_idx = e; x1_norm, y1_norm = peaks[p1_idx]; x2_norm, y2_norm = peaks[p2_idx]
             if x1_norm > eps and y1_norm > eps and x2_norm > eps and y2_norm > eps:
                 x1, y1 = int(x1_norm * W), int(y1_norm * H); x2, y2 = int(x2_norm * W), int(y2_norm * H)
                 hsv_color = [ie / float(len(edges)), 1.0, 1.0]; rgb_color_float = matplotlib.colors.hsv_to_rgb(hsv_color) * 255
                 bgr_color_int = (int(rgb_color_float[2]), int(rgb_color_float[1]), int(rgb_color_float[0]))
                 cv2.line(canvas, (x1, y1), (x2, y2), bgr_color_int, thickness=line_thickness); num_lines_drawn += 1
        for i, keypoint in enumerate(peaks):
            x_norm, y_norm = keypoint
            if x_norm > eps and y_norm > eps:
                 x, y = int(x_norm * W), int(y_norm * H)
                 cv2.circle(canvas, (x, y), dot_radius, (0, 0, 255), thickness=-1); num_dots_drawn += 1
    return canvas


def draw_facepose(canvas, all_lmks, options={}):
    H, W, C = canvas.shape
    dot_size_mod = options.get('face_dot_size_modifier', 0); dot_radius = max(1, 3 + dot_size_mod)
    for lmks in all_lmks:
        lmks = np.array(lmks)
        if lmks.ndim != 2 or lmks.shape[1] != 2: continue
        for lmk in lmks:
            x, y = lmk
            if x > eps and y > eps:
                 x, y = int(x * W), int(y * H)
                 cv2.circle(canvas, (x, y), dot_radius, (255, 255, 255), thickness=-1)
    return canvas