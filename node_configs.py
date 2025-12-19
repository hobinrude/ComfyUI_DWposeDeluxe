# ComfyUI_DWposeDeluxe/node_configs.py

class DWposeNodeBase:

    # Base class for DWpose nodes, providing a method to fetch descriptions.

    SHORT_DESCRIPTION = "No short description available."
    LONG_DESCRIPTION = "No detailed description available."

    @classmethod

    # Fetches the node's descriptions (short and long). Returns a tuple: (long_description, short_description).
    def get_description(cls):
        node_config = get_node_config(cls)
        if node_config:
            long_desc = node_config.get('long_description', cls.LONG_DESCRIPTION)
            short_desc = node_config.get('short_description', cls.SHORT_DESCRIPTION)
            return long_desc, short_desc
        # Fallback descriptions if not found in the config
        return cls.LONG_DESCRIPTION, cls.SHORT_DESCRIPTION

class NodeConfig:
    def __init__(self, long_description, short_description=""):
        self.long_description = long_description
        self.short_description = short_description

    def get(self, key, default=None):
        if key == 'long_description':
            return self.long_description
        if key == 'short_description':
            return self.short_description
        return default

NODE_CONFIGS = {}


def add_node_config(node_name, config):
    NODE_CONFIGS[node_name] = config


def get_node_config(node_class):
    return NODE_CONFIGS.get(node_class.__name__)


add_node_config("KeypointDiff", NodeConfig(
    short_description="Compares two keypoint datasets and reports differences.",
    long_description="""
**Function:** Compares every (x, y, confidence) value between two keypoint datasets of the same length.

### Inputs
- **keypoints_1:** The first keypoint dataset (baseline).
- **keypoints_2:** The second keypoint dataset (to compare).

### Outputs
- **diff_info:** A string report detailing frame count, people count, and value mismatches.

### Notes
- Automatically ignores minor rounding differences (delta < 0.0001).
- Logs a summary to the console: `logger.info` for a perfect match, `logger.warning` if discrepancies are found.
- Useful for verifying data integrity after multiple conversions or reversals.
    """
))

add_node_config("DWposeDeluxeNode", NodeConfig(
    short_description="Main node for DWpose, detects and renders human poses.",
    long_description="""
**Function:** The main node that detects and renders human poses from an input image sequence.

### Inputs
- **image:** The source image frames to process.
- **frame_count:** The total number of frames in the sequence.
- **audio:** Optional audio to pass through with the video.
- **video_info:** Optional video metadata for frame rate handling.
- **custom_options:** Optional advanced rendering and detection parameters.

### Parameters
- **poses_to_detect:** Max number of poses to detect per frame (by bbox area).
- **show_body:** Toggle visibility of the main body skeleton.
- **show_feet:** Toggle visibility of the foot points and bones.
- **show_face:** Toggle visibility of face keypoints.
- **show_hands:** Toggle visibility of hand points and bones.
- **provider_type:** Hardware backend for estimation (CPU or GPU/TensorRT).
- **precision:** Floating point precision for GPU models.
- **detector_model:** The person detector model to use.
- **estimator_model:** The pose estimator model to use.
- **save_keypoints:** Save the output keypoint data to a file.

### Outputs
- **pose_image:** The rendered pose skeletons on a black background.
- **blend_image:** The pose skeletons blended over the source image.
- **face_image:** A horizontal atlas of cropped faces (left to right).
- **source_image:** The original input image, passed through for easy noodling.
- **audio:** The original input audio, passed through from video (up)load node.
- **frame_rate:** The original video frame rate, passed through from video_info.
- **keypoints:** The raw detected keypoint data in JSON format.

### Notes
- Max poses_to_detect is far from perfect and can be glitchy on high action
videos with multiple foreground poses but works like charm to exclude unwanted
background photo-bombers. A more robust function coming soon!
- The face_crop images are concatenated in left to right order, so will
re-arrange if characters swap places during video sequence. Fix coming soon!
Workaround: range_from_batch, crop and re-batch nodes are your friends.
    """
))

add_node_config("CustomOptions", NodeConfig(
    short_description="Advanced controls for estimation appearance and thresholds.",
    long_description="""
**Function:** Provides advanced control over the appearance and thresholds of the pose estimation.

### Inputs
*None*

### Parameters
- **body_dot_size_modifier:** Adjusts the size of body keypoint dots.
- **body_line_thickness_modifier:** Adjusts the thickness of body bones.
- **hand_dot_size_modifier:** Adjusts the size of hand keypoint dots.
- **hand_line_thickness_modifier:** Adjusts the thickness of hand bones.
- **face_dot_size_modifier:** Adjusts the size of face keypoint dots.
- **pose_threshold:** Minimum confidence for a person to be detected.
- **body_threshold:** Minimum confidence for body keypoints.
- **face_threshold:** Minimum confidence for face keypoints.
- **hand_threshold:** Minimum confidence for hand keypoints.
- **face_padding:** Amount of padding to add around cropped faces.
- **neck_validity:** Confidence threshold for the neck connection validity.
- **nms_threshold:** Non-Maximum Suppression threshold for bounding boxes.
- **score_threshold:** Score threshold for detector bounding boxes.

### Outputs
- **custom_options:** A package of options to be connected to the main Estimator node.
    """
))

add_node_config("FrameNumberNode", NodeConfig(
    short_description="Overlays frame numbers on images.",
    long_description="""
**Function:** Overlays the frame number onto each image in a sequence.

### Inputs
- **image:** The image frames to apply numbering to.

### Parameters
- **skip_first_frames:** An offset for the starting frame number.
- **print_every_nth:** The value to increment frame numbers by.
- **frame_numbers:** Enable or disable the numbering overlay.
- **number_position:** Position of the number on the frame.
- **font_size:** Size of the font for the frame number.
- **font_name:** The `.ttf` font file to use for rendering.
- **overlay_opacity:** Transparency of the numbering overlay.

### Outputs
- **image:** The image sequence with frame numbers applied.
    """
))

add_node_config("KeypointConverter", NodeConfig(
    short_description="Converts, cleans, remaps and saves pose keypoint data.",
    long_description="""
**Function:** Converts, cleans, remaps and saves pose keypoint data between different formats and coordinate systems.

### Inputs
- **pose_keypoints:** The keypoint data to process.

### Parameters
- **padding_mode:** Determine canvas size if missing (`auto-padding` or `no-padding`).
- **confidence_threshold:** Filters out keypoints below this confidence value.
- **reset_confidence:** Sets confidence of valid points to 1.0, and invalid to 0.0.
- **force_normalized:** Ensures the output coordinates are relative (normalized 0-1).
- **pretty_json:** Outputs the keypoint data with a human-readable JSON indentation.
- **save_output:** Saves the converted keypoint data to a file.

### Outputs
- **pose_keypoints:** The processed keypoint data.
- **input_keypoint_info:** Detailed information about the input keypoints.

### Notes
- The **pretty_json** format creates very large files (whitespaces, eols) so use only
on small batches when needed for manual pose debugging, etc. You have been warned.
    """
))

add_node_config("KeypointPrinter", NodeConfig(
    short_description="Renders pose skeletons from raw keypoint data.",
    long_description="""
**Function:** Renders an image of the pose skeletons from provided raw keypoint data.

### Inputs
- **keypoints:** The keypoint data to render onto black canvas.
- **custom_options:** Optional advanced rendering parameters.

### Parameters
- **poses_to_print:** The maximum number of poses to render from the data.
- **show_body:** Toggle visibility of the main body skeleton.
- **show_feet:** Toggle visibility of the foot points and bones.
- **show_face:** Toggle visibility of face keypoints.
- **show_hands:** Toggle visibility of hand points and bones.

### Outputs
- **pose_image:** The rendered pose skeletons on a black background.
    """
))

add_node_config("LoadPoseKeypoints", NodeConfig(
    short_description="Loads pose keypoint data from a JSON file.",
    long_description="""
**Function:** (Up)loads pose keypoint data from a JSON file from the ComfyUI/input directory.

### Inputs
- **upload_button:** To select and uplaod JSON file from any local path.

### Parameters
- **file:** The JSON file to load.

### Outputs
- **pose_keypoints:** The loaded keypoint data.
- **keypoint_info:** A string containing detailed information about the loaded file.
    """
))

add_node_config("PoseInterpolation", NodeConfig(
    short_description="Generates interpolated poses between two source poses.",
    long_description="""
**Function:** Generates a sequence of interpolated poses between two source poses.

### Inputs
- **keypoints_1:** The starting keypoint data.
- **keypoints_2:** The ending keypoint data.

### Parameters
- **keypoints_1_frame:** The frame number to use from the first keypoint input.
- **keypoints_2_frame:** The frame number to use from the second keypoint input.
- **frame_count:** The number of interpolated frames to generate.

### Outputs
- **keypoints:** The resulting sequence of interpolated keypoint data.

### Notes
- To auto-select last frame from batch use '-1' as frame number.
    """
))

add_node_config("PoseResize", NodeConfig(
    short_description="Rescales, crops and pads pose keypoint coordinates.",
    long_description="""
**Function:** Loss-less rescale, crop, move and add padding to pose keypoint coordinates.

### Inputs
- **pose_keypoints:** The keypoint data to resize.

### Parameters
- **width:** Initial width re-scale for the pose content.
- **height:** Initial height re-scale for the pose content.
- **resize_method:** (stretch,crop/pad) Type of canvas transform.
- **left, right, top, bottom:** Specific padding for each side.
- **extra_padding:** Uniform padding added to all sides.
- **x_offset:** Moves all keypoints left or right.
- **y_offset:** Moves all keypoints up or down.
- **divisible_by:** Rounds target W and H up to nearest multiple of value.

### Outputs
- **pose_keypoints:** The rescaled and padded keypoint data.

### Notes
- Use '0' in either W or H to keep original canvas aspect ratio.
- Negative padding values work as cropping.
    """
))

add_node_config("KeypointRangeFromBatch", NodeConfig(
    short_description="Extracts a subset of keypoint frames.",
    long_description="""
**Function:** Extracts a range of frames from a keypoint batch.

### Inputs
- **pose_keypoints:** The keypoint sequence to process.

### Parameters
- **start_frame:** The starting frame index (0-based).
- **frame_count:** The number of frames to extract.

### Outputs
- **pose_keypoints:** The extracted subset of keypoints.
    """
))

add_node_config("BatchKeypoints", NodeConfig(
    short_description="Concatenates two keypoint batches.",
    long_description="""
**Function:** Appends frames from the second keypoint batch to the end of the first one.

### Inputs
- **keypoints_1:** The first keypoint batch.
- **keypoints_2:** The second keypoint batch (appended to first).

### Outputs
- **pose_keypoints:** The combined keypoint sequence.

### Notes
- Both batches should have the same canvas dimensions.
    """
))

add_node_config("MergeKeypoints", NodeConfig(
    short_description="Merges people from two keypoint batches into single frames.",
    long_description="""
**Function:** Combines pose detections from two inputs into a single canvas for each frame.

### Inputs
- **keypoints_1:** The first keypoint dataset.
- **keypoints_2:** The second keypoint dataset (merged into first).

### Outputs
- **pose_keypoints:** The merged keypoint data containing people from both inputs.

### Notes
- Inputs must have the same canvas size and same number of frames.
- Coordinate format (absolute/normalized) will be matched to keypoints_1.
    """
))

add_node_config("CherryPickerFrames", NodeConfig(
    short_description="Selects best keypoints from over-sampled data.",
    long_description="""
**Function:** Reduces an over-sampled (high FPS) keypoint dataset by selecting the highest-confidence individual keypoints from each frame chunk.

### Inputs
- **pose_keypoints:** The keypoint dataset (e.g., from a 2x VFI source).

### Parameters
- **factor:** The downsampling factor (e.g., 2 for 2x source to 1x output).

### Outputs
- **pose_keypoints:** The downsampled keypoint data with cherry-picked confident points.

### Notes
- This node is designed to combat jitter and noise from detection models by leveraging over-sampling.
- It is ideal for use after generating keypoints from a VFI (Video Frame Interpolation) output.
- Logs a summary of confidence gain per body point upon completion.
    """
))

add_node_config("ReverseKeypoints", NodeConfig(
    short_description="Reverses a keypoint batch.",
    long_description="""
**Function:** Reverses the temporal order of frames in a keypoint batch.

### Inputs
- **pose_keypoints:** The keypoint sequence to reverse.

### Outputs
- **pose_keypoints:** The reversed keypoint sequence.
    """
))

add_node_config("CherryPickerTwoInputs", NodeConfig(
    short_description="Cherry-picks best keypoints between two datasets.",
    long_description="""
**Function:** Takes two keypoint datasets of the same length and creates a new one by choosing the most confident individual keypoints from either input for each frame.

### Inputs
- **keypoints_1:** The first keypoint dataset (baseline for gain summary).
- **keypoints_2:** The second keypoint dataset.

### Outputs
- **pose_keypoints:** The merged keypoint data with the best points from both inputs.

### Notes
- Inputs must have the same number of frames.
- Useful for merging results from different detection/estimation models.
- Logs a summary of confidence gain per body point upon completion.
    """
))
