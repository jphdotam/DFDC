import cv2
import math
import numpy as np
import skimage.measure
from PIL import Image

TWO_FRAME_OVERLAP = False
MIN_FRAMES_FOR_FACE = 30
MAX_FRAMES_FOR_FACE = 300


def load_video(filename, every_n_frames=None, specific_frames=None, to_rgb=True, rescale=None, inc_pil=False,
               max_frames=None):
    """Loads a video.
    Called by:

    1) The finding faces algorithm where it pulls a frame every FACE_FRAMES frames up to MAX_FRAMES_TO_LOAD at a scale of FACEDETECTION_DOWNSAMPLE, and then half that if there's a CUDA memory error.

    2) The inference loop where it pulls EVERY frame up to a certain amount which it the last needed frame for each face for that video"""
    assert every_n_frames or specific_frames, "Must supply either every n_frames or specific_frames"
    assert bool(every_n_frames) != bool(
        specific_frames), "Supply either 'every_n_frames' or 'specific_frames', not both"

    cap = cv2.VideoCapture(filename)
    n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width_out = int(width_in * rescale) if rescale else width_in
    height_out = int(height_in * rescale) if rescale else height_in

    if max_frames:
        n_frames_in = min(n_frames_in, max_frames)

    if every_n_frames:
        n_frames_out = n_frames_in // every_n_frames
        specific_frames = [i * every_n_frames for i in range(n_frames_out)]
    else:
        n_frames_out = len(specific_frames)

    out_pil = []

    out_video = np.empty((n_frames_out, height_out, width_out, 3), np.dtype('uint8'))

    i_frame_in = 0
    i_frame_out = 0
    ret = True

    while (i_frame_in < n_frames_in and ret):
        ret, frame_in = cap.read()

        if i_frame_in not in specific_frames:
            i_frame_in += 1
            continue

        try:
            if rescale:
                frame_in = cv2.resize(frame_in, (width_out, height_out))
            if to_rgb:
                frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error for frame {i_frame_in} for video {filename}: {e}; using 0s")
            frame_in = np.zeros((height_out, width_out, 3))

        out_video[i_frame_out] = frame_in
        i_frame_out += 1

        if inc_pil:
            try:  # https://www.kaggle.com/zaharch/public-test-errors
                pil_img = Image.fromarray(frame_in)
            except Exception as e:
                print(f"Using a blank frame for video {filename} frame {i_frame_in} as error {e}")
                pil_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))  # Use a blank frame
            out_pil.append(pil_img)

        i_frame_in += 1

    cap.release()

    if inc_pil:
        return out_video, out_pil
    else:
        return out_video


def get_roi_for_each_face(faces_by_frame, video_shape, temporal_upsample, downsample_for_calcs=1, upsample=1):
    if not downsample_for_calcs:
        downsample_for_calcs = 1

    # Create boolean face array
    frames_video, rows_video, cols_video, channels_video = video_shape
    frames_video = math.ceil(frames_video)
    if downsample_for_calcs != 1:
        boolean_face_3d = np.zeros(
            (frames_video, rows_video // downsample_for_calcs, cols_video // downsample_for_calcs),
            dtype=np.bool)  # Remove colour channel
    else:
        boolean_face_3d = np.zeros((frames_video, rows_video, cols_video), dtype=np.bool)  # Remove colour channel
    for i_frame, faces in enumerate(faces_by_frame):
        if faces is not None:  # May not be a face in the frame
            for face in faces:
                left, top, right, bottom = face
                if downsample_for_calcs != 1:
                    left = round(left / downsample_for_calcs)
                    top = round(top / downsample_for_calcs)
                    right = round(right / downsample_for_calcs)
                    bottom = round(bottom / downsample_for_calcs)
                boolean_face_3d[i_frame, int(top):int(bottom), int(left):int(right)] = True

    # Replace blank frames if face(s) in neighbouring frames with overlap
    for i_frame, frame in enumerate(boolean_face_3d):
        if i_frame == 0 or i_frame == frames_video - 1:  # Can't do this for 1st or last frame
            continue
        if True not in frame:
            neighbour_overlap = boolean_face_3d[i_frame - 1] & boolean_face_3d[i_frame + 1]
            boolean_face_3d[i_frame] = neighbour_overlap

    # Find faces through time
    id_face_3d, n_faces = skimage.measure.label(boolean_face_3d, return_num=True)

    # Iterate over faces in video
    rois = []
    for i_face in range(1, n_faces + 1):
        # Find the first and last frame containing the face
        frames = np.where(np.any(id_face_3d == i_face, axis=(1, 2)) == True)
        starting_frame, ending_frame = frames[0].min(), frames[0].max()

        # Iterate over the frames with faces in and find the min/max cols/rows (bounding box)
        cols, rows = [], []
        for i_frame in range(starting_frame, ending_frame + 1):
            rs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=1) == True)
            rows.append((rs[0].min(), rs[0].max()))
            cs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=0) == True)
            cols.append((cs[0].min(), cs[0].max()))
        frame_from, frame_to = starting_frame * temporal_upsample, ((ending_frame + 1) * temporal_upsample) - 1
        rows_from, rows_to = np.array(rows)[:, 0].min(), np.array(rows)[:, 1].max()
        cols_from, cols_to = np.array(cols)[:, 0].min(), np.array(cols)[:, 1].max()

        frame_to = min(frame_to, frame_from + MAX_FRAMES_FOR_FACE)

        if frame_to - frame_from >= MIN_FRAMES_FOR_FACE:
            rois.append(((frame_from, frame_to),
                         (int(rows_from * upsample * downsample_for_calcs),
                          int(rows_to * upsample * downsample_for_calcs)),
                         (int(cols_from * upsample * downsample_for_calcs),
                          int(cols_to * upsample * downsample_for_calcs))))
    return np.array(rois)


def get_frame_rois_for_valid_faces(faces_by_frame, video_shape, temporal_upsample, downsample_for_calcs=1, upsample=1,
                                   n_frames_per_face=5, min_frames_for_face=100):
    if not downsample_for_calcs:
        downsample_for_calcs = 1

    # Create boolean face array
    frames_video, rows_video, cols_video, channels_video = video_shape
    frames_video = math.ceil(frames_video)
    if downsample_for_calcs != 1:
        boolean_face_3d = np.zeros(
            (frames_video, rows_video // downsample_for_calcs, cols_video // downsample_for_calcs),
            dtype=np.bool)  # Remove colour channel
    else:
        boolean_face_3d = np.zeros((frames_video, rows_video, cols_video), dtype=np.bool)  # Remove colour channel
    for i_frame, faces in enumerate(faces_by_frame):
        if faces is not None:  # May not be a face in the frame
            for face in faces:
                left, top, right, bottom = face
                if downsample_for_calcs != 1:
                    left = round(left / downsample_for_calcs)
                    top = round(top / downsample_for_calcs)
                    right = round(right / downsample_for_calcs)
                    bottom = round(bottom / downsample_for_calcs)
                boolean_face_3d[i_frame, int(top):int(bottom), int(left):int(right)] = True

    # Replace blank frames if face(s) in neighbouring frames with overlap
    for i_frame, frame in enumerate(boolean_face_3d):
        if i_frame == 0 or i_frame == frames_video - 1:  # Can't do this for 1st or last frame
            continue
        if True not in frame:
            if TWO_FRAME_OVERLAP:
                if i_frame > 1:
                    pre_overlap = boolean_face_3d[i_frame - 1] | boolean_face_3d[i_frame - 2]
                else:
                    pre_overlap = boolean_face_3d[i_frame - 1]
                if i_frame < frames_video - 2:
                    post_overlap = boolean_face_3d[i_frame + 1] | boolean_face_3d[i_frame + 2]
                else:
                    post_overlap = boolean_face_3d[i_frame + 1]
                neighbour_overlap = pre_overlap & post_overlap
            else:
                neighbour_overlap = boolean_face_3d[i_frame - 1] & boolean_face_3d[i_frame + 1]
            boolean_face_3d[i_frame] = neighbour_overlap

    # Find faces through time
    id_face_3d, n_faces = skimage.measure.label(boolean_face_3d, return_num=True)

    list_of_frame_roi_dicts = []

    for i_face in range(1, n_faces + 1):
        frame_roi = {}
        frames = np.where(np.any(id_face_3d == i_face, axis=(1, 2)) == True)
        starting_frame, ending_frame = frames[0].min(), frames[0].max()

        # Skip faces with not enough frames
        face_length_in_frames = ((ending_frame + 1) * temporal_upsample) - (starting_frame * temporal_upsample)
        if face_length_in_frames <= min_frames_for_face:
            # print(f"Skipping video as {face_length_in_frames} < {min_frames_for_face} frames for this face"
            #       f"From {(starting_frame * temporal_upsample)} minus {((ending_frame + 1) * temporal_upsample)}")
            continue

        frame_numbers = [int(round(f)) for f in np.linspace(starting_frame, ending_frame, n_frames_per_face)]

        for i_frame in frame_numbers:
            rs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=1) == True)
            cs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=0) == True)
            # frame_roi[i_frame] = rs[0].min(), rs[0].max(), cs[0].min(), cs[0].max()
            frame_roi[i_frame * temporal_upsample] = (int(rs[0].min() * upsample * downsample_for_calcs),
                                                      int(rs[0].max() * upsample * downsample_for_calcs),
                                                      int(cs[0].min() * upsample * downsample_for_calcs),
                                                      int(cs[0].max() * upsample * downsample_for_calcs))
            # print(f"ROIS are {frame_roi[i_frame * temporal_upsample]}")

        list_of_frame_roi_dicts.append(frame_roi)

    return list_of_frame_roi_dicts

    # # Iterate over faces in video
    # rois = []
    # for i_face in range(1, n_faces + 1):
    #     # Find the first and last frame containing the face
    #     frames = np.where(np.any(id_face_3d == i_face, axis=(1, 2)) == True)
    #     starting_frame, ending_frame = frames[0].min(), frames[0].max()
    #
    #     # Iterate over the frames with faces in and find the min/max cols/rows (bounding box)
    #     cols, rows = [], []
    #     for i_frame in range(starting_frame, ending_frame + 1):
    #         rs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=1) == True)
    #         rows.append((rs[0].min(), rs[0].max()))
    #         cs = np.where(np.any(id_face_3d[i_frame] == i_face, axis=0) == True)
    #         cols.append((cs[0].min(), cs[0].max()))
    #     frame_from, frame_to = starting_frame * temporal_upsample, ((ending_frame + 1) * temporal_upsample) - 1
    #     rows_from, rows_to = np.array(rows)[:, 0].min(), np.array(rows)[:, 1].max()
    #     cols_from, cols_to = np.array(cols)[:, 0].min(), np.array(cols)[:, 1].max()
    #
    #     frame_to = min(frame_to, frame_from + MAX_FRAMES_FOR_FACE)
    #
    #     if frame_to - frame_from >= MIN_FRAMES_FOR_FACE:
    #         rois.append(((frame_from, frame_to),
    #                      (int(rows_from * upsample * downsample_for_calcs), int(rows_to * upsample * downsample_for_calcs)),
    #                      (int(cols_from * upsample * downsample_for_calcs), int(cols_to * upsample * downsample_for_calcs))))
    return np.array(rois)
