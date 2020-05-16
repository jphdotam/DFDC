import cv2
import numpy as np

def resize_and_square_face(video, output_size):
    input_size = max(video.shape[1], video.shape[2])  # We will square it, so this is the effective input size
    out_video = np.empty((len(video), output_size[0], output_size[1], 3), np.dtype('uint8'))

    for i_frame, frame in enumerate(video):
        padded_image = np.zeros((input_size, input_size, 3))
        padded_image[0:frame.shape[0], 0:frame.shape[1]] = frame
        if (input_size, input_size) != output_size:
            frame = cv2.resize(padded_image, (output_size[0], output_size[1])).astype(np.uint8)
        else:
            frame = padded_image.astype(np.uint8)
        out_video[i_frame] = frame
    return out_video


def center_crop_video(video, crop_dimensions):
    height, width = video.shape[1], video.shape[2]
    crop_height, crop_width = crop_dimensions

    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width

    video_out = np.zeros((len(video), crop_height, crop_width, 3))
    for i_frame, frame in enumerate(video):
        video_out[i_frame] = frame[y1:y2, x1:x2]

    return video_out


def get_last_frame_needed_across_faces(faces):
    last_frame = 0

    for face in faces:
        (frame_from, frame_to), (row_from, row_to), (col_from, col_to) = face
        last_frame = max(frame_to, last_frame)

    return last_frame