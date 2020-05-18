import os
import pickle
import skvideo.io
import multiprocessing

from tqdm import tqdm
from glob import glob
from cnn3d.export.export_utils import get_roi_for_each_face, load_video
from cnn3d.export.video_utils import resize_and_square_face

"""
20 EXAMPLE VIDEOS EXTRACTED BY THIS ARE AVAILABLE IN the 'data/face_videos_by_part' folder.

This file uses the pickles created by 1_export_rois.py to create MP4s which are cropped down as much as possible to a
face, whilst always including the full face as it moves through the video. We use the ROIs for the faces in the real
video, so that we don't have to extract ROIs for every fake separately.

In summary, the bounding box co-ordinates are then used to set a mask in a 3D array.
Faces which are contiguous (overlapping) in 3D are assumed to be a single person's face moving through face and time.
We then extract a bounding box which includes this entire face over time and create a video from this region of
interest, including every frame (not just every 10th).

One nice thing about this method, even ignoring the video aspect, is it greatly reduced false positives, 
because the 'face' had to be present for a long period of time of the video to count as a face.
"""

VIDEO_ROOT = "E:\\DFDC\\data"  # This is the DFDC source video root, ie this folder should contain dfdc_train_part_0, dfdc_train_part_1 etc.
OUTPUT_MP4_DIR = "../../data/face_videos_by_part"  # Where we save our new MP4s
BOX_PICKLE_DIR = "../../data/rois"  # Where the pickle are kept containing the ROIs for the faces

MAX_FRAMES_TO_LOAD = 300  # Videos in DFDC were rarely longer than 300 frames, and if they were we bin the rest
FACE_FRAMES = 10   # This must match what was used when the pickles were extracted in the previous script
FACEDETECTION_DOWNSAMPLE = 0.5  # Downsampling before MTCNN speeds things up
OUTPUT_FACE_SIZE = (256, 256)  # Output resolution of our videos

FRAME_CLUMP_SIZE_FRAMES = 150  # Start afresh every 150 frames so we can have a smaller ROI and therefore a bigger face
FRAME_CLUMP_SIZE_BOXES = FRAME_CLUMP_SIZE_FRAMES // FACE_FRAMES

if FACEDETECTION_DOWNSAMPLE:
    facedetection_upsample = 1 / FACEDETECTION_DOWNSAMPLE
else:
    facedetection_upsample = 1

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def export_chunks_for_video(params):
    mp4_path, boxes_all = params

    mp4_dir = os.path.basename(os.path.dirname(mp4_path))
    mp4_name = os.path.basename(mp4_path)

    mp4_path = os.path.join(VIDEO_ROOT, mp4_dir, mp4_name)

    pickle_id = os.path.dirname(mp4_path).rsplit('_',1)[-1]
    pickle_output_dir = os.path.join(OUTPUT_MP4_DIR, pickle_id)

    video_all = load_video(mp4_path,
                           every_n_frames=1,
                           to_rgb=True,
                           rescale=None,
                           inc_pil=False,
                           max_frames=MAX_FRAMES_TO_LOAD)

    for i_chunk in range(len(video_all) // FRAME_CLUMP_SIZE_FRAMES):  # -> [0, 1]
        video = video_all[i_chunk * FRAME_CLUMP_SIZE_FRAMES:(i_chunk + 1) * FRAME_CLUMP_SIZE_FRAMES]
        boxes = boxes_all[i_chunk * FRAME_CLUMP_SIZE_BOXES:(i_chunk + 1) * FRAME_CLUMP_SIZE_BOXES]

        faces = get_roi_for_each_face(faces_by_frame=boxes,
                                      video_shape=video.shape,
                                      temporal_upsample=FACE_FRAMES,
                                      upsample=facedetection_upsample,
                                      downsample_for_calcs=4)

        for i_face, face in enumerate(faces):
            (frame_from, frame_to), (row_from, row_to), (col_from, col_to) = face
            x = video[frame_from:frame_to, row_from:row_to + 1, col_from:col_to + 1]
            x = resize_and_square_face(x, output_size=OUTPUT_FACE_SIZE)
            skvideo.io.vwrite(os.path.join(pickle_output_dir, f"{mp4_name}_{i_chunk}_{i_face}.mp4"), x)

if __name__ == "__main__":
    box_pickles = sorted(glob(os.path.join(BOX_PICKLE_DIR, "*.pickle")))

    N_WORKERS = multiprocessing.cpu_count()

    for i_pickle, box_pickle_path in enumerate(box_pickles):
        print(f"Pickle {box_pickle_path} ({i_pickle} of {len(box_pickles)})")

        pickle_id = box_pickle_path.rsplit('_', 2)[1]
        pickle_output_dir = os.path.join(OUTPUT_MP4_DIR, pickle_id)
        if not os.path.exists(pickle_output_dir):
            os.makedirs(pickle_output_dir)

            with open(box_pickle_path, 'rb') as f:
                boxes_by_mp4 = pickle.load(f)

                with multiprocessing.Pool(N_WORKERS) as p:
                    for _ in tqdm(p.imap(export_chunks_for_video, boxes_by_mp4.items()), total=len(boxes_by_mp4)):
                        pass

        else:
            continue