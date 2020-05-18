import os
import pickle
from glob import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN

from cnn3d.export.export_utils import load_video

"""
NB THIS SCRIPT TAKES A LONG TIME - ALL OF THE ROI PICKLES HAVE THEREFORE BEEN PROVIDED IN THE data/rois FOLDER.
Unfortunately, because it relies on MTCNN, multiprocessing with python was not practical. The next step, using these
to create MP4 files, however, was highly parallelisable, and so these two steps have been separated, and this step
creates pickle files which the next step can use.

This file reads in every _REAL_ MP4 from the original DFDC dataset (which MP4 root should be set to), which
was in the format of:

    MP4_ROOT/
        dfdc_train_part_0/
            aaqaifqrwn.mp4
            aayrffkzxn.mp4
            ...
        ...
        dfdc_train_part_49/
            ...
            zzxireqbdi.mp4
            zzylfwxjbb.mp4
        
For each of the dfdc_train_part_* folders, we create a pickle in the OUT_DIR, so we are left with:

    OUT_DIR/
        faces_dfdc_train_part_0_10.pickle
        faces_dfdc_train_part_1_10.pickle
        ...
        faces_dfdc_train_part_49_10.pickle
        
Each pickle file is a dictionary, where:
    The key is the full path to each MP4 in that part
    The value is a series of numpy arrays, one corresponding to every FACE_FRAMES'th frame
        Where each of these numpy arrays is of shape <n_faces_found_in_frame> * 4
        With the 4 corresponding to the 4 bounding box coordinates returned by MTCNN
        
    For example, if video aaqaifqrwn.mp4
        - is in dfdc_train_part_0
        - contains 300 frames
        - has a single face visible in every frame
    -> We would end up with an entry as follows in dfdc_train_part_0's pickled dictionary:
    
    {   ...,
        'E:\\DFDC\\data\\dfdc_train_part_0\\aaqaifqrwn.mp4': <ndarray of shape (30, 1, 4)>,
        ...
    }
    
These pickle files will then be used the the next file, '2_export_mp4s.py'
"""

# Data
OUT_DIR = "../../data/rois"
MP4_ROOT = "E:\\DFDC\\data\\dfdc_train_part_*"

# Face detection
MAX_FRAMES_TO_LOAD = 300
MAX_FACES = 2
FACE_FRAMES = 10
FACEDETECTION_DOWNSAMPLE = 0.5
MTCNN_THRESHOLDS = (0.6, 0.7, 0.7)  # Default [0.6, 0.7, 0.7]
MMTNN_FACTOR = 0.709  # Default 0.709 p

mp4_dirs = sorted(glob(MP4_ROOT))
mp4_dirs_exported = [os.path.basename(d).split('_',1)[1].rsplit('_',1)[0] for d in glob(os.path.join(OUT_DIR, "*"))]
mp4_dirs_unexported = [d for d in mp4_dirs if os.path.basename(d) not in mp4_dirs_exported]

print(f"Found {len(mp4_dirs)} MP4 dirs; {len(mp4_dirs_unexported)} needing export")
keep_all = True if MAX_FACES > 1 else False

if FACEDETECTION_DOWNSAMPLE:
    facedetection_upsample = 1/FACEDETECTION_DOWNSAMPLE
else:
    facedetection_upsample = 1

mtcnn = MTCNN(margin=0, keep_all=keep_all, post_process=False, device='cuda:0',
              thresholds=MTCNN_THRESHOLDS, factor=MMTNN_FACTOR)


for i_mp4_dir, mp4_dir in enumerate(mp4_dirs_unexported[1:]):
    boxes_by_frame_by_videopath = {}
    mp4_dir_name = os.path.basename(mp4_dir)

    print(f"Dir {mp4_dir} ({i_mp4_dir+1} of {len(mp4_dirs_unexported)})")
    mp4_paths = glob(os.path.join(mp4_dir, "*.mp4"))
    for mp4_path in tqdm(mp4_paths):
        try:
            video, pil_frames = load_video(mp4_path,
                                           every_n_frames=FACE_FRAMES,
                                           to_rgb=True,
                                           rescale=FACEDETECTION_DOWNSAMPLE,
                                           inc_pil=True,
                                           max_frames=MAX_FRAMES_TO_LOAD)

            if len(pil_frames):  # Randomly in 1 committ this was an empty list?!
                boxes, _probs = mtcnn.detect(pil_frames, landmarks=False)  # MAX_FRAMES / FACE_FRAMES boxes (e.g. 150 / 10 -> 15)
                boxes_by_frame_by_videopath[mp4_path] = boxes
        except Exception as e:
            print(f"Error with file {mp4_path}: {e}")

    with open(f"{OUT_DIR}/faces_{mp4_dir_name}_{FACE_FRAMES}.pickle", 'wb') as f:
        pickle.dump(boxes_by_frame_by_videopath, f)
