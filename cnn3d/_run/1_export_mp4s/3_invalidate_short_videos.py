import shutil
import skvideo.io
import multiprocessing

from glob import glob
from tqdm import tqdm

"""This script just renames files that have < 64 frames (and so can't be used to train 3D CNNs)
from *.mp4 to *.mp4.short"""

PATH = "../../data/face_videos_by_part"
MIN_FRAMES = 64

def check_and_move_video(videopath):
    v = skvideo.io.vread(videopath)
    if v.shape[0] < MIN_FRAMES:
        print(f'Moving {videopath}')
        shutil.move(videopath, videopath + '.short')

if __name__ == "__main__":
    N_WORKERS = multiprocessing.cpu_count()

    videopaths = glob(PATH, recursive=True)

    with multiprocessing.Pool(N_WORKERS) as p:
        for _ in tqdm(p.imap(check_and_move_video, iter(videopaths)), total=len(videopaths)):
            pass


