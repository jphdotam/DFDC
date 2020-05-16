import os
import shutil
import pandas as pd
from glob import glob
from tqdm import tqdm

"""This file should be self-explanatory. It copies MP4s of faces from folders where they are stored according to the
original DFDC part they came in"""

SOURCE_DIR = "E:\\DFDC\\data"  # Used to locate metadata.json
MP4_DIR = "../../data/face_videos_by_part"
OUT_DIR = "../../data/face_videos_by_real_fake"

metadata_files = glob(os.path.join(SOURCE_DIR, "**", "metadata.json"), recursive=True)
print(f"Found {len(metadata_files)} metadata files")
df_metadata = []
for metadata_file in metadata_files:
    df = pd.read_json(metadata_file).T
    df['dir'] = os.path.basename(os.path.dirname(metadata_file))
    df['path'] = df['dir'] + '/' + df.index
    df_metadata.append(df)
df_metadata = pd.concat(df_metadata)

mp4paths = sorted(glob(os.path.join(MP4_DIR, "**/*.mp4")))

# Make the real and fake dir if needed
for realfake in ('REAL','FAKE'):
    if not os.path.exists(os.path.join(OUT_DIR, realfake)):
        os.makedirs(os.path.join(OUT_DIR, realfake))

for mp4path in tqdm(mp4paths):
    i_chunk = os.path.basename(os.path.dirname(mp4path))
    video_id = os.path.basename(mp4path).split('_',1)[0]
    realfake = df_metadata.loc[video_id].label

    outpath = os.path.join(OUT_DIR, realfake, f"{i_chunk}_{os.path.basename(mp4path)}")

    shutil.copy(mp4path, outpath)