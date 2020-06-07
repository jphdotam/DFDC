This is the first step of the system.

The idea is these scripts will take us from the original deepfake dataset to ~250,000 MP4 files which are closely
cropped to individual people's faces, tracking a bounding box temporally through time.

Examples of such videos are seen in ../data/face_videos_by_real_fake/

There are 4 steps with corresponding scripts required to generate this dataset

* 1_export_rois.py - This analyses every 10th frame of the original DFDC videos and stores bounding box coordinates for
face face present. Only the real videos (non-faked) are analysed, to save time. Fake videos use the ROIs from their
corresponding real partner.

* 2_export_mp4s.py - This uses the coordinates produced by the previous script to export MP4s for every unique face
present in a video. These MP4s are cropped down so they are as small as possible, whilst containing the full face even
if it moves across the screen. (We actually analyse each 300 frame videos as 2 smaller 150 frame videos).

* 3_invalidate_short_videos.py - MP4s created in the previous step are checked to ensure they have <= 64 frames. If not,
they are renamed to have a different file extension and are not used for training.

* 4_split_videos_into_real_and_fake.py - Self-explanatory, we move videos from being shorted by the chunk number they
were in in the original DFDC dataset, to the final training folder (REAL or FAKE).
