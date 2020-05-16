import skimage.measure
import decord
import pickle
import pandas as pd
import numpy as np
import math
import json

from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom


class VideoDataset(Dataset):

    def __init__(self, videos, NUMFRAMES):
        self.videos = videos
        self.NUMFRAMES = NUMFRAMES

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        try:
            vr = decord.VideoReader(self.videos[i], ctx=decord.cpu())
            start = np.random.choice(len(vr)-self.NUMFRAMES)
            vid = vr.get_batch(list(range(start, start+self.NUMFRAMES))).asnumpy()
        except Exception as e:
            return 0
        return (vid, start)


class VideoFirstFramesDataset(Dataset):

    def __init__(self, videos, NUMFRAMES):
        self.videos = videos
        self.NUMFRAMES = NUMFRAMES

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        try:
            vr = decord.VideoReader(self.videos[i], ctx=decord.cpu())
            vid = np.asarray([vr[i].asnumpy() for i in range(self.NUMFRAMES)])
            start = 0
        except Exception as e:
            print(e)
            return 0
        return (vid, start)


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2


def get_coords(faces_roi, i, j):
    coords = np.argwhere(faces_roi[j] == i+1)
    if coords.shape[0] == 0:
        return None
    y1, x1 = coords[0]
    y2, x2 = coords[-1]
    return x1, y1, x2, y2


def interpolate_center(c1, c2, length):
    x1, y1 = c1
    x2, y2 = c2
    xi, yi = np.linspace(x1, x2, length), np.linspace(y1, y2, length)
    return np.vstack([xi, yi]).transpose(1,0)


def load_roi(pklfile):
    with open(pklfile, 'rb') as f:
        rois = pickle.load(f)
    return {k.split('\\')[-2]+'/'+k.split('\\')[-1] : v for k,v in rois.items()}


def load_metadata(jsonfile):
    with open(jsonfile) as f:
        x = json.load(f)
    part = jsonfile.split('/')[-2]
    videos = []
    labels = []
    splits = []
    originals = []
    for k,v in x.items():
        videos.append(k)
        labels.append(v['label'])
        splits.append(v['split'])
        try:
            originals.append(v['original'])
        except KeyError:
            originals.append(None)
    return pd.DataFrame({
        'filename': videos,
        'video_label': labels, 
        'split': splits,
        'original': originals,
        'train_part': [part] * len(videos)
        })


def get_roi_for_each_face(faces_by_frame, video_shape, DOWNSAMPLE):
    # Create boolean face array
    frames_video, rows_video, cols_video, channels_video = video_shape
    frames_video = math.ceil(frames_video)
    boolean_face_3d = np.zeros((frames_video, rows_video, cols_video), dtype=np.bool)  # Remove colour channel
    for i_frame, faces in enumerate(faces_by_frame):
        if faces is not None:  # May not be a face in the frame
            for face in faces:
                face = face / (DOWNSAMPLE/2)
                left, top, right, bottom = face
                boolean_face_3d[i_frame, int(top):int(bottom), int(left):int(right)] = True
             
    # Replace blank frames if face(s) in neighbouring frames with overlap
    for i_frame, frame in enumerate(boolean_face_3d):
        if i_frame == 0 or i_frame == frames_video-1:  # Can't do this for 1st or last frame
            continue
        if True not in frame:
            neighbour_overlap = boolean_face_3d[i_frame-1] & boolean_face_3d[i_frame+1]
            boolean_face_3d[i_frame] = neighbour_overlap

    # Find faces through time
    id_face_3d, n_faces = skimage.measure.label(boolean_face_3d, return_num=True)
    return id_face_3d, n_faces


def get_faces(vid, bboxes, start, MAX_FACES=2, DOWNSAMPLE=8, NUMFRAMES=130, DIVISOR=10): 
    faces_roi, num_faces = get_roi_for_each_face(bboxes, (len(bboxes), vid.shape[1]//DOWNSAMPLE, vid.shape[2]//DOWNSAMPLE, 3), DOWNSAMPLE)
    faces_roi = zoom(faces_roi, [1, DOWNSAMPLE, DOWNSAMPLE], order=0, prefilter=False)
    all_faces = []
    for i in range(min(MAX_FACES, num_faces)):
        faces = np.asarray([get_coords(faces_roi, i, j) for j in range(len(faces_roi))])
        if None in faces:
            not_none = np.where(faces != None)[0]
            if np.min(not_none) <= start//DIVISOR and np.max(not_none) >= start//DIVISOR+NUMFRAMES//DIVISOR:
                for coord_ind, coord in enumerate(faces):
                    if type(coord) == type(None):
                        faces[coord_ind] = np.asarray([0,0,0,0])
                faces = np.vstack(faces)
            else:
                continue
        all_faces.append(faces)

    extracted_faces = []
    for face in all_faces:
        # Get max dim size
        max_dim = np.concatenate([face[:,2]-face[:,0],face[:,3]-face[:,1]])
        max_dim = np.percentile(max_dim, 90)
        # Enlarge by 1.2
        max_dim = int(max_dim * 1.2)
        # Get center coords
        centers = np.asarray([get_center(_) for _ in face])
        # Interpolate
        centers = np.vstack([interpolate_center(centers[i], centers[i+1], length=10) for i in range(len(centers)-1)]).astype('int')
        centers = centers[start:start+NUMFRAMES]
        x1y1 = centers - max_dim // 2
        x2y2 = centers + max_dim // 2 
        x1, y1 = x1y1[:,0], x1y1[:,1]
        x2, y2 = x2y2[:,0], x2y2[:,1]
        # If x1 or y1 is negative, turn it to 0
        # Then add to x2 y2 or y2
        x2[x1 < 0] -= x1[x1 < 0]
        y2[y1 < 0] -= y1[y1 < 0]
        x1[x1 < 0] = 0
        y1[y1 < 0] = 0
        # If x2 or y2 is too big, turn it to max image shape
        # Then subtract from y1
        y1[y2 > vid.shape[1]] += vid.shape[1] - y2[y2 > vid.shape[1]]
        x1[x2 > vid.shape[2]] += vid.shape[2] - x2[x2 > vid.shape[2]]
        y2[y2 > vid.shape[1]] = vid.shape[1]
        x2[x2 > vid.shape[2]] = vid.shape[2]
        vidface = np.asarray([vid[_,
                                 y1[_]:y2[_],
                                 x1[_]:x2[_]] for _, c in enumerate(centers)])
        extracted_faces.append(vidface)

    return extracted_faces
