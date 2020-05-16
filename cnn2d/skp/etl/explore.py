import decord
import pickle
import numpy as np
import math
import cv2
import skimage.measure

vr = decord.VideoReader('cqroxivqio.mp4', ctx=decord.cpu())
video = vr.get_batch(list(range(0, len(vr), 10))).asnumpy()

with open('sample_roi.pkl', 'rb') as f:
    roi = pickle.load(f)

for frame_ind, i in enumerate(roi): 
    for roi_ind, j in enumerate(i):
        x1, y1, x2, y2 = (j*2).astype('int')
        crop = video[frame_ind,y1:y2,x1:x2]
        status = cv2.imwrite('FRAME{:03d}_ROI{}.png'.format(frame_ind, roi_ind), crop[...,::-1])

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

face_map, _ = get_roi_for_each_face(roi, (len(roi), video.shape[1]//8, video.shape[2]//8, 3), DOWNSAMPLE=8)
face_map[face_map != 1] = 0

faces = []
for i in range(len(face_map)):
    coords = np.argwhere(face_map[i] == 1)
    y1, x1 = coords[0]*8
    y2, x2 = coords[-1]*8
    faces.append(video[i][y1:y2,x1:x2])

for ind, i in enumerate(faces):
    status = cv2.imwrite('FRAME{:03d}_FACE.png'.format(ind))

    