#!/usr/bin/env python

import sys
import cv2
import skvideo.io
import h5py
import numpy as np
import matplotlib.pyplot as plt

# TODO read from config file

# Parameters for Shi-Tomasi corner detection
ST_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def process_video(video_path, save_path=None):
    videogen = skvideo.io.vreader(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    fg_masks = []
    opt_flow = []
    prev_frame_gray = None
    p0 = None
    idx = 0
    for idx, frame in enumerate(videogen):
        # Background subtraction
        fg_mask = fgbg.apply(frame)
        fg_masks.append(fg_mask)

        # Optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            if p0 is None or len(p0) == 0:
                p0 = cv2.goodFeaturesToTrack(prev_frame_gray, mask=None, **ST_PARAMS)
            if p0 is None:
                opt_flow.append([])
            else:
                p1, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, p0, None, **LK_PARAMS)
                good_new = p1[status == 1]
                good_old = p0[status == 1]
                opt_flow.append(good_new - good_old)
                p0 = good_new.reshape(-1, 1, 2)
        prev_frame_gray = frame_gray.copy()
    print('[o] Processed %d frames.' % idx)

    print('Sample FG mask')
    print('--------------')
    print(fg_masks[0])
    print('--------------')
    print('Number of FG masks: %d' % len(fg_masks))
    print('Number of displacement fields: %d' % len(opt_flow))

    if save_path is not None:
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('fg_masks', data=np.array(fg_masks))
        h5f.create_dataset('opt_flow', data=np.array(opt_flow))
        h5f.close()
        print('[+] Saved features to %s.' % save_path)

if __name__ == '__main__':
    process_video(sys.argv[1])  # debug
