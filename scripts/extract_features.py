#!/usr/bin/env python

import os
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

def process_video_dir(video_dir, save_path=None):
    print('[o] Video directory: %s' % video_dir)
    all_features = []
    lengths = []
    video_features, n_feat = None, -1
    for _file in os.listdir(video_dir):
        if _file.endswith('.avi'):
            video_path = os.path.join(video_dir, _file)
            video_features = process_video(video_path, save_path=None)
            if video_features.shape[1] > n_feat:
                n_feat = video_features.shape[1]  # number of features
            all_features.append(video_features)
            lengths.append(len(video_features))  # number of frames
    lengths = np.array(lengths)

    # Post-process features (e.g. to account for variability in length)
    pp_all_features = np.zeros((len(all_features), n_feat))
    i = 0
    for video_features in all_features:
        nfe, nfr = video_features.shape
        pp_all_features[i:i+nfe, :nfr] = video_features
        i += nfe
    all_features = pp_all_features
    print('[o] Shape of `all_features`: %r' % (all_features.shape,))

    if save_path is not None:
        _dir = os.path.dirname(save_path)
        if not os.path.exists(_dir):
            print('[o] The directory `%s` does not yet exist. Creating it...' % _dir)
            os.makedirs(_dir)
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('all_features', data=all_features)
        h5f.create_dataset('lengths', data=lengths)
        h5f.close()
        print('[+] Saved all features (for videos in `%s`) to %s.' % (video_dir, save_path))

    return all_features, lengths

def process_video(video_path, save_path=None, verbose=False):
    print('[o] Video path: %s' % video_path)
    videogen = skvideo.io.vreader(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    fg_masks = []
    opt_flow = []
    prev_frame_gray = None
    p0 = None
    n_feat = -1
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
                opt_flow.append(np.array([]))
            else:
                p1, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, p0, None, **LK_PARAMS)
                good_new = p1[status == 1]
                good_old = p0[status == 1]
                if good_new.size > n_feat:
                    n_feat = good_new.size  # track the maximum number of features (points * 2) in any flow
                opt_flow.append(good_new - good_old)
                p0 = good_new.reshape(-1, 1, 2)
        prev_frame_gray = frame_gray.copy()

    # Post-process optical flow
    pp_opt_flow = []
    for flow in opt_flow:
        pp_flow = np.zeros(n_feat)
        pp_flow[:flow.size] = flow.flatten()
        pp_opt_flow.append(pp_flow)
    opt_flow = pp_opt_flow

    fg_masks = np.array(fg_masks)
    opt_flow = np.array(opt_flow)
    print('[o] Processed %d frames.' % idx)

    if verbose:
        print('Sample opt flow field')
        print('---------------------')
        print(opt_flow[-1])
        print('---------------------')
    print('Shape of FG mask data: %r' % (fg_masks.shape,))
    print('Shape of displacement field data: %r' % (opt_flow.shape,))

    if save_path is not None:
        _dir = os.path.dirname(save_path)
        if not os.path.exists(_dir):
            print('[o] The directory `%s` does not yet exist. Creating it...' % _dir)
            os.makedirs(_dir)
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('fg_masks', data=fg_masks)
        h5f.create_dataset('opt_flow', data=opt_flow)
        h5f.close()
        print('[+] Saved features (for individual video `%s`) to %s.' % (video_path, save_path))

    return opt_flow  # ultimately should contain all features

if __name__ == '__main__':
    process_video(sys.argv[1])  # debug
