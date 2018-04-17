#!/usr/bin/env python

import os
import cv2
import skvideo.io
import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ============
# - DEFAULTS -
# ============

# Parameters for Shi-Tomasi corner detection
ST_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# =========
# - UTILS -
# =========

def _nondestructive_update(dict0, dict1, disallow_strings=False):
    """Returns a version of DICT_0 updated using DICT_1."""
    ret = dict0.copy()
    if dict1 is not None:
        ret.update(dict1)
        if disallow_strings:
            for k, v in ret.items():
                if type(v) == str:
                    ret[k] = eval(v)
    return ret

# ======================
# - FEATURE EXTRACTION -
# ======================

def process_all_video_dirs(base_dir, save_path=None, config=None):
    """Extracts features from all directories in BASE_DIR.

    Parameters
    ----------
    base_dir : str
        path to the directory containing video subdirectories

    save_path : str
        path to which features should be saved

    config : dict
        parameters for feature extraction
    """
    save_dir = os.path.dirname(save_path)
    if not save_dir:
        save_dir = save_path
    i = 0
    for video_dir in os.listdir(base_dir):
        fvideo_dir = os.path.join(base_dir, video_dir)
        if os.path.isdir(fvideo_dir):
            name = os.path.basename(os.path.normpath(video_dir))
            process_video_dir(fvideo_dir, save_path=os.path.join(save_dir, name + '.h5'), config=config)
            i += 1
    print('---------- DONE. PROCESSED %d VIDEO DIRECTORIES.' % i)

def process_video_dir(video_dir, save_path=None, config=None):
    """Extracts features from all videos in the directory.

    Parameters
    ----------
    video_dir : str
        path to the directory containing videos

    save_path : str
        path to which features should be saved

    config : dict
        parameters for feature extraction
    """
    print('---------- BEGIN VIDEO DIRECTORY PROCESSING')
    print('[o] Video directory: %s' % video_dir)
    all_features = []
    lengths = []
    n_feat = -1
    for _file in os.listdir(video_dir):
        if _file.endswith('.avi'):
            video_path = os.path.join(video_dir, _file)
            video_features = process_video(video_path, save_path=None, config=config)
            if video_features is not None:
                if video_features.shape[1] > n_feat:
                    n_feat = video_features.shape[1]  # number of features
                all_features.append(video_features)
                lengths.append(len(video_features))  # number of frames
    lengths = np.array(lengths)

    # Post-process features (e.g. to account for variability in length)
    pp_all_features = np.zeros((np.sum(lengths), n_feat))
    i = 0
    for video_features in all_features:
        nfe, nfr = video_features.shape
        pp_all_features[i:i+nfe, :nfr] = video_features
        i += nfe
    all_features = pp_all_features
    print('\n---------- SUMMARY')
    print('[o] Shape of `all_features`: %r' % (all_features.shape,))

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not save_dir:
            save_dir = save_path
        if not os.path.exists(save_dir):
            print('[o] The directory `%s` does not yet exist. Creating it...' % save_dir)
            os.makedirs(save_dir)
        h5f = h5py.File(save_path, 'w')
        h5f.create_dataset('all_features', data=all_features)
        h5f.create_dataset('lengths', data=lengths)
        h5f.close()
        print('[+] Saved all features (for videos in `%s`) to %s.' % (video_dir, save_path))
    print('\n')

    return all_features, lengths

def process_video(video_path, save_path=None, config=None):
    """Extracts features from a single video.

    Parameters
    ----------
    video_path : str
        path to a single video

    save_path : str or None
        path to which features should be saved

    config : dict
        parameters for feature extraction

    Returns
    -------
    features : array, shape (n_frames, n_features)
        feature matrix for the video
    """
    print('---')
    print('[o] Video path: %s' % video_path)
    videogen = skvideo.io.vreader(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Process config
    features = config.get('features', None)
    st = config.get('st', None)
    lk = config.get('lk', None)
    verbose = config.get('verbose', False)
    debug = config.get('debug', False)
    denoise = config.get('denoise', {})
    denoise_kernel_size = denoise.get('kernel_size', 5)
    denoise_threshold = denoise.get('threshold', 3)

    # Load Shi-Tomasi and Lucas-Kanade configs
    st_config = _nondestructive_update(ST_PARAMS, st, disallow_strings=True)
    lk_config = _nondestructive_update(LK_PARAMS, lk, disallow_strings=True)

    opt_flow = []
    prev_frame_gray = None
    p0 = None
    n_feat = -1
    idx = 0
    try:
        for idx, frame in enumerate(videogen):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Background subtraction
            fg_mask = fgbg.apply(frame_gray)
            fg_mask = np.clip(fg_mask, 0, 1)
            corr = signal.correlate2d(fg_mask, np.ones((denoise_kernel_size, denoise_kernel_size)),
                                      mode='same', boundary='fill', fillvalue=0)
            fg_mask *= (corr > denoise_threshold)  # remove noise
            fg = frame_gray * fg_mask
            if debug:
                plt.imshow(fg)
                plt.show()

            # Shape feature extraction
            activepts_grayfg = np.nonzero(frame_gray)
            centroid_grayfg = np.array([activepts_grayfg[0].mean(), activepts_grayfg[1].mean()])

            edges = cv2.Canny(fg_mask,0,127)
            activepts_edges = np.nonzero(edges)
            centroid_edges = np.array([activepts_edges[0].mean(), activepts_edges[1].mean()])

            centroid_diff = np.subtract(centroid_grayfg, centroid_edges)
            # TODO: DFT (20 dim) then PCA (8 dim) on D1
            # I think I need to run DFT on the entire array also

            # Optical flow
            if prev_frame_gray is not None:
                if p0 is None or len(p0) == 0:
                    p0 = cv2.goodFeaturesToTrack(prev_frame_gray, mask=None, **st_config)
                if p0 is None:
                    opt_flow.append(np.array([]))
                else:
                    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, p0, None, **lk_config)
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

        opt_flow = np.array(opt_flow)
        print('[o] Processed %d frames.' % idx)

        if verbose:
            print('Sample opt flow field')
            print('---------------------')
            print(opt_flow[-1])
            print('---------------------')
        print('Shape of displacement field data: %r' % (opt_flow.shape,))

        if save_path is not None:
            h5f = h5py.File(save_path, 'w')
            h5f.create_dataset('opt_flow', data=opt_flow)
            h5f.close()
            print('[+] Saved features (for individual video `%s`) to %s.' % (video_path, save_path))

        return opt_flow  # ultimately should contain all features
    except RuntimeError:
        print('[-] Failed to read `%s`.' % video_path)
        return None

# =============
# - RELOADING -
# =============

def load_features(infile):
    """Loads features from an input file.

    Parameters
    ----------
    infile : str
        path to the HDF5 file containing features

    Returns
    -------
    all_features : array, shape (n_frames, n_features)
        feature matrix

    lengths : array, shape (n_sequences,)
        lengths of each sequence
    """
    h5f = h5py.File(infile, 'r')
    all_features = h5f['all_features'][:]
    lengths = h5f['lengths'][:]
    h5f.close()

    return all_features, lengths
