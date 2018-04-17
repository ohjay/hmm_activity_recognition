#!/usr/bin/env python

import os
import cv2
import skvideo.io
import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============
# - DEFAULTS -
# ============


OPTFLOW_MIN = -1.0
OPTFLOW_MAX = 1.0

MEANS = {
    'optical_flow': 0.0,
    'freq_optical_flow': 0.0,
    'edge': 0.0,
    'shape': 0.0,
}
STDEVS = {
    'optical_flow': 1.0,
    'freq_optical_flow': 1.0,
    'edge': 1.0,
    'shape': 1.0,
}

# Parameters for Shi-Tomasi corner detection
ST_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

proc_count = 0


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


def use_feature(name, feature_toggles, verbose=True):
    """Returns true if feature NAME should be included in each frame's feature vector.
    The decision is made based on FEATURE_TOGGLES.

    Parameters
    ----------
    name: str
        the name of the feature

    feature_toggles: dict
        settings specifying whether or not to use each feature

    Returns
    -------
    use_or_not: bool
        True if the feature should be included, False if not
    """
    use_or_not = feature_toggles is None or feature_toggles.get(name, False)
    if verbose:
        if use_or_not:
            print('[o]     including feature "%s"' % name)
        else:
            print('[o] NOT including feature "%s"' % name)
    return use_or_not


# ======================
# - FEATURE EXTRACTION -
# ======================


def subtract_background_v1(img, fgbg, kernel_size, threshold):
    """Perform background subtraction using OpenCV and outlier rejection.

    To perform noise/outlier rejection, we zero out regions
    that don't contain many pixels after background subtraction.

    Parameters
    ----------
    img: ndarray
        grayscale frame to process

    fgbg
        OpenCV background subtraction object

    kernel_size: int
        size of the kernel used for denoising

    threshold: int
        denoising threshold
    """
    fg_mask = fgbg.apply(img)
    fg_mask = np.clip(fg_mask, 0, 1)
    corr = signal.correlate2d(fg_mask, np.ones((kernel_size, kernel_size)),
                              mode='same', boundary='fill', fillvalue=0)
    fg_mask *= (corr > threshold)  # remove noise
    fg = img * fg_mask

    return fg, fg_mask


def subtract_background_v2(img, avg):
    """Eliminate the background by subtracting a running average of each pixel value.

    Parameters
    ----------
    img: ndarray
        grayscale frame to process

    avg: ndarray
        running average of pixel values
    """
    cv2.accumulateWeighted(img, avg, 0.01)
    res = cv2.convertScaleAbs(avg)
    fg = img - res
    fg_mask = fg != 0
    return fg, fg_mask


def condense(feature_list):
    """Convert all of the components in FEATURE_LIST into one array.
    Individual components will be flattened.

    Parameters
    ----------
    feature_list: list
        a list of feature components

    Returns
    -------
    feature_vec: ndarray
        a 1D array containing all of the feature components
    """
    if not feature_list:
        return None
    feature_list = [f.flatten() for f in feature_list]
    return np.concatenate(feature_list, axis=0)


def feat_shape(img, fg_mask):
    """Extract shape features.

    Parameters
    ----------
    img: ndarray
        grayscale frame to process

    fg_mask: ndarray
        foreground mask

    Returns
    -------
    centroid_diff: ndarray
        difference of centroids
    """
    activepts_grayfg = np.nonzero(img)
    centroid_grayfg = np.array([activepts_grayfg[0].mean(), activepts_grayfg[1].mean()])

    edges = cv2.Canny(fg_mask, 0, 127)
    activepts_edges = np.nonzero(edges)
    centroid_edges = np.array([activepts_edges[0].mean(), activepts_edges[1].mean()])

    centroid_diff = np.subtract(centroid_grayfg, centroid_edges)
    # TODO: DFT (20 dim) then PCA (8 dim) on D1
    # I think I need to run DFT on the entire array also

    return centroid_diff


def feat_edge(img, edges=None):
    """Extract edge features.
    This consists of the Fourier transform of the centroid-centered edge representation.
    """
    if edges is None:
        edges = cv2.Canny(img, 0, 255)
    _nz_active = tuple(np.nonzero(edges))
    if _nz_active[0].size == 0:
        centroid = np.array(img.shape) // 2
    else:
        centroid = np.array([np.mean(_nz_active[0]), np.mean(_nz_active[1])]).astype(np.int32)
    mdim2 = np.min(img.shape) // 2
    edges_padded = np.pad(edges, ((mdim2, mdim2), (mdim2, mdim2)), 'constant')
    centroid += mdim2
    mdim4 = mdim2 // 2
    centered = edges_padded[centroid[0]-mdim4:centroid[0]+mdim4, centroid[1]-mdim4:centroid[1]+mdim4]
    return centered


def feat_optical_flow(prev_img, img, p0, st_config, lk_config):
    """Extract optical flow features.

    Parameters
    ----------
    prev_img: ndarray
        grayscale frame at time t-1

    img: ndarray
        grayscale frame at time t

    p0: ndarray
        points to track

    st_config: dict
        Shi-Tomasi parameters

    lk_config: dict
        Lucas-Kanade parameters

    Returns
    -------
    flow: ndarray
        optical flow features

    p0: ndarray
        updated (?) points to track
    """
    if p0 is None or len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(prev_img, mask=None, **st_config)
    if p0 is None or len(p0) == 0:  # still
        flow = np.array([])
    else:
        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, p0, None, **lk_config)
        good_new = p1[status == 1]
        good_old = p0[status == 1]
        flow = good_new - good_old
        p0 = good_new.reshape(-1, 1, 2)
    return flow, p0


def feat_freq_optical_flow(prev_img, img, p0, st_config, lk_config, n_bins):
    """Extract "histogram of flow" features.

    Accepts the same parameters as `feat_optical_flow`,
    but returns a histogram instead of the raw optical flow features.
    """
    flow, p0 = feat_optical_flow(prev_img, img, p0, st_config, lk_config)
    flow = flow.flatten()
    bins = np.concatenate(([-200.0], np.linspace(OPTFLOW_MIN, OPTFLOW_MAX, n_bins - 1), [200.0]))
    hist, bin_edges = np.histogram(flow, bins=bins)
    return hist, p0


# ====================
# - VIDEO PROCESSING -
# ====================


def process_all_video_dirs(base_dir, save_path=None, config=None):
    """Extracts features from all directories in BASE_DIR.

    Parameters
    ----------
    base_dir: str
        path to the directory containing video subdirectories

    save_path: str
        path to which features should be saved

    config: dict
        parameters for feature extraction
    """
    if os.path.isdir(save_path):
        save_dir = save_path
    else:
        save_dir = os.path.dirname(save_path)
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
    video_dir: str
        path to the directory containing videos

    save_path: str
        path to which features should be saved

    config: dict
        parameters for feature extraction
    """
    print('---------- BEGIN VIDEO DIRECTORY PROCESSING')
    print('[o] Video directory: %s' % video_dir)
    all_features = []
    lengths = []
    n_features = -1
    for _file in os.listdir(video_dir):
        if _file.endswith('.avi'):
            video_path = os.path.join(video_dir, _file)
            video_features = process_video(video_path, save_path=None, config=config)
            if video_features is not None:
                if video_features.shape[1] > n_features:
                    n_features = video_features.shape[1]  # number of features
                all_features.append(video_features)
                lengths.append(len(video_features))  # number of frames
    lengths = np.array(lengths)

    # Post-process features (e.g. to account for variability in length)
    pp_all_features = np.zeros((np.sum(lengths), n_features))
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
    video_path: str
        path to a single video

    save_path: str or None
        path to which features should be saved

    config: dict
        parameters for feature extraction

    Returns
    -------
    features: ndarray, shape (n_frames, n_features)
        feature matrix for the video
    """
    global proc_count
    print('--- %d' % proc_count)
    print('[o] Video path: %s' % video_path)
    proc_count += 1

    # Process config
    if config is None:
        config = {}
    fg_handler = config.get('fg_handler', 1)
    st = config.get('st', None)
    lk = config.get('lk', None)
    verbose = config.get('verbose', False)
    debug = config.get('debug', False)
    denoise = config.get('denoise', {})
    denoise_kernel_size = denoise.get('kernel_size', 5)
    denoise_threshold = denoise.get('threshold', 3)
    feature_toggles = config.get('feature_toggles', None)
    n_bins = config.get('n_bins', 20)
    trim = config.get('trim', 0)  # how many frames to ignore on each end
    edge_dim = config.get('edge_dim', 20)
    normalize = config.get('normalize', True)

    # Determine whether to use features
    use_edge = use_feature('edge', feature_toggles)
    use_shape = use_feature('shape', feature_toggles)
    use_optical_flow = use_feature('optical_flow', feature_toggles)
    use_freq_optical_flow = use_feature('freq_optical_flow', feature_toggles)

    # Load Shi-Tomasi and Lucas-Kanade configs
    st_config = _nondestructive_update(ST_PARAMS, st, disallow_strings=True)
    lk_config = _nondestructive_update(LK_PARAMS, lk, disallow_strings=True)

    # Main setup
    videogen = skvideo.io.vreader(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2() if fg_handler == 1 else None

    features = []
    features_indiv = defaultdict(list)
    avg = None
    prev_img = None
    p0 = None
    n_features = -1
    idx = 0
    try:
        for idx, frame in enumerate(videogen):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_edge = cv2.Canny(frame_gray, 0, 255)

            # Background subtraction
            # ----------------------
            if fg_handler == 1:
                fg, fg_mask = subtract_background_v1(frame_gray, fgbg, denoise_kernel_size, denoise_threshold)
            elif fg_handler == 2:
                if avg is None:
                    avg = np.float32(frame_gray)
                fg, fg_mask = subtract_background_v2(frame_gray, avg)
            else:
                # No foreground handling
                fg, fg_mask = frame_gray, None
            if debug:
                plt.imshow(fg)
                plt.show()

            # [FEATURE] Shape
            # ---------------
            if use_shape:
                centroid_diff = feat_shape(frame_gray, fg_mask)
                features_indiv['shape'].append(centroid_diff)

            # [FEATURE] Edge
            # --------------
            if use_edge:
                edge_result = feat_edge(frame_gray, edges=frame_edge).flatten()
                features_indiv['edge'].append(edge_result)

            # [FEATURE] Optical flow
            # ----------------------
            if use_optical_flow or use_freq_optical_flow:
                _k = 'freq_optical_flow' if use_freq_optical_flow else 'optical_flow'
                if prev_img is None:
                    features_indiv[_k].append(np.zeros(n_bins))
                else:
                    if use_freq_optical_flow:
                        flow, p0 = feat_freq_optical_flow(prev_img, frame_edge, p0, st_config, lk_config, n_bins)
                    else:
                        flow, p0 = feat_optical_flow(prev_img, frame_edge, p0, st_config, lk_config)
                    features_indiv[_k].append(flow)
                prev_img = frame_edge.copy()

        # Reduce dimensionality of edges
        if len(features_indiv['edge']) > 0:
            pca = PCA(n_components=edge_dim)
            edge_std = StandardScaler().fit_transform(np.array(features_indiv['edge']))
            features_indiv['edge'] = pca.fit_transform(edge_std)

        # Normalization
        if normalize:
            for k, v in features_indiv.items():
                features_indiv[k] = (np.array(v) - MEANS[k]) / STDEVS[k]

        # Feature combination
        # -------------------
        for j in range(idx + 1):
            frame_feature_list = [features_indiv[k][j] for k in sorted(features_indiv)]
            # Do combination
            frame_feature_vec = condense(frame_feature_list)
            if frame_feature_vec.size > n_features:
                n_features = frame_feature_vec.size
            if frame_feature_vec is not None:
                features.append(frame_feature_vec)

        # Post-process features
        pp_features = []
        features = features[trim:-trim]  # drop the first and final TRIM frames
        for frame_feature_vec in features:
            pp_vec = np.zeros(n_features)
            pp_vec[:frame_feature_vec.size] = frame_feature_vec
            pp_features.append(pp_vec)
        features = np.array(pp_features)

        print('[o] Processed %d frames.' % (idx + 1))
        if verbose:
            print('Sample feature vector')
            print('---------------------')
            print(features[-1, :])
            print('---------------------')
        print('Shape of video feature matrix: %r' % (features.shape,))

        if save_path is not None:
            h5f = h5py.File(save_path, 'w')
            h5f.create_dataset('features', data=features)
            h5f.close()
            print('[+] Saved features (for individual video `%s`) to %s.' % (video_path, save_path))

        return features  # (n_frames, n_features)
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
    infile: str
        path to the HDF5 file containing features

    Returns
    -------
    all_features: ndarray, shape (n_frames, n_features)
        feature matrix

    lengths: array, shape (n_sequences,)
        lengths of each sequence
    """
    h5f = h5py.File(infile, 'r')
    all_features = h5f['all_features'][:]
    lengths = h5f['lengths'][:]
    h5f.close()

    return all_features, lengths
