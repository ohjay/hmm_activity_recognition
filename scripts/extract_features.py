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
from .dataset_utils import read_sequences_file


# ============
# - DEFAULTS -
# ============


OPTFLOW_MIN = -1.0
OPTFLOW_MAX = 1.0

NORMALIZE_FEAT_CENTROID = True

MEANS = {
    'optical_flow': 0.0,
    'freq_optical_flow': 0.0,
    'edge': 0.0,
    'centroid': 0.0,
}
STDEVS = {
    'optical_flow': 1.0,
    'freq_optical_flow': 1.0,
    'edge': 1.0,
    'centroid': 1.0,
}

# Parameters for Shi-Tomasi corner detection
ST_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

proc_count = 0
seq_info = {}


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


def top_k_indices(arr, k):
    """Returns the top K indices of an array.

    Returns
    -------
    indices: array, shape (k,)
        indices of the top k values in the flattened array
    """
    return np.argpartition(arr.ravel(), -k)[-k:]  # alt: np.argsort(arr, axis=None)[-k:]


def extract_mres_wind(arr, winsize):
    """Extract the window of maximum response from ARR (which is presumed to be 2D)."""
    ones_filter = np.ones((winsize, winsize))
    wsh = winsize // 2
    response = signal.fftconvolve(arr, ones_filter, mode='same')
    max_iy, max_ix = np.unravel_index(response.argmax(), response.shape)
    arr = np.pad(arr, winsize, 'constant')
    max_iy += winsize
    max_ix += winsize
    return arr[max_iy - wsh:max_iy + wsh, max_ix - wsh:max_ix + wsh]


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


def compute_centroid(img, normalize=False):
    """Return the centroid of IMG as a (y, x) position."""
    h, w = img.shape[0], img.shape[1]
    _nz_active = tuple(np.nonzero(img))
    if _nz_active[0].size == 0:
        _nz_active = ([float(h) / 2], [float(w) / 2])
    if normalize:
        return np.array([np.mean(_nz_active[0]) / h, np.mean(_nz_active[1]) / w])
    return np.array([np.mean(_nz_active[0]), np.mean(_nz_active[1])])


def feat_edge(img, edges=None):
    """Extract edge features.
    This consists of the centroid-centered edge representation.
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


def feat_centroid(img):
    """Extract centroid feature."""
    return compute_centroid(img, normalize=NORMALIZE_FEAT_CENTROID)


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
    flow: tuple
        (opt_flow_y, opt_flow_x) tuple, i.e. optical flow features

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
    if flow.size == 0 or len(flow.shape) == 1:
        flow_y, flow_x = flow.flatten(), flow.flatten()
    else:
        flow_y, flow_x = flow[:, 0], flow[:, 1]
    return (flow_y, flow_x), p0


def feat_freq_optical_flow(prev_img, img, p0, st_config, lk_config, n_bins):
    """Extract "histogram of flow" features.

    Accepts the same parameters as `feat_optical_flow`,
    but returns histograms instead of the raw optical flow features.
    """
    (flow_y, flow_x), p0 = feat_optical_flow(prev_img, img, p0, st_config, lk_config)
    bins = np.concatenate(([-200.0], np.linspace(OPTFLOW_MIN, OPTFLOW_MAX, n_bins - 1), [200.0]))
    hist_y, _ = np.histogram(flow_y, bins=bins)
    hist_x, _ = np.histogram(flow_x, bins=bins)
    return (hist_y, hist_x), p0


def feat_dense_optical_flow(prev_img, img, dense_params):
    """Extract dense optical flow features.
    Returns a flow over a ROI whose dimensions are specified by DENSE_PARAMS.
    """
    # Extract ROI
    h, w = img.shape
    roi_h = int(h * dense_params['roi_h'])
    roi_w = int(w * dense_params['roi_w'])
    prev_padded = np.pad(prev_img, ((roi_h, roi_h), (roi_w, roi_w)), 'wrap')
    curr_padded = np.pad(img, ((roi_h, roi_h), (roi_w, roi_w)), 'wrap')
    cy, cx = compute_centroid(img)
    cy, cx = int(cy) + roi_h, int(cx) + roi_w
    prev_roi = prev_padded[cy-roi_h/2:cy+roi_h/2, cx-roi_w/2:cx+roi_w/2]
    curr_roi = curr_padded[cy-roi_h/2:cy+roi_h/2, cx-roi_w/2:cx+roi_w/2]

    # Compute dense optical flow using Farneback's algorithm
    pyr_scale = dense_params.get('pyr_scale', 0.5)
    levels = dense_params.get('levels', 3)
    winsize = dense_params.get('winsize', 15)
    iterations = dense_params.get('iterations', 3)
    poly_n = dense_params.get('poly_n', 5)
    poly_sigma = dense_params.get('poly_sigma', 1.2)
    top_k = dense_params.get('top_k', None)
    mres_wind = dense_params.get('mres_wind', None)
    flow = cv2.calcOpticalFlowFarneback(
        prev_roi, curr_roi, flow=np.zeros_like(prev_img, dtype=np.float32),
        pyr_scale=pyr_scale, levels=levels, winsize=winsize, iterations=iterations,
        poly_n=poly_n, poly_sigma=poly_sigma, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    if type(mres_wind) == int:
        # Extract a square window around the region of maximum response
        wind_y = extract_mres_wind(flow[:, :, 0], mres_wind)
        wind_x = extract_mres_wind(flow[:, :, 1], mres_wind)
        flow = np.stack((wind_y, wind_x), axis=-1)
    if type(top_k) == int:
        # Only use the maximum K optical flow features (sorted)
        flow_y = flow[:, :, 0].flatten()
        flow_x = flow[:, :, 1].flatten()
        iy = sorted(top_k_indices(flow_y, top_k), key=lambda i: flow_y[i])
        ix = sorted(top_k_indices(flow_x, top_k), key=lambda i: flow_x[i])
        flow_y = np.concatenate((flow_y[iy], flow_y[ix]))
        flow_x = np.concatenate((flow_x[iy], flow_x[ix]))
        return np.stack((flow_y, flow_x), axis=-1)
    return np.reshape(flow, (-1, flow.shape[-1]))


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
            video_features_seq = process_video(video_path, save_path=None, config=config)
            if video_features_seq is not None:
                for video_features in video_features_seq:
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
    features: list, shape (n_sequences, n_frames, n_features)
        list of feature matrices for each sequence in the video
    """
    global proc_count
    print('--- %d' % proc_count)
    print('[o] Video path: %s' % video_path)
    proc_count += 1

    # KTH sequence spec
    if len(seq_info) == 0:
        sequences_path = config.get('sequences_path', None)
        if sequences_path is not None:
            seq_info.update(read_sequences_file(sequences_path))
    frame_limits = seq_info.get(os.path.split(video_path)[-1], None)

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
    dense_params = config.get('dense_params', None)  # must be defined if using dense opt flow

    # Determine whether to use features
    use_edge = use_feature('edge', feature_toggles)
    use_centroid = use_feature('centroid', feature_toggles)
    use_optical_flow = use_feature('optical_flow', feature_toggles)
    use_freq_optical_flow = use_feature('freq_optical_flow', feature_toggles)
    use_dense_optical_flow = use_feature('dense_optical_flow', feature_toggles)

    # Load Shi-Tomasi and Lucas-Kanade configs
    st_config = _nondestructive_update(ST_PARAMS, st, disallow_strings=True)
    lk_config = _nondestructive_update(LK_PARAMS, lk, disallow_strings=True)

    # Main setup
    videogen = skvideo.io.vreader(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2() if fg_handler == 1 else None

    features = []
    features_indiv = defaultdict(list)  # ALL features in `features_indiv` will be coalesced
    avg = None
    prev_img = None
    p0 = None
    n_features = -1
    idx = 0
    try:
        for idx, frame in enumerate(videogen):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_edge = cv2.Canny(frame_gray, 0, 255)
            h, w = frame_gray.shape

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

            # [FEATURE] Edge
            # --------------
            if use_edge:
                edge_result = feat_edge(frame_gray, edges=frame_edge).flatten()
                features_indiv['edge'].append(edge_result)

            # [FEATURE] Centroid
            if use_centroid:
                features_indiv['centroid'].append(feat_centroid(frame_edge))

            # [FEATURE] Optical flow variants
            # -------------------------------
            if use_optical_flow or use_freq_optical_flow:
                if prev_img is None:
                    flow = np.zeros(n_bins)
                elif use_freq_optical_flow:
                    (flow_y, flow_x), p0 = feat_freq_optical_flow(
                        prev_img, frame_edge, p0, st_config, lk_config, n_bins)
                    flow = flow_y + flow_x
                else:
                    (flow_y, flow_x), p0 = feat_optical_flow(
                        prev_img, frame_edge, p0, st_config, lk_config)
                    flow = np.array([flow_y, flow_x]).flatten()
                optflow_k = 'freq_optical_flow' \
                    if use_freq_optical_flow else 'optical_flow'
                features_indiv[optflow_k].append(flow)
                prev_img = frame_edge.copy()
            elif use_dense_optical_flow:
                if prev_img is None:
                    top_k = dense_params.get('top_k', None)
                    mres_wind = dense_params.get('mres_wind', None)
                    if type(top_k) == int:
                        features_indiv['dense_optical_flow_y'].append(np.zeros(top_k * 2))
                        features_indiv['dense_optical_flow_x'].append(np.zeros(top_k * 2))
                    elif type(mres_wind) == int:
                        features_indiv['dense_optical_flow_y'].append(np.zeros(mres_wind ** 2))
                        features_indiv['dense_optical_flow_x'].append(np.zeros(mres_wind ** 2))
                    else:
                        roi_h = int(h * dense_params['roi_h'])
                        roi_w = int(w * dense_params['roi_w'])
                        features_indiv['dense_optical_flow_y'].append(np.zeros(roi_h * roi_w))
                        features_indiv['dense_optical_flow_x'].append(np.zeros(roi_h * roi_w))
                else:
                    flow = feat_dense_optical_flow(prev_img, frame_edge, dense_params)
                    features_indiv['dense_optical_flow_y'].append(flow[:, 0])
                    features_indiv['dense_optical_flow_x'].append(flow[:, 1])
                prev_img = frame_edge.copy()

        # Reduce dimensionality of edges
        if 'edge' in features_indiv:
            pca = PCA(n_components=edge_dim)
            edge_std = StandardScaler().fit_transform(np.array(features_indiv['edge']))
            features_indiv['edge'] = pca.fit_transform(edge_std)

        # Reduce dimensionality of dense optical flow fields
        for ax in ('y', 'x'):
            if 'dense_optical_flow_%s' % ax in features_indiv:
                n_components = dense_params.get('n_components', None)
                if type(n_components) == int:
                    pca = PCA(n_components=n_components)
                    d_optflow_std = StandardScaler().fit_transform(
                        np.array(features_indiv['dense_optical_flow_%s' % ax]))
                    features_indiv['dense_optical_flow_%s' % ax] = pca.fit_transform(d_optflow_std)

        # Normalization
        if normalize:
            for k, v in features_indiv.items():
                _mean = MEANS.get(k, 0.0)
                _stdev = STDEVS.get(k, 1.0)
                features_indiv[k] = (np.array(v) - _mean) / _stdev

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
        # - slice, trim, extend
        if frame_limits is None:
            print('[-] Warning: `frame_limits` is None. This is only a problem if using the KTH dataset.')
            frame_limits = ((0, idx),)
        features_seq = []
        for seq_start, seq_end in frame_limits:
            # Drop the first and final TRIM frames
            features_seq.append(features[seq_start+trim:seq_end+1-trim])
            pp_features = []
            for frame_feature_vec in features_seq[-1]:
                pp_vec = np.zeros(n_features)
                pp_vec[:frame_feature_vec.size] = frame_feature_vec
                pp_features.append(pp_vec)
            features_seq[-1] = np.array(pp_features)

        print('[+] Processed %d frames.' % (idx + 1))
        if verbose:
            print('Sample feature vector')
            print('---------------------')
            print(features_seq[0][-1, :])
            print('---------------------')
        print('Shape of video feature matrices: %r' % ([fm.shape for fm in features_seq],))

        if save_path is not None:
            h5f = h5py.File(save_path, 'w')
            h5f.create_dataset('features_seq', data=features_seq)
            h5f.close()
            print('[+] Saved features (for individual video `%s`) to %s.' % (video_path, save_path))

        return features_seq  # (n_sequences, n_frames, n_features)
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
