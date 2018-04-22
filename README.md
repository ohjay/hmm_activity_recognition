# HMM Activity Recognition

EE 126 project, with William Jow and David Lin. [[Report]](http://owenjow.xyz/hmm_activity_recognition/report.pdf)

## Usage

To download the KTH dataset, run `./get_data.sh`.
You may want to split the data into train and test subsets,
which we have hitherto done manually on our end. Note that the
[sequences metadata file](http://www.nada.kth.se/cvap/actions/00sequences.txt)
will need to be obtained separately. Do not rename the files or tamper with
the directory structure, as the program parses activities from filenames and
expects a data folder with six subfolders in it (one for each activity's videos).

```
python main.py extract  <path to config>  # extract features
python main.py build    <path to config>  # build models
python main.py classify <path to config>  # classify activity
```

## Config

From a user perspective, the program is almost fully specified by the input config file.
There are examples of such files in the `config` directory, e.g. [Owen's config file](https://github.com/ohjay/hmm_activity_recognition/blob/master/config/owen.yaml).
Descriptions of configurable parameters can be found below. Note that parameters are
grouped by their associated run command.

```yaml
extract_features:

  # Path to the video from which features should be extracted.
  # Only one of `base_dir`, `video_dir`, and `video_path` will be used;
  # `video_path` has the lowest priority, and is used primarily for debugging.
  video_path: string, e.g. /Users/owen/hmm_activity_recognition/data/kth/train/walking/person01_walking_d4_uncomp.avi

  # Path to the data folder, which should contain activity video subfolders.
  # Only one of `base_dir`, `video_dir`, and `video_path` will be used;
  # `base_dir` has the highest priority, and is used when training all of the HMMs.
  base_dir: string, e.g. /Users/owen/hmm_activity_recognition/data/kth/train

  # Path to the folder or file to which feature matrices should be saved.
  # Comment out if saving is not desirable (e.g. if debugging).
  save_path: string, e.g. /Users/owen/hmm_activity_recognition/features

  # Set these to True for a little more output.
  debug: boolean, e.g. False
  verbose: boolean, e.g. False

  # Method of foreground estimation.
  #   0: no foreground estimation
  #   1: OpenCV background subtractor + noise filter
  #   2: average subtraction
  fg_handler: int, e.g. 0

  # Parameters for Shi-Tomasi corner detection.
  st:
    maxCorners: int, e.g. 200
    qualityLevel: float, e.g. 0.05
    minDistance: int, e.g. 3
    blockSize: int, e.g. 10

  # Parameters for Lucas-Kanade optical flow.
  lk:
    winSize: tuple, e.g. (15, 15)
    maxLevel: int, e.g. 2

  # Settings for feature inclusion.
  # A feature will be included if its value is True, and excluded if its value is False.
  feature_toggles:
    optical_flow: boolean, e.g. False
    freq_optical_flow: boolean, e.g. False
    dense_optical_flow: boolean, e.g. True
    edge: boolean, e.g. True
    shape: boolean, e.g. False
    centroid: boolean, e.g. True

  # Noise filtering parameters. Used for foreground method #1.
  denoise:
    kernel_size: int, e.g. 5
    threshold: int, e.g. 3

  # Dense optical flow parameters.
  #   roi_h: window height as fraction of total height
  #   roi_w: window width as fraction of total width
  #   n_components: dimensionality after PCA
  dense_params:
    roi_h: float, e.g. 0.8
    roi_w: float, e.g. 0.4
    pyr_scale: float, e.g. 0.5
    levels: int, e.g. 3
    winsize: int, e.g. 15
    iterations: int, e.g. 3
    poly_n: int, e.g. 5
    poly_sigma: float, e.g. 1.2
    n_components: int, e.g. 20

  # Desired dimensionality of frequency optical flow feature.
  n_bins: int, e.g. 20

  # Desired dimensionality of edge feature.
  edge_dim: int, e.g. 20

  # Amount by which frames should be trimmed (at the beginning and the end).
  trim: int, e.g. 1

  # Whether features should be normalized.
  # according to the means and standard deviations at the top of `extract_features.py`.
  normalize: boolean, e.g. True

  # Path to the sequences metadata file.
  sequences_path: string, e.g. /Users/owen/hmm_activity_recognition/data/kth/sequences.txt


build_models:

  # Path to the directory containing saved feature matrices.
  h5_dir: string, e.g. /Users/owen/hmm_activity_recognition/features

  # Path to the directory into which models should be saved.
  model_dir: string, e.g. /Users/owen/hmm_activity_recognition/models

  # Dimensionality to which all feature matrices will be standardized.
  # If using the `optical_flow` feature, this must be provided (as an integer).
  # Otherwise, it's probably preferable to leave it unspecified (set to a non-integral value).
  n_features: int (or anything), e.g. infer

  # Whether stats should be computed for output normalization.
  compute_stats: boolean, e.g. False

  # Configuration for each model in the ensemble.
  # To add a new model, just add a list item similar to the one shown below.
  mconf:
    - n_components: int, e.g. 4
      n_iter: int, e.g. 20
      m_type: string, e.g. gmm
      subsample: float, e.g. 1.0


classify_activity:

  # Whether all activities should be classified, or just one.
  all: boolean, e.g. True

  # Path to video data. A directory if `all` is True, otherwise a video.
  path: string, e.g. /Users/owen/hmm_activity_recognition/data/kth/test

  # Path to folder in which models are saved.
  model_dir: string, e.g. /Users/owen/hmm_activity_recognition/models

  # Portion of videos that should be classified. Only meaningful if `all` is True.
  eval_fraction: float, e.g. 1.0

  # Dimensionality to which all feature matrices will be standardized.
  # If using the `optical_flow` feature, this must be provided (as an integer).
  # Otherwise, it's probably preferable to leave it unspecified (set to a non-integral value).
  n_features: int (or anything), e.g. infer
```
