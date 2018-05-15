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

From a user perspective, the program is almost entirely specified by the input config file.
There are examples of such files in the `config` directory, e.g. [Owen's](https://github.com/ohjay/hmm_activity_recognition/blob/master/config/owen.yaml).
Descriptions of configurable parameters can be found below. Note that parameters are
grouped by their associated run command.

```yaml
extract_features:

  # Path to the video from which features should be extracted.
  # Only one of `base_dir`, `video_dir`, and `video_path` will be used;
  # `video_path` has the lowest priority, and is used primarily for debugging.
  video_path: /Users/owen/hmm_activity_recognition/data/kth/train/boxing/person01_boxing_d4_uncomp.avi

  # Path to the top-level folder containing all of the activity video subfolders.
  # Only one of `base_dir`, `video_dir`, and `video_path` will be used;
  # `base_dir` has the highest priority, and is used when training all of the HMMs.
  base_dir: /Users/owen/hmm_activity_recognition/data/kth/train

  # Path to the folder or file to which feature matrices should be saved.
  # Comment out if saving is not desirable, e.g. if debugging.
  save_path: /Users/owen/hmm_activity_recognition/features

  # Set these to True for a little (or a lot?) more output.
  debug: False
  verbose: False

  # Foreground estimation method.
  # 0 means no foreground estimation.
  # 1 means OpenCV background subtraction followed by a noise filter.
  # 2 means average subtraction.
  fg_handler: 0

  # Parameters for Shi-Tomasi corner detection.
  st:
    maxCorners: 200
    qualityLevel: 0.05
    minDistance: 3
    blockSize: 10

  # Parameters for Lucas-Kanade optical flow.
  lk:
    winSize: (15, 15)
    maxLevel: 2

  # Settings for feature inclusion.
  # Each feature will be included if its value is True, and excluded if its value is False.
  feature_toggles:
    optical_flow: False
    freq_optical_flow: False
    dense_optical_flow: True
    # If True, PCA will not be applied.
    freq_dense_optical_flow: False
    divergence: False
    curl: False
    avg_velocity: False
    edge: True
    centroid: True

  # Noise filtering parameters.
  # Used in the first foreground estimation method.
  denoise:
    kernel_size: 5
    threshold: 3

  # Dense optical flow parameters.
  dense_params:
    # Window height as fraction of total height.
    roi_h: 0.8
    # Window width as fraction of total width.
    roi_w: 0.4
    pyr_scale: 0.5
    levels: 3
    winsize: 15
    iterations: 3
    poly_n: 5
    poly_sigma: 1.2
    # Dimensionality after PCA.
    # If None, PCA is not used.
    n_components: 20
    # Number of maximal values to use for each flow direction.
    # If None, all values are used.
    top_k: None
    # Side length of square maximum response window.
    # If None, the entire ROI shall be considered.
    mres_wind: None

  # Divergence parameters.
  div_params:
    # If None, all divergence values are used
    # (instead of taking a histogram of divergence values).
    n_bins: None
    # If None, PCA is not used.
    pca_dim: 8

  # Curl parameters.
  curl_params:
    # If None, all curl values are used
    # (instead of taking a histogram of curl values).
    n_bins: None
    # If None, PCA is not used.
    pca_dim: 8

  # Dimensionality of `freq_optical_flow` feature.
  n_bins: 20

  # Dimensionality after PCA of `edge` feature.
  edge_dim: 20

  # Amount by which frames should be trimmed (at the beginning and the end).
  trim: 1

  # Whether features should be normalized
  # according to the means and standard deviations at the top of `extract_features.py`.
  normalize: True

  # Path to the sequences metadata file.
  sequences_path: /Users/owen/hmm_activity_recognition/data/kth/sequences.txt


build_models:

  # Path to the directory containing saved feature matrices.
  h5_dir: /Users/owen/hmm_activity_recognition/features

  # Path to the directory into which models should be saved.
  model_dir: /Users/owen/hmm_activity_recognition/models

  # Dimensionality to which all feature matrices will be standardized.
  # If using the `optical_flow` feature, this must be provided as an integer.
  # Otherwise, it's probably preferable to leave it unspecified (i.e. set to a non-integral value).
  n_features: infer

  # Whether stats should be computed for output normalization.
  compute_stats: False

  # Configuration for each model in the ensemble.
  # To add another model, just add a list item similar to the one(s) shown below.
  mconf:
      # Number of states in the HMM.
    - n_components: 4
      # Maximum number of iterations we're willing to run.
      n_iter: 20
      # Model type (either `gmm` or `gaussian`).
      m_type: gmm
      # Fraction of sequences the model should train on.
      subsample: 1.0


classify_activity:

  # Whether all activities should be classified, or just one.
  all: True

  # Path to video data. A directory if `all` is True, otherwise a video.
  path: /Users/owen/hmm_activity_recognition/data/kth/test

  # Path to folder in which models are saved.
  model_dir: /Users/owen/hmm_activity_recognition/models

  # Portion of videos that should be classified. Only meaningful if `all` is True.
  eval_fraction: 1.0

  # Dimensionality to which all feature matrices will be standardized.
  # If using the `optical_flow` feature, this must be provided as an integer.
  # Otherwise, it's probably preferable to leave it unspecified (i.e. set to a non-integral value).
  n_features: infer
```
