#!/usr/bin/env bash

# ------------------------------------------------------------
# DATA ACQUISITION
# ------------------------------------------------------------

# Download the KTH dataset and set up train/validation splits.
./get_data.sh

# ------------------------------------------------------------
# VIRTUAL ENVIRONMENT SETUP
# ------------------------------------------------------------

# Create a virtual environment.
virtualenv -p python2.7 ./hmm-activity-recog-env

# Activate the virtual environment.
source ./hmm-activity-recog-env/bin/activate

# Upgrade pip.
if [ "$(uname)" == "Darwin" ]; then
  curl https://bootstrap.pypa.io/get-pip.py | python
fi
pip install --upgrade pip

# Install setuptools.
pip install -U pip setuptools
pip install setuptools_scm

# Install requests[security].
pip install requests
pip install 'requests[security]'

# Install correct versions of Python modules.
pip install -r requirements.txt

# ------------------------------------------------------------
# THE FUN STUFF
# ------------------------------------------------------------

echo  # (blank line)

# Extract features from videos.
python main.py extract config/quickstart.yaml

echo  # (blank line)

# Build per-class models from corresponding video features.
python main.py build config/quickstart.yaml

echo  # (blank line)

# Evaluate the models on the validation dataset.
python main.py classify config/quickstart.yaml

# Deactivate the virtual environment.
deactivate
