#!/usr/bin/env bash

# Flags
GET_KTH=true
GET_WEIZMANN=false
GET_KTH_NORM_STATS=true
SET_UP_KTH_TRAIN_TEST_SPLIT=true

mkdir -p data

# Get KTH
if [ "$GET_KTH" = true ]; then
  echo "Getting KTH dataset..."
  mkdir -p data/kth
  wget -P data/kth http://www.nada.kth.se/cvap/actions/walking.zip
  wget -P data/kth http://www.nada.kth.se/cvap/actions/jogging.zip
  wget -P data/kth http://www.nada.kth.se/cvap/actions/running.zip
  wget -P data/kth http://www.nada.kth.se/cvap/actions/boxing.zip
  wget -P data/kth http://www.nada.kth.se/cvap/actions/handwaving.zip
  wget -P data/kth http://www.nada.kth.se/cvap/actions/handclapping.zip

  for f in data/kth/*.zip; do
    dir=${f%.zip}
    unzip -d "./$dir" "./$f"
    rm $f
  done

  wget -O data/kth/sequences.txt http://www.nada.kth.se/cvap/actions/00sequences.txt

  if [ "$SET_UP_KTH_TRAIN_TEST_SPLIT" = true ]; then
    echo "Configuring KTH train/test split..."
    for action in walking jogging running boxing handwaving handclapping; do
      mkdir -p data/kth/train/${action}
      for i in $(seq -f "%02g" 1 16); do
        mv data/kth/${action}/person${i}* data/kth/train/${action}
      done
      mkdir -p data/kth/test/${action}
      for i in $(seq -f "%02g" 17 25); do
        mv data/kth/${action}/person${i}* data/kth/test/${action}
      done
      rmdir data/kth/${action}
    done
  fi

  if [ "$GET_KTH_NORM_STATS" = true ]; then
    echo "Getting precomputed KTH norm stats..."
    wget -P data/kth https://github.com/ohjay/hmm_activity_recognition/files/3036370/norm_stats.zip --no-check-certificate
    unzip -d data/kth data/kth/norm_stats.zip
    rm data/kth/norm_stats.zip
  fi
fi

# Get Weizmann
if [ "$GET_WEIZMANN" = true ]; then
  echo "Getting Weizmann dataset..."
  mkdir -p data/weizmann
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/walk.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/run.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/jump.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/side.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/bend.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/wave1.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/wave2.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/pjump.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/jack.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/skip.zip
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/classification_masks.mat
  wget -P data/weizmann http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/backgrounds.zip

  for f in data/weizmann/*.zip; do
    dir=${f%.zip}
    unzip -d "./$dir" "./$f"
    rm $f
  done
fi
