# Flags
GET_KTH=true
GET_WEIZMANN=false

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
