#! /bin/sh

perfectpitch_commit="b16778f8882dacb3907e62505ccc5f82dd78e92b"
experiment_url="gs://perfectpitch/experiments/test-1/"
train_url="gs://perfectpitch/dataset/perfectpitch_maestro_v1.0.0/train.hdf5"
validation_url="gs://perfectpitch/dataset/perfectpitch_maestro_v1.0.0/validation.hdf5"
train_path="/tmp/perfectpitch_maestro_v1.0.0/train.hdf5"
validation_path="/tmp/perfectpitch_maestro_v1.0.0/validation.hdf5"

echo
echo
echo "START OF INSTALLING PERFECTPITCH"
sudo apt install -qqy libsndfile1
pip install git+git://github.com/ArmanMazdaee/perfectpitch.git@$perfectpitch_commit
echo "END OF INSTALLING PERFECTPITCH"

echo
echo
echo "START OF DOWNLOADING DATASETS"
gsutil -q cp $train_url $train_path
gsutil -q cp $validation_url $validation_path
echo "END OF DOWNLOADING DATASETS"

echo
echo
echo "START OF TRAINING THE MODEL"
perfectpitch train-acoustic --train-dataset=$train_path --validation-dataset=$validation_path --qpu
echo "END OF TRAINING THE MODEL"