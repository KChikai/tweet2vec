#!/bin/bash

# specify data file here
datafile="../data/encoder_example.txt"

# specify model path here
modelpath="model/tweet2vec/"

# specify result path here
resultpath="tweet_encoding"

mkdir -p $resultpath

# test
python encode_char.py $datafile $modelpath $resultpath
