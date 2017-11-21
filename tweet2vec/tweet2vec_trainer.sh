#!/bin/bash

# specify train and validation files here
traindata="../data/tweet2vec_trainer.txt"
valdata="../data/tweet2vec_tester.txt"

# specify model name here
exp="tweet2vec"

# model save path
modelpath="model/$exp/"
mkdir -p $modelpath

# train
echo "Training..."
python char.py $traindata $valdata $modelpath

