#!/bin/bash

# specify test file here
fulltestdata="../data/tweet2vec_tester.txt"

# specify model path here
modelpath="model/tweet2vec/"

# specify result path here
resultpath="result/"

mkdir -p $resultpath

# test
python test_char.py $fulltestdata $modelpath $resultpath
