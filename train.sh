#!/bin/bash

#How to run
# sh train.sh folderName gpu
# example: sh train.sh AllVPNet_sample 0


folder=$1
gpu=$2


echo "Commencing training..."
echo "Folder: "$folder
echo "Using GPU: "$gpu
CUDA_VISIBLE_DEVICES=$gpu th main.lua -benchmark 0 -fromScratch 0 -modelDirName $folder/ -dropoutNet 0 -singleVPNet 0 -conditional 0 -silhouetteInput 0 -experiment 0 -maxEpochs 100 -targetBatchSize 8 -dataset nonbenchmark_ownCamPos -VAE 2_0_VAE_deepmerge

echo "Training Completed!"
echo "---------------Summary---------------"
echo "Folder: "$folder
echo "GPU: "$gpu