#!/bin/bash

#How to run
# sh reconstruct.sh folderName fromEpoch gpu
# example: sh reconstruct.sh AllVPNet_sample 80 0


folder=$1
epoch=$2
gpu=$3


echo "Commencing 3D reconstruction..."
echo "Folder: "$folder
echo "Reconstruct at Epoch: "$epoch
echo "Using GPU: "$gpu
CUDA_VISIBLE_DEVICES=$gpu th main.lua -modelDirName $folder/ -experiment 1 -conditional 0 -expType forwardPass -forwardPassType reconstructAllSamples -dropoutNet 0 -singleVPNet 0 -fromEpoch $epoch -dataset nonbenchmark_ownCamPos


echo "3D reconstruction Completed!"
echo "---------------Summary---------------"
echo "Folder: "$folder
echo "Reconstruct at Epoch: "$epoch
echo "GPU: "$gpu