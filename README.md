# deepMerge
deepMerge: A model to reconstruct 3D model from depth map

## Dataset & Pretrained model
Download the [Dataset and pretrained model](https://www.dropbox.com/sh/vwn649s13cwy4yi/AADGCP_GejZzPVt5FfmM7OT8a?dl=0)

## Setting up
After downloading the Dataset and pretrained model, unzip the files and store the files in their respective repository.

**Dataset:**
Download the Dataset and store the folder in the Data/nonbenchmark_ownCamPos folder

The files and folder should be in the following format:
```
Data/nonbenchmark_ownCamPos/Datasets/test/Test-224x224-0.data
Data/nonbenchmark_ownCamPos/Datasets/test/Test-224x224-1.data
Data/nonbenchmark_ownCamPos/Datasets/test/Test-224x224-2.data

Data/nonbenchmark_ownCamPos/Datasets/train/Train-224x224-0.data
Data/nonbenchmark_ownCamPos/Datasets/train/Train-224x224-1.data
Data/nonbenchmark_ownCamPos/Datasets/train/Train-224x224-2.data
.
.
.
Data/nonbenchmark_ownCamPos/Datasets/train/Train-224x224-67.data

Data/nonbenchmark_ownCamPos/Datasets/validation/Valid-224x224-0.data
Data/nonbenchmark_ownCamPos/Datasets/validation/Valid-224x224-1.data
Data/nonbenchmark_ownCamPos/Datasets/validation/Valid-224x224-2.data
Data/nonbenchmark_ownCamPos/Datasets/validation/Valid-224x224-3.data
```


**Pretrained model:**
Download the pretrained_model for epoch 80, 90 and 100 and store the folders in the pretrained_model/model repository.

The files and folder should be in the following format:
```
pretrained_model/model/epoch80/mean_logvar.t7
pretrained_model/model/epoch80/model.t7

pretrained_model/model/epoch90/mean_logvar.t7
pretrained_model/model/epoch90/model.t7

pretrained_model/model/epoch100/mean_logvar.t7
pretrained_model/model/epoch100/model.t7
```


## How to run
In order to run the commands below, you need to download the Dataset and pretrained model and store them in their respective repositories

**1. Using the pretrained model**

run the command below:
- folderName = the name of the parent folder where the model resides (eg. pretrained_model)
- fromEpoch = select an epoch from which the model should be used for reconstructions (eg. 80)
- GPU = select a GPU to use from 0 to N (where N is the total number of GPUs available minus 1).
        set it to 0 if you only have one GPU (eg. 0)

> sh reconstruct.sh folderName fromEpoch GPU
```
sh reconstruct.sh pretrained_model 80 0
```

**2. Training the model from scratch**

run the command below:
- folderName = the name of the folder where the model will reside (eg. sampleModel)
- GPU = select a GPU to use from 0 to N (where N is the total number of GPUs available minus 1).
        set it to 0 if you only have one GPU (eg. 0)

> sh train.sh folderName GPU
```
sh train.sh sampleModel 0
```
