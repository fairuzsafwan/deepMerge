# deepMerge
deepMerge: A model to reconstruct 3D model from depth map by utilizing local and generic features

## Requirements
- [Torch](http://torch.ch/)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [cutorch](https://github.com/torch/cutorch)
- [cunn](https://github.com/torch/cunn)
- [paths](https://github.com/torch/paths)
- [image](https://github.com/torch/image)
- [optim](https://github.com/torch/optim)
- [gnuplot](https://github.com/torch/gnuplot)
- [openCV](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)

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


## Computing IoU (Windows only)

**1. Install the following requirements:**
- [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)
- [Visual C++ Redistributable Packages for Visual Studio 2013](https://www.microsoft.com/en-us/download/details.aspx?id=40784)
- [python](https://www.python.org/downloads/)
- [Anaconda](https://www.anaconda.com/distribution/)
- [openCV](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.2/opencv-3.4.2-vc14_vc15.exe/download)

**2. Register openCV path in environment variables**

**3. How to run**

run the command below:
> cd to the "zip-IoU" folder
- folderName = the input path of the folder where the estimated depth maps & silhouettes reside (eg. C:\Users\safwan\Desktop\model_v1)
- outputFolder = the output path where the 3D models will be reconstructed (eg. C:\Users\safwan\Desktop\outputModel_v1)
- experimentName = the name of the experiment folder where the 3D models will be reconstructed (eg. experimentSample)

> sh train.sh folderName outputFolder experimentName
```
run-loop_v3.bat C:\Users\safwan\Desktop\model_v1 C:\Users\safwan\Desktop\outputModel_v1 experimentSample
```

