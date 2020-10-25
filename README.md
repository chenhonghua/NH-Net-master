***Geometry and Learning Co-Supported Normal Estimation for Unstructured Point Cloud***
---

# Paper

Please acknowledge our paper :

"Geometry and Learning Co-Supported Normal Estimation for Unstructured Point Cloud" by Haoran Zhou, Honghua Chen et al., 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

# Dependencies

The code has been tested on Windows 10 with Anaconda, python 3.7.3, pytorch 1.1.0, torchvision 0.3.0, cudnn 7.6.4, cuda 10.0.130.

- matlab >= R2018a
- PyTorch


# Introduction

This is the code for point cloud normal estimation using MFPS (geometric estimate) and NH-Net (final normals). We provide a matlab implementation for the proposed geometric normal estimation method, which also collects data for the following network training process.

For the network module, we use pytorch implementation. It allows to be trained on noisy point cloud data provided with ground truth normals (we show one case using synthetic mesh model dataset: [Wang](https://wang-ps.github.io/denoising.html)). In the test stage, the network predicts unreoriented normals from unstructured points.


# Usage

Due to the third-party code we used, the data collection process for training is implemented by matlab (>=R2018a). Simply run `matlab/RUN.m` to collect training data and also estimate the geometric normals (in < PATH TO MODEL >/train/). Then, run `train.py --id_cluster 1` to train NH-Net for a single cluster (saved in < PATH TO MODEL >/). You may specify the cluster index to train your entire model of all clusters: `train.py --id_cluster <cluster_id>`. The train process is automatic.

In order to use your own trainset, you should specify the noisy data path and ground truth path in `matlab/RUN.m`.
    
    path_noisy = < PATH TO NOISY DATA >;
    path_original = < PATH TO GROUND TRUTH MESH >;

Also for test stage, first run `matlab/TEST.m` to collect data for the network and estimate geometric normals (in < PATH TO MODEL >/test/), and run `test.py` to predict the final normals for the whole testset (in < PATH TO MODEL >/normals/). In order to test on your own test set, you should specify the dataset path in `matlab/TEST.m`.

    test_noisy = < PATH TO NOISY DATA >;

The pretrained model is stored in `Models/pretrained/`. By specifying the desired model directory, you can use the pretrained models.

Note that, when training or testing, you should make sure that the model directory is the same in both matlab and python.

