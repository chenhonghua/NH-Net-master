clear;clc;close all;
addpath('kdtree');
addpath('IO');
addpath('cvx');
addpath('npy-matlab-master/npy-matlab');
addpath('HMP');
cvx_setup

%% setting
% model
ModelName = 'model';

% path
% path_original = 'D:\mesh_denoising\Synthetic\train\original\';
% path_noisy = 'D:\mesh_denoising\Synthetic\train\noisy\';

path_noisy = '';  % < PATH TO NOISY DATA >
path_original = '';  % < PATH TO GROUND TRUTH MESH >

path_model = ['../Models/', ModelName, '/'];
path_result = [path_model, 'results/'];
path_train = [path_model, 'train/'];
warning off MATLAB:MKDIR:DirectoryExists
mkdir(path_result); mkdir(path_train);

% parameter
Ks = [50, 100, 150];
% Ks2 = [100, 200, 300];
sigma_s = [1.0, 2.0];
sigma_r = [0.1, 0.2, 0.35, 0.5];
rotate_feature = true; self_included = true;
map_size = 7;

% cluster
pca_k = 3; cluster_k = 4; cluster_threshold = 5;

%% Collect data for training

Estimator = GeoNormal(sigma_s, sigma_r, rotate_feature, self_included, ...
                map_size, pca_k, cluster_k, cluster_threshold);
parobj = parpool(4);

Estimator.run(path_noisy, path_original, path_result, path_train, Ks);

delete(parobj);


% save estimator
save([path_model, 'Estimator.mat'], 'Estimator');

