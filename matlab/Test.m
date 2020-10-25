clear;clc;close all;
addpath('kdtree');
addpath('IO');
addpath('cvx');
addpath('npy-matlab-master/npy-matlab');
addpath('HMP');
cvx_setup

%% configuration
ModelName = 'pretrained';

% path
test_noisy = '../test/'; % < PATH TO NOISY DATA >

test_model = ['../Models/', ModelName, '/'];
test_result = [test_model, 'test/'];
warning off MATLAB:MKDIR:DirectoryExists
mkdir(test_result);

% p
Ks = [50, 100, 150];
% Ks2 = [100, 200, 300];


%% Collect data for testing

load([test_model, 'Estimator.mat']);

parobj = parpool(4);

Estimator.test(test_noisy, test_result, Ks);

delete(parobj);

