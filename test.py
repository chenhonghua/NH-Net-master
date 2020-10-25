import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import sys
import os
import argparse

from model import Net
from dataset import load_data_test, data_loader_test

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--path_model', type=str, default='./Models/pretrained', help='Path of pretrained models.')
    parser.add_argument('--path_result', type=str, default='normals', help='Saving predicted normals, if none (/normals) is used.')
    parser.add_argument('--path_dataset', type=str, default='test', help='Noisy point cloud to input.')
    parser.add_argument('--list_filenames', type=str, default='filenames.txt', help='Collecting filenames for predicted point clouds.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--sampling_strategy', type=str, default='full', help='orders in which the training samples are origanized:\n'
                    'random: randomly selected from the training dataset\n'
                    'full: without shuffle')

    return parser.parse_args()


def test(args):
    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))

    outdir = os.path.join(args.path_model, args.path_result)
    indir = os.path.join(args.path_model, args.path_dataset)
    list_filenames = os.path.join(indir, args.list_filenames)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Data Loading
    print('Load data from: {} ...'.format(indir))
    HMP, Nf, Rot, idx_cluster = load_data_test(indir)
    nfeatures = int(Nf.shape[1]/3)
    num_cluster = int(np.amax(idx_cluster))

    # read shape list
    with open(list_filenames) as f:
        files = f.readlines()
    filenames = []
    pnums = []
    for file in files:
        filenames.append(file.split()[0])
        pnums.append(int(file.split()[1]))
    total_point_num = Nf.shape[0]

    predict_normals = torch.zeros(total_point_num, 3)

    # create model
    net = Net(nfeatures)
    net.to(device)

    # predict
    for cluster_ in range(num_cluster):
        
        id_cluster = cluster_ + 1
        CL = (idx_cluster == id_cluster)

        ds_loader, sample_cluster_num = data_loader_test(HMP[CL,:,:,:], Nf[CL,:], Rot[CL,:,:], BatchSize=args.batch_size, sampling=args.sampling_strategy)

        model_filename = os.path.join(args.path_model, 'model_cluster{}.pth'.format(id_cluster))
        net.load_state_dict(torch.load(model_filename))

        # for each cluster
        normals_cluster = torch.zeros(sample_cluster_num, 3)

        net.eval()

        test_batch_num = len(ds_loader)
        offset = 0

        for batch_idx, inputs in enumerate(ds_loader):
            x, y, rot = inputs

            x = x.to(device)
            y = y.to(device)
            rot = rot.to(device)

            # forward
            with torch.no_grad():
                output = net.forward(x, y)

            # rotation applied
            normals_cluster[offset : offset+output.size(0)] = torch.matmul(rot.transpose(1, 2), output.unsqueeze(2)).squeeze().cpu()
            offset += output.size(0)

            if batch_idx%10 == 0:
                print('[%d: %d/%d]' % (id_cluster, batch_idx/10, (test_batch_num-1)/10))

        predict_normals[torch.from_numpy(np.array(CL, dtype=np.uint8)), :] = normals_cluster


    # collect data
    offset = 0
    for shape_id, shape_point_num in enumerate(pnums):
        output_filename = os.path.join(outdir, '{}.normals'.format(filenames[shape_id]))
        np.savetxt(output_filename, predict_normals[ offset : offset+shape_point_num ].numpy())
        offset += shape_point_num



if __name__ == '__main__':
    args = parse_arguments()
    test(args)
