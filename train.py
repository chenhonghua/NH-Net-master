import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import sys
import os
import argparse

from model import Net 
from utils import log_write, cos_angle
from dataset import load_data, data_loader, load_data_cluster

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='training using clustered data input', help='description')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--path_model', type=str, default='./Models/model', help='Path to model.')
    parser.add_argument('--path_dataset', type=str, default='train', help='Path to train data (/train).')
    parser.add_argument('--id_cluster', type=int, default=1, help='Network for i-th cluster.')
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--rate_split', type=float, default=0.85, help='Random split for train/validation set.')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximun number of epochs to train.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum.')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay (L2 regularization term for optimizer).')
    parser.add_argument('--sampling_strategy', type=str, default='random', help='Orders in which the training samples are origanized:\n'
                    'random: randomly selected from the training dataset\n'
                    'full: without shuffle\n')
    parser.add_argument('--patience', type=int, default=30, help='Patience (epoch).')
    parser.add_argument('--normal_loss', type=str, default='mse_loss', help='Normal loss type:\n'
                        'mse_loss: element-wise mean square error\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')
    parser.add_argument('--normalize_output', type=int, default=False, help='Apply normalization on output normal.')

    return parser.parse_args()


def train(args):
    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))

    outdir = args.path_model
    indir = os.path.join(args.path_model, args.path_dataset)

    learning_rate = args.lr
    batch_size = args.batch_size

    # Data Loading
    HMP, Nf, Ng, idx_train, idx_val = load_data(path=indir, id_cluster=args.id_cluster, split=args.rate_split)

    ds_loader_train, ds_loader_val = data_loader(HMP, Nf, Ng, idx_train, idx_val, BatchSize=args.batch_size, sampling=args.sampling_strategy)
    nfeatures = int(Nf.shape[1]/3)

    # Create model
    net = Net(nfeatures)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    log_filename = os.path.join(outdir, 'log_trainC{}.txt'.format(args.id_cluster))
    model_filename = os.path.join(outdir, 'model_cluster{}.pth'.format(args.id_cluster))
    args_filename = os.path.join(outdir, 'args_cluster{}.pth'.format(args.id_cluster))

    # Training
    print("Training...")

    '''
    if os.path.exists(model_filename):
        response = input('A training instance ({}) already exists, overwrite? (y/n) '.format(model_filename))
        if response == 'y' or response == 'Y':
            if os.path.exists(log_filename):
                os.remove(log_filename)
            if os.path.exists(model_filename):
                os.remove(model_filename)
        else:
            print('Training exit.')
            sys.exit()'''

    if os.path.exists(model_filename):
        raise ValueError('A training instance already exists: {}'.format(model_filename))

    # LOG
    LOG_file = open(log_filename, 'w')
    log_write(LOG_file, str(args))
    log_write(LOG_file, 'data size = {}, train size = {}, val size = {}'.format(HMP.shape[0], np.sum(idx_train), np.sum(idx_val)))
    log_write(LOG_file, '***************************\n')

    train_batch_num = len(ds_loader_train)
    val_batch_num = len(ds_loader_val)

    min_error = 180
    epoch_best = -1
    bad_counter = 0

    for epoch in range(args.max_epochs):

        loss_cnt = 0
        err_cnt = 0
        cnt = 0

        # update learning rate
        scheduler.step()

        learning_rate = optimizer.param_groups[0]['lr']

        log_write(LOG_file, 'EPOCH #{}'.format(str(epoch+1)))
        log_write(LOG_file, 'lr = {}, batch size = {}'.format(learning_rate, batch_size))
        

        net.train()

        for i, inputs in enumerate(ds_loader_train):
            x, y, label = inputs

            x = x.to(device)
            y = y.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # forward backward
            output = net(x, y)

            loss = compute_loss(output, label, loss_type=args.normal_loss, normalize=args.normalize_output)
            loss.backward()
            optimizer.step()

            cnt += x.size(0)
            loss_cnt += loss.item()
            err = torch.abs(cos_angle(output, label)).detach().cpu().numpy()
            err = np.rad2deg(np.arccos(err))
            err_cnt += np.sum(err)

        train_loss = loss_cnt/train_batch_num
        train_err = err_cnt/cnt

        # validate
        net.eval()

        loss_cnt = 0
        err_cnt = 0
        cnt = 0

        for i, inputs in enumerate(ds_loader_val):
            x, y, label = inputs

            x = x.to(device)
            y = y.to(device)
            label = label.to(device)

            # forward
            with torch.no_grad():
                output = net(x, y)

            loss = compute_loss(output, label, loss_type=args.normal_loss, normalize=args.normalize_output)
            loss_cnt += loss.item()
            cnt += x.size(0)

            err = torch.abs(cos_angle(output, label)).detach().cpu().numpy()
            err = np.rad2deg(np.arccos(err))
            err_cnt += np.sum(err)

        val_loss = loss_cnt/val_batch_num
        val_err = err_cnt/cnt


        # log
        log_write(LOG_file, 'train loss = {}, train error = {}'.format(train_loss, train_err))
        log_write(LOG_file, 'val loss = {}, val error = {}'.format(val_loss, val_err))


        if min_error>val_err:
            min_error = val_err
            epoch_best = epoch+1
            bad_counter = 0
            log_write(LOG_file, 'Current best epoch #{} saved in file: {}'.format(epoch_best, model_filename), show_info=False)
            torch.save(net.state_dict(), model_filename)
        else:
            bad_counter += 1

        if bad_counter >= args.patience:
            break


def compute_loss(output, target, loss_type, normalize):

    loss = 0

    if normalize:
        output = F.normalize(output, dim=1)
        target = F.normalize(target, dim=1)

    if loss_type == 'mse_loss':
        loss += F.mse_loss(output, target)
    elif loss_type == 'ms_euclidean':
        loss += torch.min((output-target).pow(2).sum(1), (output+target).pow(2).sum(1)).mean()
    elif loss_type == 'ms_oneminuscos':
        loss += (1-torch.abs(cos_angle(output, target))).pow(2).mean()
    else:
        raise ValueError('Unsupported loss type: {}'.format(loss_type))

    return loss


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
