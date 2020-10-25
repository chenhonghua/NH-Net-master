import numpy as np
import torch
import torch.utils.data as Data
import os


def load_data(path, id_cluster, split):
    # Load data from file
    idx_cluster = np.load(os.path.join(path, 'idx_cluster.npy'))
    Nf = np.load(os.path.join(path, 'Nf.npy'))
    HMP = np.load(os.path.join(path, 'HMP.npy'))
    Ng = np.load(os.path.join(path, 'Ng.npy'))

    idx_cluster = np.squeeze(idx_cluster)
    CL = (idx_cluster == id_cluster)
    HMP, Nf, Ng = HMP[CL,:,:,:], Nf[CL,:], Ng[CL,:]

    # random split
    assert split >= 0 and split <= 1, 'rate_split should be in (0,1).'
    data_size = HMP.shape[0]
    p = np.random.permutation(data_size)
    idx_train = np.zeros(data_size, dtype=bool)
    idx_train[ p[ 0 : int(data_size*split) ] ] = True
    idx_val = ~idx_train

    return HMP, Nf, Ng, idx_train, idx_val


def data_loader(HMP, Nf, Ng, idx_train, idx_val, BatchSize, sampling='random'):
    # numpy data to torch
    inputx_train = torch.from_numpy(HMP[idx_train,:,:,:])
    inputy_train = torch.from_numpy(Nf[idx_train,:])
    targets_train = torch.from_numpy(Ng[idx_train,:])
    inputx_val = torch.from_numpy(HMP[idx_val,:,:,:])
    inputy_val = torch.from_numpy(Nf[idx_val,:])
    targets_val = torch.from_numpy(Ng[idx_val,:])

    if sampling == 'random':
        ds_train = Data.TensorDataset(inputx_train, inputy_train, targets_train)
        ds_loader_train = Data.DataLoader(ds_train, batch_size=BatchSize, shuffle=True)
        ds_val = Data.TensorDataset(inputx_val, inputy_val, targets_val)
        ds_loader_val = Data.DataLoader(ds_val, batch_size=BatchSize, shuffle=True)
    elif sampling == 'full':
        ds_train = Data.TensorDataset(inputx_train, inputy_train, targets_train)
        ds_loader_train = Data.DataLoader(ds_train, batch_size=BatchSize, shuffle=False)
        ds_val = Data.TensorDataset(inputx_val, inputy_val, targets_val)
        ds_loader_val = Data.DataLoader(ds_val, batch_size=BatchSize, shuffle=False)
    else:
        raise ValueError('Unknown sampling strategy: {}'.format(sampling))

    return ds_loader_train, ds_loader_val


def load_data_test(path):
    # Load data from file
    idx_cluster = np.load(os.path.join(path, 'idx_cluster.npy'))
    Nf = np.load(os.path.join(path, 'Nf.npy'))
    HMP = np.load(os.path.join(path, 'HMP.npy'))
    Rot = np.load(os.path.join(path, 'Rot.npy'))

    idx_cluster = np.squeeze(idx_cluster)

    assert HMP.shape[0] == Nf.shape[0] and Nf.shape[0] == Rot.shape[0], 'Unmatched data size.'

    return HMP, Nf, Rot, idx_cluster


def data_loader_test(HMP, Nf, Rot, BatchSize, sampling='full'):
    # numpy data to torch
    inputx = torch.from_numpy(HMP)
    inputy = torch.from_numpy(Nf)
    inputr = torch.from_numpy(Rot)
    sample_num = inputx.size(0)

    if sampling == 'random':
        ds_test = Data.TensorDataset(inputx, inputy, inputr)
        ds_loader_test = Data.DataLoader(ds_test, batch_size=BatchSize, shuffle=True)
    elif sampling == 'full':
        ds_test = Data.TensorDataset(inputx, inputy, inputr)
        ds_loader_test = Data.DataLoader(ds_test, batch_size=BatchSize, shuffle=False)
    else:
        raise ValueError('Unknown sampling strategy: {}'.format(sampling))

    return ds_loader_test, sample_num


