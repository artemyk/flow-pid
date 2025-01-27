import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import pickle

import sys
sys.path.append('/Users/adithya/Documents/flow-pid')
from pid.models.fusions import Concat, Sequential2
from pid.models.unimodels import LeNet, MLP, LeNetEncoder, DeLeNet
from pid.optimizers.supervised_learning import train, single_test
from pid.optimizers.flow_pid import flow_pid
from pid.optimizers.tilde_pid import exact_gauss_tilde_pid

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def clustering(X, pca=False, n_clusters=30):
  X = np.nan_to_num(X)
  if len(X.shape) > 2:
    X = X.reshape(X.shape[0],-1)
  if pca:
    # print(np.any(np.isnan(X)), np.all(np.isfinite(X)))
    X = normalize(X)
    X = PCA(n_components=5).fit_transform(X)
  kmeans = KMeans(n_clusters=n_clusters).fit(X)
  return kmeans.labels_

def prepare_data(data_dir, modalities):
    all_data = np.load(data_dir)
    dims = [64,64,64]
    # dims = [256,256,256]
    
    dx = dims[modalities[0]]
    dy = dims[modalities[1]]
    dm = 1


    
    x_data = all_data[:, modalities[0]*64:(modalities[0]+1)*64]
    y_data = all_data[:, modalities[1]*64:(modalities[1]+1)*64]
    m_data = all_data[:, 64*3:]
    return x_data, y_data, m_data


if __name__ == '__main__':
    # assume features are already extracted
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    # x_data, y_data, m_data = prepare_data('./pretrained/humor/humor_features_mfm_64.npy', modalities = [1,2])
    data_dir = './data/humor/humor.pkl'
    dataset = pd.read_pickle(data_dir)
    print(dataset['train'].keys())

    data_cluster = dict()
    for split in dataset:
        if split != 'train':
            continue
        data_cluster[split] = dict()
        data = dataset[split]
        print(data['vision'].shape, data['audio'].shape, data['text'].shape)
        data_cluster[split]['vision'] = data['vision'].reshape(data['vision'].shape[0],-1)
        data_cluster[split]['audio'] = data['audio'].reshape(data['audio'].shape[0],-1)
        data_cluster[split]['text'] = data['text'].reshape(data['text'].shape[0],-1)
        data_cluster[split]['labels'] = data['labels']
        data_cluster[split]['id'] = data['id']
    
    x_data = data_cluster['train']['vision']
    y_data = data_cluster['train']['text']
    m_data = data_cluster['train']['labels']

    scale = 1/3.0
    eps = np.random.rand(m_data.shape[0], 1) * scale
    m_data = m_data.reshape(-1, 1) + eps
    print(x_data.shape, y_data.shape, m_data.shape)

    # mxy = np.vstack((m_data.T, x_data.T, y_data.T))
    # cov = np.corrcoef(mxy)    # Shape: (dm+dx+dy, dm+dx+dy)
    # print(f'cov.shape: {cov.shape} \n')

    # ret = exact_gauss_tilde_pid(cov, m_data.shape[1], x_data.shape[1], y_data.shape[1])
    # print(f'tilde pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')
    

    ret = flow_pid(m_data, x_data, y_data, n_flows=1, n_epochs=250, batch_size=1000, lr=1e-5, verbose=True, device=device)
    norm = ret[7] + ret[5] + ret[6] + ret[8]
    print(ret[7], ret[5], ret[6], ret[8])
    r, ux, uy, si = ret[7] / norm, ret[5] / norm, ret[6] / norm, ret[8] / norm
    print(f"flow pid, normalized I_mxy, R: {r}, UX: {ux}, UY: {uy}, S: {si}")



'''
RESULTS: (R, U1, U2, S)
Modalities [0,1]:
Tilde PID: 0.024774008932926226, 0.0009687824072653721, 2.8643805577432957e-08, 0.028909533385730626
Flow PID: 0.0 0.0266930092021487 0.02546858585407597 0.006298729901798694 

Modalities [0,2]:
Tilde PID:  0.025319963263488676, 0.0, 0.12088104305029784, 0.021446413849763202
Flow PID: 0.005431484230561556 0.020573832688725202 0.14632160416149584 0.004191151276086547

Modalities [1,2]:
Tilde PID: 0.02415196369461009, 0.0, 0.12339018476014167, 0.023594958415570744 
FLow PID: 0.0048521387660799 0.020305270412499188 0.14851273899309045 0.003731041530629947
'''

