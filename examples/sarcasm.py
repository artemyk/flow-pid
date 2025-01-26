import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from pid.models.fusions import Concat, Sequential2
from pid.models.unimodels import LeNet, MLP, LeNetEncoder, DeLeNet
from pid.optimizers.supervised_learning import train, single_test
from pid.optimizers.flow_pid import flow_pid
from pid.optimizers.tilde_pid import exact_gauss_tilde_pid



def prepare_data(data_dir, modalities):
    all_data = np.load(data_dir)
    dims = [64,64,64]
    
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

    x_data, y_data, m_data = prepare_data('./pretrained/sarcasm/sarcasm_features_mfm.npy', modalities = [1,2])

    scale = 1/3.0
    eps = np.random.rand(m_data.shape[0], 1) * scale
    m_data = m_data.reshape(-1, 1) + eps
    print(x_data.shape, y_data.shape, m_data.shape)

    mxy = np.vstack((m_data.T, x_data.T, y_data.T))
    cov = np.corrcoef(mxy)    # Shape: (dm+dx+dy, dm+dx+dy)
    print(f'cov.shape: {cov.shape} \n')

    ret = exact_gauss_tilde_pid(cov, m_data.shape[1], x_data.shape[1], y_data.shape[1])
    print(f'tilde pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')
    

    ret = flow_pid(m_data, x_data, y_data, n_flows=20, n_epochs=1000, batch_size=128, lr=2e-4, verbose=True, device=device)
    norm = ret[7] + ret[5] + ret[6] + ret[8]
    print(ret[7], ret[5], ret[6], ret[8])
    r, ux, uy, si = ret[7] / norm, ret[5] / norm, ret[6] / norm, ret[8] / norm
    print(f"flow pid, normalized I_mxy, R: {r}, UX: {ux}, UY: {uy}, S: {si}")



'''
RESULTS: (R, U1, U2, S)
Modalities [0,1]:
Tilde PID: 0.11939196920057815, 0.04489586164333269, 0.0, 0.15243772539941902 
Flow PID: 0.012377845298483847 0.15521206033589413 0.10864323213595867 0.03592769911066762

Modalities [0,2]:
Tilde PID:  0.15721594111628553, 0.018078612459983656, 0.0, 0.18724224530828606 
Flow PID: 0.02364692935879925 0.14705045078967616 0.1641617918789335 0.05124527439076615

Modalities [1,2]:
Tilde PID: 0.12763541467810505, 0.0, 0.026070926449911463, 0.1435946156085531 
FLow PID: 0.02523528175487788 0.10100008910507285 0.13798292502492332 0.05245861661146917
'''

