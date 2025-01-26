import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from pid.optimizers.flow_pid import flow_pid
from pid.optimizers.tilde_pid import exact_gauss_tilde_pid



def prepare_data(data_dir):
    all_data = np.load(data_dir)
    dims = [16,16]

    x_data = all_data[:, :dims[0]]
    y_data = all_data[:, dims[0]:dims[0]+dims[1]]
    m_data = all_data[:, dims[0]+dims[1]:]
    return x_data, y_data, m_data


if __name__ == '__main__':
    # assume features are already extracted
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    x_data, y_data, m_data = prepare_data('./pretrained/enrico/enrico_features.npy')

    scale = 1/2.0*0
    eps = np.random.rand(m_data.shape[0], 1) * scale
    m_data = m_data.reshape(-1, 1) + eps
    print(x_data.shape, y_data.shape, m_data.shape)

    mxy = np.vstack((m_data.T, x_data.T, y_data.T))
    cov = np.corrcoef(mxy)    # Shape: (dm+dx+dy, dm+dx+dy)
    print(f'cov.shape: {cov.shape} \n')

    ret = exact_gauss_tilde_pid(cov, m_data.shape[1], x_data.shape[1], y_data.shape[1])
    print(f'tilde pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')
    

    ret = flow_pid(m_data, x_data, y_data, n_flows=10, n_epochs=2000, batch_size=1000, lr=2e-3, verbose=True, device=device)
    norm = ret[7] + ret[5] + ret[6] + ret[8]
    print(ret[7], ret[5], ret[6], ret[8])
    r, ux, uy, si = ret[7] / norm, ret[5] / norm, ret[6] / norm, ret[8] / norm
    print(f"flow pid, normalized I_mxy, R: {r}, UX: {ux}, UY: {uy}, S: {si}")



'''
RESULTS: (R, U1, U2, S)
Tilde PID: 0.3606968946941195, 0.0882106535427345, 0.0, 0.14927728064635393 
Flow PID: 0.186392074251296 0.30056444756017114 0.12172080169204297 0.0
'''

