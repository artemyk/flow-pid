import numpy as np
import torch
from sklearn.decomposition import PCA
from pid import exact_gauss_thin_pid
from ensemble import train, test  # noqa
from get_data import get_dataloader
import argparse


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./data/experiments/DATA_mix5.pickle", type=str,
                        help="input path of synthetic dataset")
    parser.add_argument("--keys", nargs='+', default=['0', '1', 'label'], type=str,
                        help="keys to access data of each modality and label, assuming dataset is structured as a dict")
    parser.add_argument("--modalities", nargs='+', default=[0, 1], type=int,
                        help="specify the index of modalities in keys")
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--input-dim", nargs='+', default=[100], type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--n-latent", default=600, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--eval", default=True, type=int)
    parser.add_argument("--setting", default='mix6', type=str)
    parser.add_argument("--weight", default=1, type=float)
    parser.add_argument("--saved-model", default=None, type=str)
    parser.add_argument("--training", default=True, type=int, help="whether to train the model or not")
    args = parser.parse_args()

    # Load data
    traindata, validdata, _, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities,
                                                       batch_size=args.bs, num_workers=args.num_workers)

    # PID estimate
    m_list = []
    x_list = []
    y_list = []
    for x, y, m in traindata:
        x_list.append(x.float())
        y_list.append(y.float())
        m = m.float().view(-1, 1)
        # m = m + torch.rand_like(m)
        m_list.append(m)

    m = torch.cat(m_list, dim=0).numpy()
    x = torch.cat(x_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()

    dx, dy = 100, 100
    n_comps = 64  # Number of components to keep
    n_comps = min(dx, dy, n_comps)

    pca_x = PCA(n_components=n_comps)
    pca_y = PCA(n_components=n_comps)
    x = pca_x.fit_transform(x)
    y = pca_y.fit_transform(y)

    mxy_hat = np.hstack((m, x, y))
    cov_hat = np.cov(mxy_hat.T)
    print(f"Covariance matrix shape: {cov_hat.shape}")

    ret = exact_gauss_thin_pid(cov_hat, 1, n_comps, n_comps)
    print(f"Thin-PID: R: {ret[7]}, U1: {ret[5]}, U2: {ret[6]}, S: {ret[8]}")
