import numpy as np
import torch
from sklearn.decomposition import PCA
from pid.optimizers import exact_gauss_thin_pid
from pid.models.unimodels import MLP, Linear
from pid.models.fusions import Concat, Sequential2, MultiplicativeInteractions2Modal
from pid.optimizers.supervised_learning import train, test  # noqa
from get_data import get_dataloader


import argparse


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./data/experiments/DATA_mix6.pickle", type=str,
                        help="input path of synthetic dataset")
    parser.add_argument("--keys", nargs='+', default=['0', '1', 'label'], type=str,
                        help="keys to access data of each modality and label, assuming dataset is structured as a dict")
    parser.add_argument("--modalities", nargs='+', default=[0, 1], type=int,
                        help="specify the index of modalities in keys")
    parser.add_argument("--bs", default=256, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--input-dim", nargs='+', default=[200], type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--output-dim", default=512, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--saved-model", default=None, type=str)
    parser.add_argument("--task", default='classification', type=str)
    parser.add_argument("--eval", default=True, type=int)
    parser.add_argument("--setting", default='mix6', type=str)
    parser.add_argument("--training", default=True, type=int, help="whether to train the model or not")
    args = parser.parse_args()

    # Load data
    traindata, validdata, _, testdata = get_dataloader(path=args.data_path, keys=args.keys, modalities=args.modalities,
                                                       batch_size=args.bs, num_workers=args.num_workers)

    # Specify model
    if len(args.input_dim) == 1:
        input_dims = args.input_dim * len(args.modalities)
    else:
        input_dims = args.input_dim
    encoders = [Linear(input_dim, args.output_dim).to(device) for input_dim in input_dims]
    head = MLP(args.output_dim, args.hidden_dim, args.num_classes).to(device)
    fusion = MultiplicativeInteractions2Modal([args.output_dim, args.output_dim], args.output_dim, 'matrix',
                                              flatten=True).to(device)

    args.saved_model = 'pretrained/mi_mix6.pt'

    # Training
    if args.training:
        train(encoders, fusion, head, traindata, validdata, args.epochs, optimtype=torch.optim.AdamW, early_stop=True,
              is_packed=False, lr=args.lr, save=args.saved_model, weight_decay=args.weight_decay,
              objective=torch.nn.CrossEntropyLoss())

    # Testing
    if args.eval:
        print("Testing:", args.saved_model)
        model = torch.load(args.saved_model).to(device)
        test(model, testdata, is_packed=False, no_robust=True, criterion=torch.nn.CrossEntropyLoss(), task=args.task)

        # PID estimate
        m_list = []
        x_list = []
        y_list = []
        zx_list = []
        zy_list = []
        for x, y, m in testdata:
            with torch.no_grad():
                z_x = model.encoders[0](x.float().to(device))
                z_y = model.encoders[1](y.float().to(device))
            zx_list.append(z_x)
            zy_list.append(z_y)
            x_list.append(x.float().to(device))
            y_list.append(y.float().to(device))
            m = m.float().view(-1, 1).to(device)
            # m = m + torch.rand_like(m, device=m.device)
            m_list.append(m)

        z_x = torch.cat(zx_list, dim=0).cpu().numpy()
        z_y = torch.cat(zy_list, dim=0).cpu().numpy()
        m = torch.cat(m_list, dim=0).cpu().numpy()
        x = torch.cat(x_list, dim=0).cpu().numpy()
        y = torch.cat(y_list, dim=0).cpu().numpy()

        dx, dy = 200, 200
        n_comps = 64  # Number of components to keep
        n_comps = min(dx, dy, n_comps)

        pca_x = PCA(n_components=n_comps)
        pca_y = PCA(n_components=n_comps)
        x_hat = pca_x.fit_transform(z_x)
        y_hat = pca_y.fit_transform(z_y)

        mxy_hat = np.hstack((m, x_hat, y_hat))
        cov_hat = np.cov(mxy_hat.T)
        print(f"Covariance matrix shape: {cov_hat.shape}")

        ret = exact_gauss_thin_pid(cov_hat, 1, n_comps, n_comps)
        print(f"Thin-PID: R: {ret[7]}, U1: {ret[5]}, U2: {ret[6]}, S: {ret[8]}")

        pca_x = PCA(n_components=n_comps)
        pca_y = PCA(n_components=n_comps)
        n_comps = min(dx, dy, n_comps)
        x = pca_x.fit_transform(x)
        y = pca_y.fit_transform(y)

        mxy = np.hstack((m, x, y))
        cov = np.cov(mxy.T)
        print(f"Covariance matrix shape: {cov.shape}")

        ret = exact_gauss_thin_pid(cov, 1, n_comps, n_comps)
        print(f"Truth: R: {ret[7]}, U1: {ret[5]}, U2: {ret[6]}, S: {ret[8]}")


    # Thin - PID: R: 1.1468324531108132, U1: 8.881784197001252e-16, U2: 1.0054297451788456, S: 0.28547980181331134
    # Truth: R: 0.9052703333529206, U1: 2.6645352591003757e-15, U2: 1.0757119289966028, S: 0.08948968471345076