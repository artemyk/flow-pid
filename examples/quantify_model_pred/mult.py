import numpy as np
import torch
from sklearn.decomposition import PCA
from pid.optimizers import exact_gauss_thin_pid
from pid.models.unimodels import Identity
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


    class HParams():
        num_heads = 8
        layers = 4
        attn_dropout = 0.1
        attn_dropout_modalities = [0, 0]
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.1
        embed_dropout = 0.2
        embed_dim = args.hidden_dim
        attn_mask = True
        output_dim = args.num_classes
        all_steps = False


    # Specify model
    if len(args.input_dim) == 1:
        input_dims = args.input_dim * len(args.modalities)
    else:
        input_dims = args.input_dim
    encoders = [Identity().to(device) for _ in args.modalities]
    fusion = MULTModel(len(args.modalities), input_dims, hyp_params=HParams).to(device)
    head = Identity().to(device)

    # Training
    train(encoders, fusion, head, traindata, validdata, args.epochs, optimtype=torch.optim.AdamW, is_packed=False,
          lr=args.lr, clip_val=1.0, save=args.saved_model, weight_decay=args.weight_decay,
          objective=torch.nn.CrossEntropyLoss())

    # Testing
    print("Testing:")
    model = torch.load(args.saved_model).to(device)
    test(model=model, test_dataloaders_all=testdata, is_packed=False, criterion=torch.nn.CrossEntropyLoss(),
         no_robust=True)