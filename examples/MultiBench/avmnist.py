import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.fusions import Concat
from models.unimodels import LeNet, MLP
from pid.supervised_learning import train, single_test
from pid.flow_pid import flow_pid


def prepare_data(data_dir, flatten_audio=False, flatten_image=False,
                   unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):
    x_data = np.load(data_dir + "/image/train_data.npy")
    y_data = np.load(data_dir + "/audio/train_data.npy")
    m_data = np.load(data_dir + "/train_labels.npy")

    if flatten_audio:
        y_data = y_data.reshape(60000, 112 * 112)
    if generate_sample:
        _saveimg(x_data[0:100])
        _saveaudio(y_data[0:9].reshape(9, 112 * 112))
    if normalize_image:
        x_data /= 255.0
    if normalize_audio:
        y_data = y_data / 255.0
    if not flatten_image:
        x_data = x_data.reshape(60000, 28, 28)
    if unsqueeze_channel:
        x_data = np.expand_dims(x_data, 1)
        y_data = np.expand_dims(y_data, 1)
    m_data = m_data.astype(int)

    return x_data, y_data, m_data


def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True, flatten_audio=False, flatten_image=False,
                   unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):
    """Get dataloaders for AVMNIST.

    Args:
        data_dir (str): Directory of data.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        flatten_audio (bool, optional): Whether to flatten audio data or not. Defaults to False.
        flatten_image (bool, optional): Whether to flatten image data or not. Defaults to False.
        unsqueeze_channel (bool, optional): Whether to unsqueeze any channels or not. Defaults to True.
        generate_sample (bool, optional): Whether to generate a sample and save it to file or not. Defaults to False.
        normalize_image (bool, optional): Whether to normalize the images before returning. Defaults to True.
        normalize_audio (bool, optional): Whether to normalize the audio before returning. Defaults to True.

    Returns:
        tuple: Tuple of (training dataloader, validation dataloader, test dataloader)
    """
    trains = [np.load(data_dir + "/image/train_data.npy"), np.load(data_dir +
                                                                   "/audio/train_data.npy"),
              np.load(data_dir + "/train_labels.npy")]
    tests = [np.load(data_dir + "/image/test_data.npy"), np.load(data_dir +
                                                                 "/audio/test_data.npy"),
             np.load(data_dir + "/test_labels.npy")]
    if flatten_audio:
        trains[1] = trains[1].reshape(60000, 112 * 112)
        tests[1] = tests[1].reshape(10000, 112 * 112)
    if generate_sample:
        _saveimg(trains[0][0:100])
        _saveaudio(trains[1][0:9].reshape(9, 112 * 112))
    if normalize_image:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_audio:
        trains[1] = trains[1] / 255.0
        tests[1] = tests[1] / 255.0
    if not flatten_image:
        trains[0] = trains[0].reshape(60000, 28, 28)
        tests[0] = tests[0].reshape(10000, 28, 28)
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    trainlist = [[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist = [[tests[j][i] for j in range(3)] for i in range(10000)]
    valids = DataLoader(trainlist[55000:60000], shuffle=False,
                        num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    tests = DataLoader(testlist, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    trains = DataLoader(trainlist[0:55000], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    return trains, valids, tests


# this function creates an image of 100 numbers in avmnist


def _saveimg(outa):
    from PIL import Image
    t = np.zeros((300, 300))
    for i in range(0, 100):
        for j in range(0, 784):
            imrow = i // 10
            imcol = i % 10
            pixrow = j // 28
            pixcol = j % 28
            t[imrow * 30 + pixrow][imcol * 30 + pixcol] = outa[i][j]
    newimage = Image.new('L', (300, 300))  # type, size

    newimage.putdata(t.reshape((90000,)))
    newimage.save("samples.png")


def _saveaudio(outa):
    from PIL import Image
    t = np.zeros((340, 340))
    for i in range(0, 9):
        for j in range(0, 112 * 112):
            imrow = i // 3
            imcol = i % 3
            pixrow = j // 112
            pixcol = j % 112
            t[imrow * 114 + pixrow][imcol * 114 + pixcol] = outa[i][j]
    newimage = Image.new('L', (340, 340))  # type, size

    newimage.putdata(t.reshape((340 * 340,)))
    newimage.save("samples2.png")


def train_encoders(nepochs=30, lr=0.1, weight_decay=0.0001):
    traindata, validdata, testdata = get_dataloader('./data/avmnist', batch_size=64, num_workers=2)
    encoders = [LeNet(1, 6, 3), LeNet(1, 6, 5)]
    fusion = Concat()
    head = MLP(240, 100, 10)
    train(encoders, fusion, head, traindata, validdata, nepochs, optimtype=torch.optim.SGD, lr=lr, weight_decay=weight_decay)

    traindata, validdata, testdata = get_dataloader('./data/avmnist', batch_size=64, num_workers=2)
    model = torch.load('best.pt', weights_only=False)
    single_test(model, testdata)


def learn_features_from_encoder(x_data, y_data, batch_size=1000, encoder_path=None, device=torch.device('cpu')):
    model = torch.load(encoder_path, weights_only=False).to(device)

    dataset = TensorDataset(
        torch.tensor(x_data, dtype=torch.float, device=device),
        torch.tensor(y_data, dtype=torch.float, device=device),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ## Extract features
    x_features = []
    y_features = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            x_feature = model.encoders[0](x_batch)
            y_feature = model.encoders[1](y_batch)
            x_features.append(x_feature.cpu().numpy())
            y_features.append(y_feature.cpu().numpy())

    x_features = np.concatenate(x_features, axis=0)
    y_features = np.concatenate(y_features, axis=0)

    return x_features, y_features


class FeatureExtractor(nn.Module):
    def __init__(self, encoder_list):
        super(FeatureExtractor, self).__init__()
        self.x_encoder = encoder_list[0]
        self.y_encoder = encoder_list[1]

    def forward(self, m, x, y):
        x = self.x_encoder(x)
        y = self.y_encoder(y)
        return m, x, y


if __name__ == '__main__':
    ## train encoders if necessary
    # train_encoders(nepochs=30, lr=0.1, weight_decay=0.0001)

    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    x_data, y_data, m_data = prepare_data('./data/avmnist')

    x_features, y_features = learn_features_from_encoder(x_data, y_data, batch_size=1000, encoder_path='./best.pt', device=device)
    scale = 1/11.0
    eps = np.random.rand(m_data.shape[0], 1) * scale
    m_features = m_data.reshape(-1, 1)

    encoders = [LeNet(1, 6, 3), LeNet(1, 6, 5)]
    feature_extractor = FeatureExtractor(encoders)

    ret = flow_pid(m_features, x_features, y_features,
                   n_flows=3, n_epochs=10, batch_size=1000, lr=1e-4,
                   encoder=None, verbose=True, device=device)
    norm = ret[7] + ret[5] + ret[6] + ret[8]
    r, ux, uy, si = ret[7] / norm, ret[5] / norm, ret[6] / norm, ret[8] / norm
    print(f"flow pid, unnorm I_mxy, R: {ret[7]}, UX: {ret[5]}, UY: {ret[6]}, S: {ret[8]}")
    print(f"flow pid, normalized I_mxy, R: {r}, UX: {ux}, UY: {uy}, S: {si}")