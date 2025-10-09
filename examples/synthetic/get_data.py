import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils import generate_randomBC


def generate_random_data(dm, dx, dy, num_data=20000):
    hx, hy = generate_randomBC(dm, dx, dy)

    sigm = np.eye(dm)
    sigx_m = np.eye(dx)
    sigy_m = np.eye(dy)
    sigw = np.zeros((dx, dy))

    # Covariance matrix construction for both unique or redundant
    cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                    [hx @ sigm, hx @ sigm @ hx.T + sigx_m, hx @ sigm @ hy.T + sigw],
                    [hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + sigy_m]])

    # Generate data
    data = np.random.multivariate_normal(mean=np.zeros(dm + dx + dy), cov=cov, size=num_data)
    m_data = data[:, :dm]
    x_data = data[:, dm:dm + dx]
    y_data = data[:, dm + dx:dm + dx + dy]
    x_data = x_data ** 3
    y_data = np.cbrt(y_data)
    m_data = np.cbrt(m_data)

    # Split data into training and testing sets
    train_size = int(0.9 * num_data)
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    m_train = m_data[:train_size]
    x_test = x_data[train_size:]
    y_test = y_data[train_size:]
    m_test = m_data[train_size:]

    # Save to files
    np.save('data/syn_data/x_train.npy', x_train)
    np.save('data/syn_data/y_train.npy', y_train)
    np.save('data/syn_data/m_train.npy', m_train)
    np.save('data/syn_data/x_test.npy', x_test)
    np.save('data/syn_data/y_test.npy', y_test)
    np.save('data/syn_data/m_test.npy', m_test)

    print("Data generated and saved to files.")
    print(f"Training data shapes: x: {x_train.shape}, y: {y_train.shape}, m: {m_train.shape}")
    print(f"Testing data shapes: x: {x_test.shape}, y: {y_test.shape}, m: {m_test.shape}")


def gpid_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True):

    trains = [np.load(data_dir + "/syn_data/x_train.npy"), np.load(data_dir +
                                                                   "/syn_data/y_train.npy"),
              np.load(data_dir + "/syn_data/m_train.npy")]
    tests = [np.load(data_dir + "/syn_data/x_test.npy"), np.load(data_dir +
                                                                 "/syn_data/y_test.npy"),
             np.load(data_dir + "/syn_data/m_test.npy")]

    # get data size
    num_train = len(trains[0])
    num_test = len(tests[0])
    num_valid = int(0.9 * num_train)

    trainlist = [[trains[j][i] for j in range(3)] for i in range(num_train)]
    testlist = [[tests[j][i] for j in range(3)] for i in range(num_test)]
    valids = DataLoader(trainlist[num_valid:num_train], shuffle=False,
                        num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    tests = DataLoader(testlist, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    trains = DataLoader(trainlist[0:num_valid], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    return trains, valids, tests


def get_dataloader(path, keys=['a', 'b', 'label'], modalities=[0, 1], batch_size=32, num_workers=4, test_path=None):
    if type(path) == list:
        data_list = []
        for dat in path:
            try:
                with open(dat, "rb") as f:
                    data_list.append(pickle.load(f))
                    f.close()
            except Exception as ex:
                print("Error during unpickling object", ex)
                exit()
        data = dict()
        data['train'] = dict()
        for f in data_list[0]['train']:
            for (i, dat) in enumerate(data_list):
                d = data['train'].get(f, [])
                if i == 0:
                    d.append(dat['train'][f])
                else:
                    d.append(dat['train'][f][:int(0.1 * len(dat['train'][f]))])
                data['train'][f] = d
            data['train'][f] = np.concatenate(data['train'][f])
    else:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as ex:
            print("Error during unpickling object", ex)
            exit()
    if test_path:
        try:
            with open(test_path, "rb") as f:
                test_data = pickle.load(f)
        except Exception as ex:
            print("Error during unpickling object", ex)
            exit()
        data['valid1'] = test_data['valid1']
        data['valid2'] = test_data['valid2']
        data['test'] = test_data['test']
    label = keys[-1]

    traindata = DataLoader(SyntheticDataset(data['train'], keys, modalities=modalities),
                           shuffle=True,
                           num_workers=num_workers,
                           batch_size=batch_size,
                           collate_fn=process_input)
    print("Train data: {}".format(data['train'][label].shape[0]))
    if 'valid' in data:
        validdata1 = DataLoader(SyntheticDataset(data['valid'], keys, modalities=modalities),
                                shuffle=False,
                                num_workers=num_workers,
                                batch_size=batch_size,
                                collate_fn=process_input)
        print("Valid data: {}".format(data['valid'][label].shape[0]))
        validdata2 = None
    else:
        validdata1 = DataLoader(SyntheticDataset(data['valid1'], keys, modalities=modalities),
                                shuffle=False,
                                num_workers=num_workers,
                                batch_size=batch_size,
                                collate_fn=process_input)
        validdata2 = DataLoader(SyntheticDataset(data['valid2'], keys, modalities=modalities),
                                shuffle=False,
                                num_workers=num_workers,
                                batch_size=batch_size,
                                collate_fn=process_input)
        print("Valid data 1: {}".format(data['valid1'][label].shape[0]))
        print("Valid data 2: {}".format(data['valid2'][label].shape[0]))
    testdata = DataLoader(SyntheticDataset(data['test'], keys, modalities=modalities),
                          shuffle=False,
                          num_workers=num_workers,
                          batch_size=batch_size,
                          collate_fn=process_input)
    print("Test data: {}".format(data['test'][label].shape[0]))
    # print data shape
    print("Train data shape: {}".format(data['train'][keys[0]].shape))

    return traindata, validdata1, validdata2, testdata


class SyntheticDataset(Dataset):
    def __init__(self, data, keys, modalities):
        self.data = data
        self.keys = keys
        self.modalities = modalities

    def __len__(self):
        return len(self.data[self.keys[-1]])

    def __getitem__(self, index):
        tmp = []
        for i, modality in enumerate(self.modalities):
            if self.keys[i] not in self.data.keys():
                raise NotImplementedError
            else:
                tmp.append(torch.tensor(self.data[self.keys[i]][index]))
        tmp.append(torch.tensor(self.data[self.keys[-1]][index]))
        return tmp


def process_input(inputs):
    processed_input = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input.append(torch.stack(feature))

    for sample in inputs:
        labels.append(sample[-1])
    processed_input.append(torch.tensor(labels).view(len(inputs), ))

    return processed_input


if __name__ == "__main__":
    dm = 1  # Number of dimensions for M
    dx = 100  # Number of dimensions for X
    dy = 100  # Number of dimensions for Y
    num_data = 20000  # Number of data points to generate

    generate_random_data(dm, dx, dy, num_data)

    train, valid, test = gpid_dataloader('./data', batch_size=64, num_workers=4)
    for x, y, m in train:
        print(f"Batch shapes - x: {x.shape}, y: {y.shape}, m: {m.shape}")
        break  # Just to check the first batch