import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from pid.optimizers import exact_gauss_thin_pid

from pid.models.fusions import Concat, Sequential2
from pid.models.unimodels import LeNet, MLP, LeNetEncoder, DeLeNet
from pid.optimizers.supervised_learning import train, single_test
from pid.optimizers.flow_pid import flow_pid
from pid.optimizers.tilde_pid import exact_gauss_tilde_pid



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


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class HumorDataset(Dataset):

    def __init__(self, id_list, max_context_len=5, max_sen_len=20):
        self.id_list = id_list
        openface_file = "./data/humor/openface_features_sdk.pkl"
        covarep_file = "./data/humor/covarep_features_sdk.pkl"
        language_file = "./data/humor/language_sdk.pkl"
        word_embedding_list_file = "./data/humor/word_embedding_list.pkl"
        humor_label_file = "./data/humor/humor_label_sdk.pkl"

        self.word_aligned_openface_sdk = load_pickle(openface_file)
        self.word_aligned_covarep_sdk = load_pickle(covarep_file)
        self.language_sdk = load_pickle(language_file)
        self.word_embedding_list_sdk = load_pickle(word_embedding_list_file)
        self.humor_label_sdk = load_pickle(humor_label_file)
        self.of_d = 371
        self.cvp_d = 81
        self.glove_d = 300
        self.total_dim = self.glove_d + self.of_d + self.cvp_d
        self.max_context_len = max_context_len
        self.max_sen_len = max_sen_len

    # left padding with zero  vector upto maximum number of words in a sentence * glove embedding dimension
    def paded_word_idx(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        pad_w = np.concatenate((np.zeros(max_sen_len - len(seq)), seq), axis=0)
        pad_w = np.array([self.word_embedding_list_sdk[int(w_id)] for w_id in pad_w])
        return pad_w

    # left padding with zero  vector upto maximum number of words in a sentence * covarep dimension
    def padded_covarep_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((np.zeros((max_sen_len - len(seq), self.cvp_d)), seq), axis=0)

    # left padding with zero  vector upto maximum number of words in a sentence * openface dimension
    def padded_openface_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((np.zeros(((max_sen_len - len(seq)), self.of_d)), seq), axis=0)

    # left padding with zero vectors upto maximum number of sentences in context * maximum num of words in a sentence * 456
    def padded_context_features(self, context_w, context_of, context_cvp, max_context_len=5, max_sen_len=20):
        context_w = context_w[-max_context_len:]
        context_of = context_of[-max_context_len:]
        context_cvp = context_cvp[-max_context_len:]

        padded_context = []
        for i in range(len(context_w)):
            p_seq_w = self.paded_word_idx(context_w[i], max_sen_len)
            p_seq_cvp = self.padded_covarep_features(context_cvp[i], max_sen_len)
            p_seq_of = self.padded_openface_features(context_of[i], max_sen_len)
            padded_context.append(np.concatenate((p_seq_w, p_seq_cvp, p_seq_of), axis=1))

        pad_c_len = max_context_len - len(padded_context)
        padded_context = np.array(padded_context)

        # if there is no context
        if not padded_context.any():
            return np.zeros((max_context_len, max_sen_len, self.total_dim))

        return np.concatenate((np.zeros((pad_c_len, max_sen_len, self.total_dim)), padded_context), axis=0)

    def padded_punchline_features(self, punchline_w, punchline_of, punchline_cvp, max_sen_len=20, left_pad=1):

        p_seq_w = self.paded_word_idx(punchline_w, max_sen_len)
        p_seq_cvp = self.padded_covarep_features(punchline_cvp, max_sen_len)
        p_seq_of = self.padded_openface_features(punchline_of, max_sen_len)
        return np.concatenate((p_seq_w, p_seq_cvp, p_seq_of), axis=1)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        hid = self.id_list[index]
        punchline_w = np.array(self.language_sdk[hid]['punchline_embedding_indexes'])
        punchline_of = np.array(self.word_aligned_openface_sdk[hid]['punchline_features'])
        punchline_cvp = np.array(self.word_aligned_covarep_sdk[hid]['punchline_features'])

        context_w = np.array(self.language_sdk[hid]['context_embedding_indexes'])
        context_of = np.array(self.word_aligned_openface_sdk[hid]['context_features'])
        context_cvp = np.array(self.word_aligned_covarep_sdk[hid]['context_features'])

        # punchline feature
        x_p = torch.LongTensor(
            self.padded_punchline_features(punchline_w, punchline_of, punchline_cvp, self.max_sen_len))
        # context feature
        x_c = torch.LongTensor(
            self.padded_context_features(context_w, context_of, context_cvp, self.max_context_len, self.max_sen_len))

        y = torch.FloatTensor([self.humor_label_sdk[hid]])

        return x_p, x_c, y


if __name__ == '__main__':
    # assume features are already extracted
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    # x_data, y_data, m_data = prepare_data('./pretrained/humor/humor_features_mfm_64.npy', modalities = [1,2])

    # scale = 1/3.0
    # eps = np.random.rand(m_data.shape[0], 1) * scale
    # m_data = m_data.reshape(-1, 1) + eps
    # print(x_data.shape, y_data.shape, m_data.shape)
    #
    # mxy = np.vstack((m_data.T, x_data.T, y_data.T))
    # cov = np.corrcoef(mxy)    # Shape: (dm+dx+dy, dm+dx+dy)
    # print(f'cov.shape: {cov.shape} \n')
    #
    # ret = exact_gauss_tilde_pid(cov, m_data.shape[1], x_data.shape[1], y_data.shape[1])
    # print(f'tilde pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')
    #
    #
    # ret = flow_pid(m_data, x_data, y_data, n_flows=10, n_epochs=250, batch_size=1000, lr=2e-4, verbose=True, device=device)
    # norm = ret[7] + ret[5] + ret[6] + ret[8]
    # print(ret[7], ret[5], ret[6], ret[8])
    # r, ux, uy, si = ret[7] / norm, ret[5] / norm, ret[6] / norm, ret[8] / norm
    # print(f"flow pid, normalized I_mxy, R: {r}, UX: {ux}, UY: {uy}, S: {si}")

    data_folds_file = "./data/humor/data_folds.pkl"
    data_folds = load_pickle(data_folds_file)
    train = data_folds['train']
    dev = data_folds['dev']
    test = data_folds['test']

    train_set = HumorDataset(train)
    dev_set = HumorDataset(dev)
    test_set = HumorDataset(test)

    batch = 10000
    train_dataloader = DataLoader(train_set, batch_size=batch, shuffle=True)
    dev_dataloader = DataLoader(dev_set, batch_size=batch, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch, shuffle=True)

    x_p, x_c, y = next(iter(test_dataloader))
    print(x_p.shape)

    # dx, dy = 200, 200  # Feature dimensions
    # n_comps = 64  # Number of components to keep
    # n_comps = min(dx, dy, n_comps)
    #
    # pca_x = PCA(n_components=n_comps)
    # pca_y = PCA(n_components=n_comps)
    # n_comps = min(dx, dy, n_comps)
    # x = pca_x.fit_transform(x)
    # y = pca_y.fit_transform(y)
    #
    # mxy = np.hstack((m, x, y))
    # cov = np.cov(mxy.T)
    # print(f"Covariance matrix shape: {cov.shape}")
    #
    # ret = exact_gauss_thin_pid(cov, 1, n_comps, n_comps)
    # print(f"Dataset: R: {ret[7]}, U1: {ret[5]}, U2: {ret[6]}, S: {ret[8]}")

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

