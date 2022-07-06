import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import tools_ascc

"""
Github: Yonv1943 Zen4 Jia1 hao2
https://github.com/Yonv1943/DL_RL_Zoo/blob/master/RNN

The source of training data 
https://github.com/L1aoXingyu/
code-of-learn-deep-learning-with-pytorch/blob/master/
chapter5_RNN/time-series/lstm-time-series.ipynb
"""

TOTAL_ACTIVITY_CNT = 14

def get_activity_set_info(data_index):
    activity_index_set_for_lstm = set()
    for i in data_index:
        activity_index_set_for_lstm.add(i)

    print("Index info: len:", len(activity_index_set_for_lstm))
    print(activity_index_set_for_lstm)

    print(max(data_index))
    print(min(data_index))

    return 0

def run_train_lstm():
    inp_dim = 3
    out_dim = TOTAL_ACTIVITY_CNT 
    mid_dim = 196
    mid_layers = 1
    batch_size = 65 #65
    mod_dir = '.'

    max_eps = 1000 * 1
    # max_eps = 1000 

    '''load data'''
    # data = load_data()
    data, ori_data = load_data_from_casas_dataset() #  [duration, timestamp, activity_index]
    print(data)
    data_x = data[:-1, :]
    data_y = ori_data[+1:, 2]

    print(data_x.shape)
    print(data_y.shape)

    print('data_y:', data_y)
    

    get_activity_set_info(data_y)

   
    assert data_x.shape[1] == inp_dim

    train_size = int(len(data_x) * 0.75)
    # train_size = 1400

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, 1))

    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    
    # criterion =  # nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)


    print('var_x:', var_x.shape)
    print('var_y:', var_y.shape)
    # print('var_y:', var_y)
    # exit(0)
    
    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = batch_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    # batch_var_x = var_x
    # batch_var_y = var_y

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training Start")
    for e in range(max_eps):
        out = net(batch_var_x)
        print('out:',out.shape)
        print('batch_var_y:', batch_var_y.shape)

        print('out 0:', out)
        print('batch_var_y:',batch_var_y)

        out_v = out.view(-1, out_dim)
        ori_y = batch_var_y.view(-1)
        ori_y = ori_y.long()

        print('out_v:', out_v.shape)
        print('ori_y:', ori_y.shape)

        print('out_v:', out_v[0])
        print('ori_y:', ori_y)

        loss = F.cross_entropy(out_v, ori_y)
        # loss = criterion(out, batch_var_y)
        # loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if e % 200 == 0:
            print('Epoch: {:4}, Loss: {:.8f}'.format(e, loss.item()))
    torch.save(net.state_dict(), '{}/net_activity.pth'.format(mod_dir))
    print("Save in:", '{}/net_activity.pth'.format(mod_dir))

    '''eval'''
    net.load_state_dict(torch.load('{}/net_activity.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()

    test_x = data_x.copy()
    test_x[train_size:, 0] = 0
    test_x = test_x[:, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

    '''simple way but no elegant'''
    # for i in range(train_size, len(data) - 2):
    #     test_y = net(test_x[:i])
    #     test_x[i, 0, 0] = test_y[-1]

    '''elegant way but slightly complicated'''
    eval_size = 1
    zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)
    test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
    test_x[train_size + 1, 0, 0] = test_y[-1]
    for i in range(train_size + 1, len(data) - 2):
        test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)
        test_x[i + 1, 0, 0] = test_y[-1]
    pred_y = test_x[1:, 0, 0]
    pred_y = pred_y.cpu().data.numpy()
    print(pred_y)
    
    print('pred_y')
    print(max(pred_y))
    print(min(pred_y))

    diff_y = pred_y[train_size:] - data_y[train_size:-1]
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

    test_pred_y = pred_y[train_size:]
    test_data_y = data_y[train_size:] 

    plt.plot(test_pred_y, 'r', label='pred')
    plt.plot(test_data_y, 'b', label='real', alpha=0.3)
    # plt.plot(pred_y, 'r', label='pred')
    # plt.plot(data_y, 'b', label='real', alpha=0.3)    
    # plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
    plt.legend(loc='best')
    plt.savefig('plot_figure/lstm_activity_prediction.png')
    plt.pause(4)

class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Dropout(0.01),
            # nn.LogSoftmax(),
            nn.Linear(mid_dim, out_dim),
            # nn.Softmax()
        )  

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


def load_data_from_casas_dataset2():
    res = tools_ascc.get_duration_from_dataset2()
    seq = np.array(res,dtype=np.float32)
    print(len(seq))

        # normalization
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)

    print(seq)

    return seq

def load_data_from_casas_dataset():
    res = tools_ascc.get_duration_from_dataset()
    seq = np.array(res,dtype=np.float32)
    print(len(seq))

    ori_data = seq
    data = seq
    data_x = data[:-1, :]
    data_y = data[+1:, 2]
    print(data_x.shape)
    print(data_y.shape)

    print('data_y:', data_y)
        # normalization
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)

    print(seq)



    return seq, ori_data


if __name__ == '__main__':
    # load_data_from_casas_dataset2()
    run_train_lstm()
    # run_train_gru()
    # run_origin()
