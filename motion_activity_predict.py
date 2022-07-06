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

https://pytorch.org/docs/stable/generated/torch.argmax.html
"""

TOTAL_ACTIVITY_CNT = 15

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
    print("Run train...")
    inp_dim = 3
    out_dim = TOTAL_ACTIVITY_CNT 
    mid_dim = 196
    mid_layers = 1
    batch_size = 65 #65
    mod_dir = '.'

    max_eps = 1000 * 100
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
    # exit(0)
    
    # print data_y information
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
    criterion = nn.MSELoss()

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
    loss_arr = []
    for e in range(max_eps):
        out = net(batch_var_x)


        #print('out 0:', out)
        #print('batch_var_y:',batch_var_y)

        out_v = out.view(-1, out_dim)
        ori_y = batch_var_y.view(-1)
        ori_y = ori_y.long()

        # print('out_v:', out_v.shape)
        # print('ori_y:', ori_y.shape)

        # print('out_v:', out_v[0])
        # print('ori_y:', ori_y)

        # loss = F.cross_entropy(out_v, ori_y)
        # loss = criterion(out, batch_var_y)

        # print('out:',out.shape)
        # print('batch_var_y:', batch_var_y.shape)
        # print('out_v:', out_v)
        # print('ori_y:', ori_y)

        # pred_y = out_v.argmax(1)
        # print('pred_y:', pred_y)
        # print('ori_y:', ori_y)
        # print('pred_y:', pred_y.shape)
        # print('ori_y:', ori_y.shape)


        # loss = (pred_y - ori_y) ** 2 * weights
        # loss = criterion(pred_y, ori_y)
        loss = F.cross_entropy(out_v, ori_y)

        loss = loss.mean()
        loss_arr.append(loss.item())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if e % 200 == 0:
            print('out:',out.shape)
            print('batch_var_y:', batch_var_y.shape)
            print('out_v:', out_v)
            print('ori_y:', ori_y)
            
            print('Epoch: {:4}, Loss: {:.8f}'.format(e, loss.item()))

        if loss.item() < 0.00000005:
            break

    print('Epoch: {:4}, Loss: {:.8f}'.format(e, loss.item()))
    torch.save(net.state_dict(), '{}/net_activity.pth'.format(mod_dir))
    print("Save in:", '{}/net_activity.pth'.format(mod_dir))

    plt.plot(loss_arr, 'r', label='loss')
    plt.legend(loc='best')
    plt.savefig('plot_figure/lstm_activity_prediction_loss.png')
    plt.pause(4)


def run_test_lstm():
    print("Run test...")
    inp_dim = 3
    out_dim = TOTAL_ACTIVITY_CNT 
    mid_dim = 196
    mid_layers = 1
    batch_size = 65 #65
    mod_dir = '.'

    max_eps = 1 * 10
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

    print("Evaluation Start")

    '''eval'''
    net.load_state_dict(torch.load('{}/net_activity.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()

    test_x = data_x[train_size:]
    test_y = data_y[train_size:]
    test_x = test_x.reshape((len(test_x), inp_dim))
    test_y = test_y.reshape((len(test_x), 1))
    '''test'''
    var_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(test_y, dtype=torch.float32, device=device)

    print('var_x:', var_x.shape)
    print('var_y:', var_y.shape)
    # print('var_y:', var_y)
    # exit(0)
    
    batch_var_x = list()
    batch_var_y = list()

    # for i in range(batch_size):
    #     j = batch_size - i
    #     batch_var_x.append(var_x[j:])
    #     batch_var_y.append(var_y[j:])

    batch_var_x.append(var_x[:])
    batch_var_y.append(var_y[:])

    # batch_var_x = var_x
    # batch_var_y = var_y

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # out = net.predict_forward(batch_var_x)
    out = net.forward(batch_var_x)
    print('out shape:',out.shape)
    print('batch_var_y shape:', batch_var_y.shape)

    #print('out 0:', out)
    #print('batch_var_y:',batch_var_y)

    out_v = out.view(-1, out_dim)
    ori_y = batch_var_y.view(-1)
    test_y = ori_y


    print('out_v:', out_v.shape)
    print('ori_y:', ori_y.shape)
    print('ori_y:', ori_y)

    pred_y = []
    pred_y = out_v.argmax(1)

    print('Pred_y(len):', len(pred_y))
    print('test_y(len):', len(test_y))
        
    print('Pred_y:', pred_y)
    print('test_y:', test_y)

    

    test_pred_y = pred_y.cpu().data.numpy()
    test_data_y = test_y.cpu().data.numpy() 

    diff_y = test_pred_y - test_data_y
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

    acc = 0.0
    for i in range(len(test_pred_y)):
        if test_pred_y[i] == test_data_y[i]:
            acc = acc + 1

    acc = acc*1.0/len(test_pred_y)
    print('acc:', acc)



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

        self.predict = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Dropout(0.01),
            # nn.LogSoftmax(),
            nn.Linear(mid_dim, out_dim),
            nn.Softmax()
        )  

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    def predict_forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.predict(y)
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
    print("Loading data...")
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
    print("Loading data end...")



    return seq, ori_data


if __name__ == '__main__':
    run_train_lstm()

    run_test_lstm()
