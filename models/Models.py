from __future__ import print_function

import torch.nn as nn
from torch.nn import LSTM, RNN, GRU
from feeders.tools import aug_look, NormalizeC, NormalizeCV, ToTensor, Skeleton2Image, Image2skeleton
from torchvision import transforms, datasets
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)



class LinearClassifier(nn.Module):
    def __init__(self,  last_layer_dim = None, n_label=None,  ):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Linear(last_layer_dim, n_label)
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)


def reshap_input_for_lstm(x1):
    n, c, t, v, m = x1.size()
    x1 = x1.permute(2, 0, 1, 3, 4).contiguous()  # t, n, c, v, m
    x1 = x1.view(t, n, -1)
    return x1


class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, flag ):
        super(Head, self).__init__()
        self.flag = flag
        if flag == 'linear':
            self.model = nn.Sequential(nn.Linear(input_dim, output_dim, ))
        elif flag == 'nonlinear':
            self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, output_dim, )
                                       )
        else:
            raise NotImplementedError("not option")

        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # if self.flag == 'nonlinear':
                #     m.bias.data.fill_(0.0)


    def forward(self, x):
        return self.model(x)




class Bi_lstm_with_head(nn.Module):
    def __init__(self, input_size, args):
        super(Bi_lstm_with_head, self).__init__()
        self.bi_lstm = LSTM(input_size=input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer, bidirectional= ('bi' in args.model))
        if 'bi' in args.model:
            bi_num = 2
        else:
            bi_num = 1
        self.head = Head(args.hidden_units * bi_num, args.hidden_units * bi_num, args.head_dim, args.head_flag)
        self.args = args
    def forward(self, x):
        x = reshap_input_for_lstm(x)
        x, _  = self.bi_lstm(x)
        x = x.permute(1, 0, 2).contiguous()  # n,t,h
        if self.args.tap == 'mean':
            x = x.mean(dim=1)
        elif self.args.tap == 'last':
            x = x[:, -1, :]
        elif self.args.tap == 'sum':
            x = x.sum(dim=1)
        else:
            raise  NotImplementedError('wrong lstm type')
        x = self.head(x)
        return  x





class RNN_model(nn.Module):
    def __init__(self, input_size, args):
        super(RNN_model, self).__init__()
        self.RNN = RNN(input_size = input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer)
        self.args = args
    def forward(self, x):
        x = reshap_input_for_lstm(x)
        x, _  = self.RNN(x)
        x = x.permute(1, 0, 2).contiguous()  # n,t,h
        if self.args.tap == 'mean':
            x = x.mean(dim=1)
        elif self.args.tap == 'last':
            x = x[:, -1, :]
        elif self.args.tap == 'sum':
            x = x.sum(dim=1)
        else:
            raise  NotImplementedError('wrong rnn type')
        return  x


class RNN_model_linear(nn.Module):
    def __init__(self, input_size, args):
        super(RNN_model_linear, self).__init__()
        self.RNN = RNN(input_size = input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer)
        if 'bi' in args.model:
            bi_num = 2
        else:
            bi_num = 1
        self.classifier = nn.Linear(args.hidden_units * bi_num, args.n_label)
        self.args = args
    def forward(self, x):
        x = reshap_input_for_lstm(x)
        x, _  = self.RNN(x)
        x = x.permute(1, 0, 2).contiguous()  # n,t,h
        if self.args.tap == 'mean':
            x = x.mean(dim=1)
        elif self.args.tap == 'last':
            x = x[:, -1, :]
        elif self.args.tap == 'sum':
            x = x.sum(dim=1)
        else:
            raise  NotImplementedError('wrong rnn type')
        x = self.classifier(x)
        return  x

class GRU_model_linear(nn.Module):
    def __init__(self, input_size, args):
        super(GRU_model_linear, self).__init__()
        self.GRU = GRU(input_size = input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer)
        if 'bi' in args.model:
            bi_num = 2
        else:
            bi_num = 1
        self.classifier = nn.Linear(args.hidden_units * bi_num, args.n_label)
        self.args = args
    def forward(self, x):
        x = reshap_input_for_lstm(x)
        x, _  = self.GRU(x)
        x = x.permute(1, 0, 2).contiguous()  # n,t,h
        if self.args.tap == 'mean':
            x = x.mean(dim=1)
        elif self.args.tap == 'last':
            x = x[:, -1, :]
        elif self.args.tap == 'sum':
            x = x.sum(dim=1)
        else:
            raise  NotImplementedError('wrong rnn type')
        x = self.classifier(x)
        return  x

class GRU_model(nn.Module):
    def __init__(self, input_size, args):
        super(GRU_model, self).__init__()
        self.GRU = GRU(input_size = input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer)
        self.args = args
    def forward(self, x):
        x = reshap_input_for_lstm(x)
        x, _  = self.GRU(x)
        x = x.permute(1, 0, 2).contiguous()  # n,t,h
        if self.args.tap == 'mean':
            x = x.mean(dim=1)
        elif self.args.tap == 'last':
            x = x[:, -1, :]
        elif self.args.tap == 'sum':
            x = x.sum(dim=1)
        else:
            raise  NotImplementedError('wrong rnn type')
        return  x

class Bi_lstm(nn.Module):
    def __init__(self, input_size, args):
        super(Bi_lstm, self).__init__()
        self.bi_lstm = LSTM(input_size=input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer, bidirectional=('bi' in args.model))
        self.args = args
    def forward(self, x):
        x = reshap_input_for_lstm(x)
        # print("bi-lstm",x.size())
        x, _  = self.bi_lstm(x)
        x = x.permute(1, 0, 2).contiguous()  # n,t,h
        if self.args.tap == 'mean':
            x = x.mean(dim=1)
        elif self.args.tap == 'last':
            x = x[:, -1, :]
        elif self.args.tap == 'sum':
            x = x.sum(dim=1)
        else:
            raise  NotImplementedError('wrong lstm type')
        return  x

class Bi_lstm_linear(nn.Module):
    def __init__(self,input_size, args):
        super(Bi_lstm_linear, self).__init__()
        self.bi_lstm = LSTM(input_size=input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer,
                            bidirectional= ('bi' in args.model))
        if 'bi' in args.model:
            bi_num = 2
        else:
            bi_num = 1
        self.classifier = nn.Linear(args.hidden_units * bi_num, args.n_label)
        self.args = args
    def forward(self, x):
        x = reshap_input_for_lstm(x)
        x, _  = self.bi_lstm(x)
        x = x.permute(1, 0, 2).contiguous()  # n,t,h
        if self.args.tap == 'mean':
            x = x.mean(dim=1)
        elif self.args.tap == 'last':
            x = x[:, -1, :]
        elif self.args.tap == 'sum':
            x = x.sum(dim=1)
        else:
            raise  NotImplementedError('wrong lstm type')
        x = self.classifier(x)
        return  x


def aug_transfrom(aug_name, args_list, norm, norm_aug, args):
    aug_name_list = aug_name.split("_")
    transform_aug = [aug_look('selectFrames', args.selected_frames)]

    if aug_name_list[0] != 'None':
        for i, aug in enumerate(aug_name_list):
            transform_aug.append(aug_look(aug, args_list[i * 2], args_list[i * 2 + 1]))

    if norm == 'normalizeC':
        transform_aug.extend([Skeleton2Image(), ToTensor(), norm_aug, Image2skeleton()])
    elif norm == 'normalizeCV':
        transform_aug.extend([ToTensor(), norm_aug])
    else:
        transform_aug.extend([ToTensor(), ])
    transform_aug = transforms.Compose(transform_aug)
    return transform_aug