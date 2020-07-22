import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

sys.path.extend(['../'])
from feeders import tools

class Feeder(Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 use_mmap=True,
                 transform1 = None,
                 transform2 = None,
                 transform3 = None,
                 dataset = 'ntu',
                 semi = False,
                 semi_rate = None,
                 augLinEval = False,
                 semi_NTU_scale = 60, 
                 ):
        self.data_path = data_path
        self.label_path = label_path
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.use_mmap = use_mmap
        self.dataset = dataset
        # print(dataset)
        self.load_data()
        self.normalization = False
        self.semi = semi
        self.rate = semi_rate
        self.augLinEval = augLinEval

        if self.semi:
            self.separate_data(semi_NTU_scale)

        if self.normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        # load data
        if self.use_mmap:
            # (40091, 3, 300, 25, 2)
            if self.dataset == 'uwa3d':
                self.data = np.load(self.data_path, mmap_mode='r')
                N, C, T, V = self.data.shape
                self.data = self.data.reshape((N, C, T, V, 1))
            else:
                # self.data = np.load(self.data_path, mmap_mode='r')
                self.data = np.load(self.data_path, mmap_mode='r')[:, :, :, :, 0:2]
        else:
            self.data = np.load(self.data_path)


    def separate_data(self, semi_NTU_scale):
        num_per_class = int(self.rate * len(self.data)/ float(semi_NTU_scale))
        self.semi_data = []
        self.semi_label = []
        if self.dataset == 'ntu':
            ntu_dict = {}
            for i, label_ in enumerate(self.label):
                if label_ in ntu_dict.keys():
                    if ntu_dict[label_] < num_per_class:
                        ntu_dict[label_] = ntu_dict[label_] + 1
                        self.semi_data.append(self.data[i])
                        self.semi_label.append(self.label[i])
                else:
                    ntu_dict[label_]  = 1
                    self.semi_data.append(self.data[i])
                    self.semi_label.append(self.label[i])
        self.semi_data = np.array(self.semi_data)

    # def get_mean_map(self):
    #     data = self.data
    #     N, C, T, V, M = data.shape
    #     self.mean_map_C = data.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0).squeeze()
    #     self.std_map_C = data.transpose((0, 2, 3, 4, 1)).reshape((N * T * V * M, C )).std(axis=0)
    #     self.mean_map_CV = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
    #     self.std_map_CV = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
    #
    #     print("mean")
    #     print(self.mean_map_C)
    #     print('std')
    #     print(self.std_map_C)
    #     # print("====mean cv")
    #     # print(self.mean_map_CV)
    #     # print("====mena std")
    #     # print(self.std_map_CV)
    #     np.save( '/home/lwg/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/CMC/{}_skeleton_mean.npy'.format(self.dataset),self.mean_map_CV)
    #     np.save( '/home/lwg/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/CMC/{}_skeleton_std.npy'.format(self.dataset),self.std_map_CV)


    def __len__(self):
        if self.semi:
            return len(self.semi_label)
        else:
            return len(self.label)

    # def __iter__(self):
    #     return self


    def __getitem__(self, index):
        if self.semi:
            data_numpy = self.semi_data[index]
            label = self.semi_label[index]
        else:
            data_numpy = self.data[index]
            label = self.label[index]

        # if self.normalization:
        #     data_numpy = (data_numpy - self.mean_map) / self.std_map

        # if self.dataset == 'uwa3d':
        #     C, T, V = data_numpy.shape
        #     data_numpy = data_numpy.reshape(C, T, V, 1)

        # print(data_numpy.shape)
        data1 = self.transform1(data_numpy)
        if self.augLinEval:
            data_aug_lin_eval = data_numpy.copy()
            data_aug_lin_eval = self.transform3(data_aug_lin_eval)


        if self.transform2 != None:
            data2 = self.transform2(data_numpy)
            # print(type(data1))
            # print(type(data2))

            data = torch.cat([data1, data2], dim=0)
        else:
            data = data1

        if self.augLinEval:
            return data, label, index, data_aug_lin_eval
        else:
            return data, label, index, 





