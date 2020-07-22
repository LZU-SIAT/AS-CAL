import numpy as np
import  glob
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
sys.path.extend(['../'])

SETS = ['s01s02','s01s03','s01s07','s02s01','s02s03','s02s06','s02s07','s03s02',
        's03s04','s03s05','s03s06','s04s02','s04s03','s04s06','s05s02','s05s03',
        's06s02','s06s03','s06s04','s07s01','s07s03']
FOLDS = [
    [ 1,  9, 15, 19],
    [ 5,  7, 10, 16],
    [ 2,  3, 20, 21],
    [ 4,  6,  8, 11],
    [12, 13, 14, 17, 18]]

ACTIONS = ['Approaching','Departing','Kicking','Punching','Pushing','Hugging',
           'ShakingHands','Exchanging']

# for action_id in range(len(ACTIONS)):


# DATA_DIR = r'/data5/xushihao/data/SBU'

def denormalize(norm_coords):
    """ SBU denormalization
        original_X = 1280 - (normalized_X .* 2560);
        original_Y = 960 - (normalized_Y .* 1920);
        original_Z = normalized_Z .* 10000 ./ 7.8125;
    """
    denorm_coords = np.empty(norm_coords.shape)
    denorm_coords[:, 0] = 1280 - (norm_coords[:, 0] * 2560)
    denorm_coords[:, 1] = 960 - (norm_coords[:, 1] * 1920)
    denorm_coords[:, 2] = norm_coords[:, 1] * 10000 / 7.8125

    return denorm_coords
# 没有进行原来程序的ntu_norm
def parse_sbu_txt(pose_filepath, normalized=False):
    video_poses_mat = np.loadtxt(pose_filepath, delimiter=',', usecols=range(1, 91))

    video_poses = []
    for frame_pose in video_poses_mat:
        people = []
        # 2 persons * 15 joints * 3 dimensions
        people_poses = frame_pose.reshape(2, 45)
        for person in people_poses:
            if normalized:
                per = person.reshape(15, 3)
            else:
                per = denormalize(person.reshape(15, 3))
            # per['confs'] = 15 * [1]
            people.append(per)
        video_poses.append(people)

    return np.array(video_poses)


def get_ground_truth(data_dir ):
    max_frams = 0
    setname_lst, fold_lst, seq_lst, action_lst, path_lst, frames_lst = [], [], [], [], [] ,[]
    for set_id, set_name in enumerate(SETS):
        for action_id in range(len(ACTIONS)):
            search_exp = '{}/{}/{:02}/*'.format(data_dir, set_name, action_id + 1)
            paths = glob.glob(search_exp)
            paths.sort()
            for path in paths:
                seq = path.split('/')[-1]
                fold = np.argwhere([set_id + 1 in lst for lst in FOLDS])[0, 0]
                frames = len(parse_sbu_txt(path + '/skeleton_pos.txt'))
                max_frams = max(max_frams, frames)

                setname_lst.append(set_name)
                fold_lst.append(fold)
                seq_lst.append(seq)
                action_lst.append(action_id)
                path_lst.append(path + '/skeleton_pos.txt')
                # frames_lst.append(frames)

    dataframe_dict = {'set_name': setname_lst,
                      'fold': fold_lst,
                      'seq': seq_lst,
                      'path': path_lst,
                      'action': action_lst,
                      # 'frames': frames_lst
                      }
    ground_truth = pd.DataFrame(dataframe_dict)
    return ground_truth, max_frams




def get_train_gt(fold_num, ground_truth):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: " + str(fold_num))

    # ground_truth, _ = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold != fold_num]

    return gt_split


def get_val_gt(fold_num, ground_truth):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: " + str(fold_num))

    # ground_truth, _ = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold == fold_num]

    return gt_split


class SBU_feeder(Dataset):
    def __init__(self,
                 root_dir,
                 fold,
                 train_or_val,
                 transform1 = None,
                 transform2 = None,
                 dataset = 'ntu',
                 semi = False,
                 semi_rate = None,
                 ):
        self.root_dir = root_dir
        self.fold = fold
        self.train_or_val = train_or_val
        self.transform1 = transform1
        self.transform2 = transform2
        self.dataset = dataset
        self.semi = semi
        self.rate = semi_rate
        if self.semi:
            self.separate_data()

        self.all_gt, self.max_frame = get_ground_truth(self.root_dir)

        if self.train_or_val == 'train':
            self.gt = get_train_gt(self.fold, self.all_gt)
        elif self.train_or_val == 'val':
            self.gt = get_val_gt(self.fold, self.all_gt)
        else:
            raise  ValueError('no such flag')


    # def separate_data(self, ):
    #     num_per_class = int(self.rate * len(self.data)/ 60.0)
    #     self.semi_data = []
    #     self.semi_label = []
    #     if self.dataset == 'ntu':
    #         ntu_dict = {}
    #         for i, label_ in enumerate(self.label):
    #             if label_ in ntu_dict.keys():
    #                 if ntu_dict[label_] < num_per_class:
    #                     ntu_dict[label_] = ntu_dict[label_] + 1
    #                     self.semi_data.append(self.data[i])
    #                     self.semi_label.append(self.label[i])
    #             else:
    #                 ntu_dict[label_]  = 1
    #                 self.semi_data.append(self.data[i])
    #                 self.semi_label.append(self.label[i])
    #     self.semi_data = np.array(self.semi_data)
    def __len__(self):
        if self.semi:
            pass
            # return len(self.semi_label)
        else:
            return len(self.gt)



    def __getitem__(self, index):
        if self.semi:
            pass
            # print('semimimimimimimimimimimimimimimimi')
            # data_numpy = self.semi_data[index]
            # label = self.semi_label[index]
        else:
            raw_data = parse_sbu_txt(self.gt.iloc[index].path)
            T, M, V, C = raw_data.shape
            raw_data = raw_data.transpose([-1,0,-2, 1]) # C, T, V, M

            data_numpy = np.zeros([C, self.max_frame, V, M])
            data_numpy[:, :T , :, :] = raw_data

            label = self.gt.iloc[index].action # 


        data1 = self.transform1(data_numpy)

        if self.transform2 != None:
            data2 = self.transform2(data_numpy)
            data = torch.cat([data1, data2], dim=0)
        else:
            data = data1

        return data, label, index

# a = parse_sbu_txt('/data5/xushihao/data/SBU/s01s03/04/002/skeleton_pos.txt')
# print(a.shape)