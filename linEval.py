"""
evaluating MoCo and Instance Discrimination

InsDis: Unsupervised feature learning via non-parametric instance discrimination
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""

from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import socket
import torch.multiprocessing as mp
import torch.distributed as dist

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from util import adjust_learning_rate, AverageMeter


from models.Models import LinearClassifier, Bi_lstm, Bi_lstm_linear, aug_transfrom, GRU_model, RNN_model,GRU_model_linear, RNN_model_linear
from torch.nn import  LSTM
import  torch.nn as nn
# from models.Pose_embedding import reload_for_ntu
import numpy as np
import  random
from feeders.tools import aug_look, NormalizeC, NormalizeCV, ToTensor, Skeleton2Image, Image2skeleton
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


pose_embedding_size = 64

max_body = 2
joints = 25
dim = 3


global global_seed
global_seed = 1
def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--feeder', default='feeders.skeleton_feeder.Feeder', help='data loader will be used')
    parser.add_argument('--selected_frames', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--epochs', type=int, default=90, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='15,35,60,75', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--nesterov', type=bool, default=True, )
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    '''<<<<<<<<<===================================================='''
    # model definition
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm','GRU','RNN'])

    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--lstm_layer', type=int, default=1)
    # dataset
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'uwa3d', 'ucla','sbu'])
    # augmentation
    parser.add_argument('--FTaug', type=str, default='subtract1')
    parser.add_argument('--Vaug', type=str, default= 'subtract1')
    parser.add_argument('--lin_eval_aug', type=str, default= 'subtract1')

    parser.add_argument('--semiFT', type=str, default= 'subtract1')

    ntu_dataset = 'sub60' #choices=['sub60', 'view60', 'sub120','view120']  when dataset == 'ntu'


    if ntu_dataset == 'sub60':
        parser.add_argument('--ntu_dataset', type=str, default='sub60')
        parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/train_data.npy')
        parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/train_label.pkl')
        parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/val_data.npy')
        parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/val_label.pkl')                       
    
    if ntu_dataset == 'view60':   
        parser.add_argument('--ntu_dataset', type=str, default='view60')               
        parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/train_data.npy')
        parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/train_label.pkl')
        parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/val_data.npy')
        parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/val_label.pkl')

    if ntu_dataset == 'sub120':  
        parser.add_argument('--ntu_dataset', type=str, default='sub120')               
        parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/train_data.npy')
        parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/train_label.pkl')
        parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/val_data.npy')
        parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/val_label.pkl')

    if ntu_dataset == 'view120': 
        parser.add_argument('--ntu_dataset', type=str, default='view120')               
        parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/train_data.npy')
        parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/train_label.pkl')
        parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/val_data.npy')
        parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/val_label.pkl')

    parser.add_argument('--aug_for_lin_eval',  default=False)
    parser.add_argument('--tap',  default='mean', help='avg on time for lstm')
    parser.add_argument('--mode',  default= 'semi', choices=['eval','supervise','semi'])
    parser.add_argument('--semi_rate', type= int, default=0.01, choices=[0.01,0.1,0.5] )
    parser.add_argument('--epochs_semi_ft', type= int, default=50, )
    '''=============================================================>>>>>'''
    # useless
    parser.add_argument('--pose',  default=False, )
    parser.add_argument('--norm', type=str, default='None', choices=['normalizeC', 'normalizeCV', "None"])
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--syncBN', action='store_true', help='enable synchronized BN')

    opt = parser.parse_args()

    if opt.mode == 'eval' or opt.mode == 'semi':
        opt.dataset = opt.model_path.split('/')[6].split('_')[0]
        opt.model_name = opt.model_path.split('/')[-2]
        opt.epoch_pth = opt.model_path.split('/')[-1]
        opt.head = (opt.model_name[opt.model_name.find('head')+4] == 'T')
        if '_' in opt.epoch_pth:
            opt.epoch_pth = opt.epoch_pth.split('.')[0].split('_')[-1]
        else:
            opt.epoch_pth = 'current'
    # set the path according to the environment
    if hostname.startswith('ubuntu'):
        opt.save_path = '/data5/xushihao/data/MoCo/CMC/{}_linear'.format(opt.dataset)
        opt.tb_path = '/data5/xushihao/data/MoCo/CMC/{}_linear_tensorboard'.format(opt.dataset)



    # model or ema
    opt.model_or_model_ema = 'model'


    if opt.dataset == 'uwa3d':
        uwa3d_data_folder = '/data5/xushihao/data/UWA3d_feeder_data'
        # opt.uwa3d_train = ['12', '13', '14', '23', '24', '34', ]
        # opt.uwa3d_test1 = ['3','2','2','1','1','1']
        # opt.uwa3d_test2 = ['4','4','3','4','3','2']
        # opt.fold = 1
        # opt.fold_test = 3

        opt.uwa3d_train = ['12', '13', '14', '23', '24', '34', ]
        opt.uwa3d_test1 = ['3','2','2','1','1','1']
        opt.uwa3d_test2 = ['4','4','3','4','3','2']
        # 0: 3,4
        # 1: 2,4
        # 2: 2,3
        # 3: 1,4
        # 4: 1,3
        # 5: 1,2

        opt.fold =  5
        opt.uwa3d_test_view = 2


        opt.train_data_path = '{}/train{}_test{}{}/training_{}/train_data.npy'.format(uwa3d_data_folder, opt.uwa3d_train[opt.fold], 
        opt.uwa3d_test1[opt.fold],opt.uwa3d_test2[opt.fold], opt.uwa3d_train[opt.fold])
        opt.train_label_path = '{}/train{}_test{}{}/training_{}/train_label.pkl'.format(uwa3d_data_folder, opt.uwa3d_train[opt.fold], 
        opt.uwa3d_test1[opt.fold],opt.uwa3d_test2[opt.fold], opt.uwa3d_train[opt.fold])

        # opt.train_data_path = r'/data5/xushihao/data/UWA3d_feeder_data/v1v2_train/train_data.npy'
        # opt.train_label_path = r'/data5/xushihao/data/UWA3d_feeder_data/v1v2_train/train_label.pkl'
        opt.val_data_path = '{}/train{}_test{}{}/test_{}/val_data.npy'.format(uwa3d_data_folder, opt.uwa3d_train[opt.fold], 
        opt.uwa3d_test1[opt.fold],opt.uwa3d_test2[opt.fold], opt.uwa3d_test_view)
        opt.val_label_path = '{}/train{}_test{}{}/test_{}/val_label.pkl'.format(uwa3d_data_folder, opt.uwa3d_train[opt.fold], 
        opt.uwa3d_test1[opt.fold],opt.uwa3d_test2[opt.fold], opt.uwa3d_test_view)
    elif opt.dataset == 'ucla':
        opt.ucla_test_view = 1
        if opt.ucla_test_view == 1:
            opt.train_data_path = r'/data5/xushihao/multiview_skeleton/view1_for_test/train_data.npy'
            opt.train_label_path = r'/data5/xushihao/multiview_skeleton/view1_for_test/train_label.pkl'
            opt.val_data_path = r'/data5/xushihao/multiview_skeleton/view1_for_test/val_data.npy'
            opt.val_label_path = r'/data5/xushihao/multiview_skeleton/view1_for_test/val_label.pkl'
        if opt.ucla_test_view == 2:
            opt.train_data_path = r'/data5/xushihao/multiview_skeleton/view2_for_test/train_data.npy'
            opt.train_label_path = r'/data5/xushihao/multiview_skeleton/view2_for_test/train_label.pkl'
            opt.val_data_path = r'/data5/xushihao/multiview_skeleton/view2_for_test/val_data.npy'
            opt.val_label_path = r'/data5/xushihao/multiview_skeleton/view2_for_test/val_label.pkl'
        if opt.ucla_test_view == 3:
            opt.train_data_path = r'/data5/xushihao/multiview_skeleton/view3_for_test/train_data.npy'
            opt.train_label_path = r'/data5/xushihao/multiview_skeleton/view3_for_test/train_label.pkl'
            opt.val_data_path = r'/data5/xushihao/multiview_skeleton/view3_for_test/val_data.npy'
            opt.val_label_path = r'/data5/xushihao/multiview_skeleton/view3_for_test/val_label.pkl'
    elif opt.dataset == 'sbu':
        opt.train_data_path = r'/data5/xushihao/data/SBU'
        opt.val_data_path = r'/data5/xushihao/data/SBU'
        opt.feeder = 'feeders.sbu_feeder.SBU_feeder'
        if opt.mode == 'eval':
            opt.fold = int(opt.model_path.split('/')[-2].split('_')[1][0])
        if opt.mode == 'supervise':
            opt.fold = 4



    # =====================================
    if opt.mode == 'eval':
        opt.learning_rate = 1
        opt.epochs = 90
        opt.lr_decay_epochs = '15,35,60,75'
        opt.weight_decay = 0
        # opt.weight_decay = 0.0001

    elif opt.mode == 'supervise' :
        opt.learning_rate = 0.01
        if opt.dataset == 'ntu':
            opt.epochs = 60
            opt.lr_decay_epochs = '30,40'
        else:
            opt.epochs = 200
            opt.lr_decay_epochs = '30,80'
        
        opt.weight_decay = 0
    elif opt.mode == 'semi':
        opt.learning_rate = 0.01
        opt.weight_decay = 0
        if opt.semi_rate == 0.01:
            opt.epochs_semi_ft = 35
            # old
            # opt.lr_decay_epochs = '10,20'
            opt.lr_decay_epochs = '15,30'

        elif opt.semi_rate == 0.1:
            opt.epochs_semi_ft = 15
            # old
            # opt.lr_decay_epochs = '8,16'
            opt.lr_decay_epochs = '8'

        elif opt.semi_rate == 0.5:
            # 以前是15
            opt.epochs_semi_ft = 10
            # old
            # opt.lr_decay_epochs = '8,16'
            opt.lr_decay_epochs = '7'


        
    # kk = global_seed
    
    # if global_seed != global_seed_pretrain:
    #     global_seed = kk
    # raise  ValueError("wrong seed")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    
    if opt.mode == 'eval' or opt.mode == 'semi':
        if opt.model_name[opt.model_name.index('F_') - 3] == '_':
            opt.selected_frames = int(opt.model_name[opt.model_name.index('F_') - 2: opt.model_name.index('F_')])
        else:
            opt.selected_frames = int(opt.model_name[opt.model_name.index('F_') - 3 : opt.model_name.index('F_') ])


    if opt.mode == 'eval' or opt.mode == 'semi':
        opt.lstm_layer = int(opt.model_name[opt.model_name.index('layer') + 5] )
        if 'lstm' in opt.model_name:
            if opt.model_name[ opt.model_name.find('lstm') + 6] =="_":
                opt.hidden_units = int(opt.model_name[opt.model_name.find('lstm') + 4: opt.model_name.find('lstm') + 6])
            else:
                opt.hidden_units = int(opt.model_name[ opt.model_name.find('lstm') + 4 : opt.model_name.find('lstm') + 7])
        if str(opt.tap) != opt.model_name[opt.model_name.index('tap') + 4 : opt.model_name.index('tap') + 8 ]:
            if str(opt.tap) != opt.model_name[opt.model_name.index('tap') + 4: opt.model_name.index('tap') + 9]:
                raise IndentationError('wrong tap match for training and test')
        if str(opt.selected_frames) != opt.model_name[opt.model_name.index('F_') - 3 : opt.model_name.index('F_') ]:
            if str(opt.selected_frames) != opt.model_name[opt.model_name.index('F_') - 2 : opt.model_name.index('F_') ]:
                raise IndentationError('wrong seq match for training and test')
    
        if opt.model_name[opt.model_name.index('aug') + 4 :] == 'None':
            opt.FTaug = 'None'
            opt.Vaug = 'None'
        else:
            print("be careful the mismatch of aug")


    if opt.dataset == 'ucla' or opt.dataset == 'uwa3d':
        norm_name = opt.model_name.split('_')[2]
    else:
        norm_name = opt.norm

    opt.norm = norm_name

    if opt.dataset == 'ntu':
        ntu_120 = ('120' in opt.train_data_path.split('/')[-3])
        if opt.mode == 'eval' or opt.mode == 'semi':
            ntu_120_self_sup = ('120' in opt.model_name.split('_')[1])
            if ntu_120 != ntu_120_self_sup:
                raise IndentationError('wrong dataset match for training and test')

            dataset_flag = opt.train_data_path.split('/')[-2] #xsub or xview
            if opt.model_name.split('_')[0] != dataset_flag:
                raise IndentationError('wrong dataset match for training and test')


    if opt.mode == 'semi':
        mode_flag = "{}{}_{}".format(opt.mode, opt.semi_rate, opt.epochs_semi_ft)
    else:
        # mode_flag = opt.mode + '_augLE_{}_{}'.format(str(opt.aug_for_lin_eval)[0], opt.lin_eval_aug)
        mode_flag = opt.mode

    # if opt.dataset == 'uwa3d':
    #     mode_flag = 'testView{}'.format(opt.uwa3d_test_view)

    if opt.FTaug == opt.Vaug:
        aug_flag = '2aug_{}'.format(opt.FTaug)
    else:
        aug_flag = "FTaug_{}_Vaug_{}".format(opt.FTaug, opt.Vaug)
    if opt.mode == 'supervise':
        if opt.dataset == 'ntu':
            first = opt.train_data_path.split('/')[-2]
            ntu_120 = ('120' in opt.train_data_path.split('/')[-3])
            if ntu_120:
                flag_sup = 120
            else:
                flag_sup = 60
        elif opt.dataset == 'ucla':
            first = opt.train_data_path.split('/')[-2]
            flag_sup = ''
        elif opt.dataset == 'sbu':
            first = opt.train_data_path.split('/')[-1] + '_{}fd'.format(opt.fold)
            flag_sup = ''
        elif opt.dataset == 'uwa3d':
            first = opt.train_data_path.split('/')[-3].split('_')[0] + '_test{}'.format(opt.uwa3d_test_view)
            flag_sup = ''

            
        opt.model_name = '{}{}_{}{}_layer{}_lr{}_bsz_{}_epoch_{}_wd{}_nstr{}_tap{}_{}F_GLC{}_{}_{}'.format(first,
                                                                                                                        flag_sup,
                                                                                                                      opt.model,
                                                                                                                      opt.hidden_units,
                                                                                                                      opt.lstm_layer,
                                                                                                                      opt.learning_rate,
                                                                                                                      opt.batch_size,
                                                                                                                      opt.epochs,
                                                                                                                       opt.weight_decay,
                                                                                                                      opt.nesterov,
                                                                                                                        opt.tap,
                                                                                                                      opt.selected_frames,
                                                                                                                        global_seed,
                                                                                                                      aug_flag,
                                                                                                                      mode_flag )
    else:
        opt.model_name = '{}_wd{}_nstr{}_tap{}_GLC{}_{}_{}_{}_{}'.format(opt.model_name,
                                                                                opt.weight_decay,
                                                                                opt.nesterov,
                                                                                opt.tap,
                                                                                global_seed,
                                                                                aug_flag,
                                                                                opt.epoch_pth,
                                                                                mode_flag,
                                                                                opt.model_or_model_ema)


    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name )
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'ntu':
        if '120' in opt.train_data_path.split('/')[-3]:
            opt.n_label = 120
        else:
            opt.n_label = 60
    elif opt.dataset == 'ucla':
        opt.n_label = 10
    elif opt.dataset == 'uwa3d':
        opt.n_label = 30
    elif opt.dataset == 'sbu':
        opt.n_label = 8
    return opt


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



def main():

    global best_acc1
    best_acc1 = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    # if args.norm == 'None':
    #     pass
    # elif args.dataset == 'ntu':
    #     # skeleton on C
    #     mean_C = [ 0.00159457, -0.02654573,  0.51572406]
    #     std_C = [ 0.16942039 , 0.16726708 , 0.73937953]
    #     normalizeC = NormalizeC(mean=mean_C, std=std_C)
    #     # skeleton on C and V
    #     mean_C_V = np.load('/home/lwg/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/CMC/ntu_skeleton_mean.npy').tolist()
    #     std_C_V = np.load('/home/lwg/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/CMC/ntu_skeleton_std.npy').tolist()
    #     normalizeCV = NormalizeCV(mean=mean_C_V, std=std_C_V)
    # elif args.dataset == 'uwa3d':
    #     mean_C = [ 375.1031189 ,356.76657104 , 282.30130005]
    #     std_C = [ 1101.50268555 , 1130.68347168 , 1192.15039062]
    #     normalizeC = NormalizeC(mean=mean_C, std=std_C)
    #     # skeleton on C and V
    #     mean_C_V = np.load(
    #         '/home/lwg/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/CMC/uwa3d_skeleton_mean.npy').tolist()
    #     std_C_V = np.load(
    #         '/home/lwg/xushihao/projects/my_gcn_lstm/Good_project_from_other_people/CMC/uwa3d_skeleton_std.npy').tolist()
    #     normalizeCV = NormalizeCV(mean=mean_C_V, std=std_C_V)
# ====================================================
# ====================================================
    Feeder = import_class(args.feeder)
    train_sampler = None
# ===================================================
    if args.mode == 'semi':
        args_semi_ft = [1, None, None, None, None, None]
        transform_semi_ft = aug_transfrom(args.semiFT, args_semi_ft, args.norm, None, args)

        ntu_scale_flag = args.train_data_path.split('/')[-3].split('_')[-1][2]

        if ntu_scale_flag == '0':
            ntu_scale_flag = 120
        if ntu_scale_flag == 'a':
            ntu_scale_flag = 60

        print('ntu_scale_flag====',ntu_scale_flag)
        train_dataset_semi_ft = Feeder(args.train_data_path,
                                       args.train_label_path,
                                       transform1 = transform_semi_ft,
                                       dataset = args.dataset, 
                                       semi = (args.mode == 'semi'), 
                                       semi_rate = args.semi_rate,
                                       semi_NTU_scale = ntu_scale_flag, 
                                       )
# ====================================================
    args_train = [1, None, None, None]
    transformFT = aug_transfrom(args.FTaug, args_train, args.norm, None, args)


    args_train_lin_eval = [1, None, None, None, None, None]
    transform_lin_eval = aug_transfrom(args.lin_eval_aug, args_train_lin_eval, args.norm, None, args)


    if args.dataset == 'sbu':
        train_dataset = Feeder(args.train_data_path,
                               args.fold,
                               'train',
                               transformFT,
                               None,
                               args.dataset,
                               False,
                               None
                               )
    else:
        train_dataset = Feeder(args.train_data_path,
                               args.train_label_path,
                               transform1 = transformFT,
                               transform3 = transform_lin_eval,
                               dataset = args.dataset,
                            #    augLinEval = args.aug_for_lin_eval
                               )
# ====================================================
    args_val = [1, None, None, None]
    transformV = aug_transfrom(args.Vaug, args_val, args.norm, None, args)

    if args.dataset == 'sbu':
        val_dataset = Feeder(args.train_data_path,
                               args.fold,
                               'val',
                                transformV,
                               None,
                               args.dataset,
                               False,
                               None
                               )  ##### add aug elements
    else:
        val_dataset = Feeder(args.val_data_path,
                             args.val_label_path,
                             transform1 = transformV,
                             dataset = args.dataset,
                             )



    print("train_dataset",len(train_dataset))
    print("val_dataset",len(val_dataset))

    if args.mode == 'semi':
        print("train_dataset_semi_ft", len(train_dataset_semi_ft))
        train_loader_semi_ft = torch.utils.data.DataLoader(
            dataset=train_dataset_semi_ft,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=init_seed(global_seed))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=init_seed(global_seed))

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=init_seed(global_seed)
        )


    if args.mode == 'semi':
        semi_model_path = semi_finetune(train_loader_semi_ft, args)
        print(semi_model_path)
        args.learning_rate = 1
        # args.epochs = 100
        args.lr_decay_epochs = '15,35,60,75'
        iterations = args.lr_decay_epochs.split(',')
        args.lr_decay_epochs = list([])
        for it in iterations:
            args.lr_decay_epochs.append(int(it))




    if 'lstm' in args.model or 'GRU' in args.model or 'RNN' in args.model :

        if args.dataset == 'ucla':
            input_size = 60  # 1  * 20 * 3
        elif args.dataset == 'uwa3d':
            input_size = 45 # 1 * 15 * 3
        elif args.dataset == 'sbu':
            input_size = 90 # 2 * 15 * 3
        else:
            input_size = max_body * joints * dim

        if args.mode == 'supervise' or args.mode == 'finetune' or args.mode == 'finetuneR':
            print('supervise success')
            if args.model == 'GRU':
                model = GRU_model_linear(input_size, args)
            if args.model == 'RNN':
                model = RNN_model_linear(input_size, args)
            if args.model == 'lstm':
                model = Bi_lstm_linear(input_size, args)
            classifier = None
        elif args.mode == 'eval' or args.mode == 'semi':
            if args.model == 'GRU':
                model = GRU_model(input_size = input_size, args = args)
                classifier = LinearClassifier(args.hidden_units , args.n_label)
            elif args.model == 'RNN':
                model = RNN_model(input_size = input_size, args = args)
                classifier = LinearClassifier(args.hidden_units , args.n_label)
            else:
                model = Bi_lstm(input_size, args)
                if 'bi' in args.model:
                    bi_num = 2
                else:
                    bi_num = 1
                classifier = LinearClassifier(args.hidden_units * bi_num, args.n_label)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    print('==> loading pre-trained model')
    if args.mode == 'eval'  or args.mode== 'semi' or args.mode == 'finetune':
        # eval, semi: Bi_lstm: self.bi_lstm
        # finetune: Bi_lstm_linear: self.bi_lstm, self.classifier

        # self-sup: Bi_lstm_head: self.bi_lstm, self.head
        # self-sup: Bi_lstm: self.bi_lstm
        # semi-ft :         : bi_lstm, classifier
        if args.mode == 'semi':
            # bi_lstm, classifier
            print('semi file: ', semi_model_path)
            ckpt = torch.load(semi_model_path)
        else:
            # bi_lstm
            ckpt = torch.load(args.model_path)

        if args.model_or_model_ema == 'model':
            self_supv_state = ckpt['model']
        if args.model_or_model_ema == 'ema':
            self_supv_state = ckpt['model_ema']
        model_dict = model.state_dict()
        # =======
        # old
        # new_self_supv_state = {}
        # for k,v in self_supv_state.items():
        #     new_self_supv_state['bi_lstm.{}'.format(k)] = v
        #
        # state_dict = {k:v for k,v in new_self_supv_state.items() if k in model_dict.keys()}
        # =======
        state_dict = {k:v for k,v in self_supv_state.items() if k in model_dict.keys()}
        if len(state_dict.keys()) == 0:
            raise ImportError('load failure')
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        if args.mode == 'semi':
            classifier_dict = classifier.state_dict()
            classifier_state_dict = {k: v for k, v in self_supv_state.items() if k in classifier_dict.keys()}
            if len(classifier_state_dict.keys()) == 0:
                raise ImportError('load failure')
            classifier_dict.update(classifier_state_dict)
            classifier.load_state_dict(classifier_dict)
        print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
        print('==> done')


    if  args.mode == 'eval' or args.mode == 'semi':
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
        if not args.adam:
            optimizer = torch.optim.SGD(classifier.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        nesterov=args.nesterov,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(classifier.parameters(),
                                         lr=args.learning_rate,
                                         betas=(args.beta1, args.beta2),
                                         weight_decay=args.weight_decay,
                                         eps=1e-8)
    else:
        model = model.cuda(args.gpu)
        if not args.adam:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        nesterov= args.nesterov,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.learning_rate,
                                         betas=(args.beta1, args.beta2),
                                         weight_decay=args.weight_decay,
                                         eps=1e-8)


    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)




    if args.mode == 'eval' or args.mode == 'semi':
        model.eval()
    else:
        model.train()

    cudnn.benchmark = True


    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.cuda(args.gpu)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if 'opt' in checkpoint.keys():
                # resume optimization hyper-parameters
                print('=> resume hyper parameters')
                if 'bn' in vars(checkpoint['opt']):
                    print('using bn: ', checkpoint['opt'].bn)
                if 'adam' in vars(checkpoint['opt']):
                    print('using adam: ', checkpoint['opt'].adam)
                if 'cosine' in vars(checkpoint['opt']):
                    print('using cosine: ', checkpoint['opt'].cosine)
                args.learning_rate = checkpoint['opt'].learning_rate
                # args.lr_decay_epochs = checkpoint['opt'].lr_decay_epochs
                args.lr_decay_rate = checkpoint['opt'].lr_decay_rate
                args.momentum = checkpoint['opt'].momentum
                args.weight_decay = checkpoint['opt'].weight_decay
                args.beta1 = checkpoint['opt'].beta1
                args.beta2 = checkpoint['opt'].beta2
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set cosine annealing scheduler
    if args.cosine:

        # last_epoch = args.start_epoch - 2
        # eta_min = args.learning_rate * (args.lr_decay_rate ** 3) * 0.1
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, last_epoch)

        eta_min = args.learning_rate * (args.lr_decay_rate ** 3) * 0.1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)
        # dummy loop to catch up with current epoch
        for i in range(1, args.start_epoch):
            scheduler.step()

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine

    train_loss_list = []
    val_loss_list = []
    iteration  = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        iteration.append(epoch)
        if args.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        train_acc, train_acc5, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, args,)
        train_loss_list.append(train_loss)

        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_acc5', train_acc5, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


        print("==> testing...")
        test_acc, test_acc5, test_loss = validate(val_loader, model, classifier, criterion, args,
                                                  )
        val_loss_list.append(test_loss)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc5', test_acc5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc1:
            # my_acc = accuracy_score(np.array(target_list), np.array(predict_list))
            # print("my ACC: ", my_acc * 100)
            # print(my_acc * 100 == test_acc)
            best_acc1 = test_acc
            if args.mode == 'eval' or args.mode == 'semi':
                state = {
                    'opt': args,
                    'epoch': epoch,
                    'classifier': classifier.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }
            else:
                state = {
                    'opt': args,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }
            save_name = '{}_epoch_{}_bestAcc_top1_{:.1f}_top5_{:.1f}.pth'.format(args.model, epoch, float(best_acc1.cpu().numpy()), float(test_acc5.cpu().numpy()))
            save_name = os.path.join(args.save_folder, save_name)
            print('saving best model!')
            torch.save(state, save_name)

        
        # save model
        # if epoch % args.save_freq == 0:
        #     print('==> Saving...')
        #     state = {
        #         'opt': args,
        #         'epoch': epoch,
        #         'classifier': classifier.state_dict(),
        #         'best_acc1': test_acc,
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
        #     save_name = os.path.join(args.save_folder, save_name)
        #     print('saving regular model!')
            # torch.save(state, save_name)
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(iteration, train_loss_list, label='train_loss')
    plt.plot(iteration, val_loss_list, label='val_loss')
    plt.legend()
    plt.draw()
    plt.tight_layout()
    save_pdf_path_train = os.path.join(args.save_folder, "loss.pdf")
    plt.savefig(save_pdf_path_train, format='pdf', transparent=True, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.close()

    print(args.save_folder)
    print('\n')
    train_loss_file = os.path.join(args.save_folder, 'train_loss.npy')
    val_loss_file = os.path.join(args.save_folder, 'val_loss.npy')
    print('train_loss \n')
    print(train_loss_list)
    print('val_loss \n')
    print(val_loss_list)
    np.save(train_loss_file, np.array(train_loss_list))
    np.save(val_loss_file, np.array(val_loss_list))



def set_lr(optimizer, lr):
    """
    set the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def pose_embed_(x, t, n, pose_model):
    # x : t, n, c
    flag = False
    if len(x.size()) == 2:
        flag = True
        x = x.unsqueeze(1)
        x = x.repeat(1, 2, 1)
    x_new = torch.zeros(t, n, pose_embedding_size)
    with torch.no_grad():
        # Pose_En, _ = reload_for_ntu(first=flag0, flag=flag1, flag2=flag2, flag3=flag3)
        # Pose_En = Pose_En.cuda()
        for i in range(t):
            temp = pose_model(x[i, :, :])  # n, c
            if flag:
                x_new[i, :, :] = temp[0, :].unsqueeze(0)
            else:
                x_new[i, :, :] = temp  #

        if flag:
            x_new = x_new.squeeze(1)
    del x
    return x_new


def semi_finetune( train_loader,  args,):
    if 'lstm' in args.model:
        if args.dataset == 'ucla':
            input_size = 60
        elif args.dataset == 'uwa3d':
            input_size = 45
        elif args.dataset == 'sbu':
            input_size = 90
        else:
            input_size = max_body * joints * dim

        model = Bi_lstm_linear(input_size, args)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    model.train()

    print('==> loading pre-trained model for semi finetune')
    # semi finetune: Bi_lstm_linear: self.bi_lstm, self.classifier

    # self-sup: Bi_lstm_head: self.bi_lstm, self.head
    # self-sup: Bi_lstm: self.bi_lstm
    ckpt = torch.load(args.model_path)
    self_supv_state = ckpt['model']
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in self_supv_state.items() if k in model_dict.keys()}
    # print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    print("==> loaded checkpoint '{}' (epoch {})".format(args.save_folder, ckpt['epoch']))
    print('==> done')

    model = model.cuda(args.gpu)
    if not args.adam:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay,
                                     eps=1e-8)


    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    model.train()

    cudnn.benchmark = True

    args.start_epoch_semi_ft = 1
    iteration = []
    loss_list = []
    best_acc = 0

    for epoch in range(args.start_epoch_semi_ft, args.epochs_semi_ft + 1):
        iteration.append(epoch)
        if args.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        train_acc, train_acc5, train_loss = semi_ft_train(epoch, train_loader, model,  criterion, optimizer, args,)
        loss_list.append(train_loss)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if train_acc > best_acc:
            best_acc = train_acc
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),

            }
            save_name = 'semi_ft_{}_epoch_{}_bestAcc_{:.1f}.pth'.format(args.model, epoch, float(best_acc.cpu().numpy()))
            save_name = os.path.join(args.save_folder, save_name)
            print('saving best model!')
            torch.save(state, save_name)
            # help release GPU memory
            del state
        

        state = {
            'opt': args,
            'epoch': epoch,
            'model': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(args.save_folder, 'semi_ft_current.pth')
        torch.save(state, save_file)
        del state
        torch.cuda.empty_cache()

    plt.switch_backend('agg')
    plt.figure()
    plt.plot(iteration, loss_list, label='loss')
    plt.draw()
    plt.tight_layout()
    save_pdf_path = os.path.join(args.save_folder, "semi_ft_loss.pdf")
    plt.savefig(save_pdf_path, format='pdf', transparent=True, dpi=300, pad_inches=0,
                bbox_inches='tight')
    plt.close()
    return save_name

def semi_ft_train(epoch, train_loader, model, criterion, optimizer, opt,):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # if opt.gpu is not None:
        #     input = input.cuda(opt.gpu, non_blocking=True)
        input = input.cuda(opt.gpu)
        input = input.float()
        # target = target.cuda(opt.gpu, non_blocking=True)
        target = target.cuda( opt.gpu,non_blocking=True)
        # ===================forward=====================
        output= model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg



def train(epoch, train_loader, model, classifier, criterion, optimizer, opt,):
    """
    one epoch training
    """
    if opt.mode == 'eval' or opt.mode == 'semi':
        model.eval()
        classifier.train()
    else:
        model.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target, _, ) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)
        # if opt.gpu is not None:
        #     input = input.cuda(opt.gpu, non_blocking=True)
        input = input.cuda(opt.gpu)
        input = input.float()
        # target = target.cuda(opt.gpu, non_blocking=True)
        target = target.cuda( opt.gpu,non_blocking=True)
        # ===================forward=====================
        if opt.mode == 'eval' or opt.mode == 'semi':
            with torch.no_grad():
                feat = model(input)
                feat = feat.detach()
            output = classifier(feat)
        else:
            output= model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
        
        # =============
        # if opt.aug_for_lin_eval:
        #     data_time.update(time.time() - end)
        #     # if opt.gpu is not None:
        #     #     input = input.cuda(opt.gpu, non_blocking=True)
        #     input_aug_lin_eval = input_aug_lin_eval.cuda(opt.gpu)
        #     input_aug_lin_eval = input_aug_lin_eval.float()
        #     # target = target.cuda(opt.gpu, non_blocking=True)
        #     # target = target.cuda( opt.gpu,non_blocking=True)
        #     # ===================forward=====================
        #     if opt.mode == 'eval' or opt.mode == 'semi':
        #         with torch.no_grad():
        #             feat_aug = model(input_aug_lin_eval)
        #             feat_aug = feat_aug.detach()
        #         output_aug = classifier(feat_aug)
        #     else:
        #         output_aug = model(input_aug_lin_eval)
        #     loss = criterion(output_aug, target)

        #     acc1, acc5 = accuracy(output_aug, target, topk=(1, 5))
        #     losses.update(loss.item(), input_aug_lin_eval.size(0))
        #     top1.update(acc1[0], input_aug_lin_eval.size(0))
        #     top5.update(acc5[0], input_aug_lin_eval.size(0))

        #     # ===================backward=====================
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     # ===================meters=====================
        #     batch_time.update(time.time() - end)
        #     end = time.time()

        #     # print info
        #     if idx % opt.print_freq == 0:
        #         print('Epoch Aug LinEval: [{0}][{1}/{2}]\t'
        #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #             'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #             epoch, idx, len(train_loader), batch_time=batch_time,
        #             data_time=data_time, loss=losses, top1=top1, top5=top5))
        #         sys.stdout.flush()
                   

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, classifier, criterion, opt, ):
    # print(top1)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    if opt.mode == 'eval' or opt.mode == 'semi':
        model.eval()
        classifier.eval()
    else:
        model.eval()

    with torch.no_grad():
        
        # predict_list = []
        # target_list = []
        end = time.time()
        for idx, (input, target, _, ) in enumerate(val_loader):
            # if opt.gpu is not None:
            #     input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float()
            input = input.cuda(opt.gpu)
            input = input.float()
            # target = target.cuda(opt.gpu, non_blocking=True)
            target = target.cuda( opt.gpu,non_blocking=True)
            # compute output
            if  opt.mode == 'eval' or opt.mode == 'semi':
                feat = model(input)
                feat = feat.detach()
                output = classifier(feat)
            else:
                output = model(input)
                # output = output.detach()

            # output_ = list(torch.max(output, 1)[1].cpu().numpy())
            # predict_list.extend(output_)

            # target_list.extend(list(target.cpu().numpy()))


            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    # print('lennnnn', len(predict_list))
    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    best_acc1 = 0
    init_seed(global_seed)
    main()
