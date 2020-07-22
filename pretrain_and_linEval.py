"""
Training MoCo and Instance Discrimination

InsDis: Unsupervised feature learning via non-parametric instance discrimination
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket
from torch.nn import LSTM

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from util import adjust_learning_rate, AverageMeter

from NCE.NCEAverage import MemoryInsDis,NTLogistic,TripletMarginLoss
from NCE.NCEAverage import MemoryMoCo, End_to_end
# from NCE.NCEAverage import MemoryMoCoTAP
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import  random
from feeders.tools import aug_look, NormalizeC, NormalizeCV, ToTensor, Skeleton2Image, Image2skeleton
from models.Models import LinearClassifier, Bi_lstm, Bi_lstm_with_head, GRU_model, RNN_model, aug_transfrom

try:
    from apex import amp, optimizers
except ImportError:
    pass
"""
TODO: python 3.6 ModuleNotFoundError
"""

max_body = 2
joints = 25
dim = 3
global_seed = 1
def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    # parser.add_argument('--moco', action='store_true', help='using MoCo (otherwise Instance Discrimination)')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--nesterov', type=bool, default=False, )
    parser.add_argument('--softmax', type=bool, default=True, help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--feeder', default='feeders.skeleton_feeder.Feeder', help='data loader will be used')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--selected_frames', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,70', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    '''<<<<<<<<<===================================================='''
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'uwa3d', 'ucla','sbu'])
    
    ntu_dataset = 'view60' #choices=['sub60', 'view60', 'sub120','view120']  when dataset == 'ntu'

    # augmentation setting
    # best : subtract1_randomFlip_shear
    parser.add_argument('--aug1', type=str, default='subtract_randomFlip_shear' )

    parser.add_argument('--aug2', type=str, default='subtract_randomFlip_shear' )
    # model definition
    parser.add_argument('--model', type=str, default='lstm',
                        choices=[ 'lstm', 'GRU', 'RNN'])
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--lstm_layer', type=int, default=1)
    # loss function
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.06)
    parser.add_argument('--nce_m', type=float, default=0.5)
    # memory setting
    parser.add_argument('--lossType',type=str, default='infonce', choices=['infonce','logit','MTrip'] )
    parser.add_argument('--mom_flag',type=str, default='moco', choices=['moco','end','ins'] )
    # parser.add_argument('--bank_or_end', default='bank', )
    parser.add_argument('--tap', default='mean', )
    parser.add_argument('--head', default=False, )
    parser.add_argument('--head_dim', default=512, )
    parser.add_argument('--head_flag', default='nonlinear',choices=['linear','nonlinear'] )

    if ntu_dataset == 'sub60':
        parser.add_argument('--ntu_dataset', type=str, default='sub60')
        parser.add_argument('--train_data_path', type=str,
                            default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/train_data.npy')
        parser.add_argument('--train_label_path', type=str,
                        default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/train_label.pkl')
    
    if ntu_dataset == 'view60':   
        parser.add_argument('--ntu_dataset', type=str, default='view60')               
        parser.add_argument('--train_data_path', type=str,
                            default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/train_data.npy')
        parser.add_argument('--train_label_path', type=str,
                            default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/train_label.pkl')

    if ntu_dataset == 'sub120':  
        parser.add_argument('--ntu_dataset', type=str, default='sub120')               
        parser.add_argument('--train_data_path', type=str,
                            default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/train_data.npy')
        parser.add_argument('--train_label_path', type=str,
                            default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/train_label.pkl')
    
    if ntu_dataset == 'view120': 
        parser.add_argument('--ntu_dataset', type=str, default='view120')               
        parser.add_argument('--train_data_path', type=str,
                            default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/train_data.npy')
        parser.add_argument('--train_label_path', type=str,
                            default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/train_label.pkl')



    '''=============================================================>>>>>'''
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('--norm', type=str, default='None', choices = ['normalizeC', 'normalizeCV',"None"])
    parser.add_argument('--norm', type=str, default='None', choices = ["None"])


    opt = parser.parse_args()
    # set the path according to the environment
    if hostname.startswith('ubuntu'):
        opt.model_path = '/data5/xushihao/data/MoCo/CMC/{}_models'.format(opt.dataset)
        opt.tb_path = '/data5/xushihao/data/MoCo/CMC/{}_tensorboard'.format(opt.dataset)
    else:
        raise NotImplementedError('server invalid: {}'.format(hostname))


    if opt.dataset == 'uwa3d':
        # max_frame = 167
        # avg_frame = 55
        uwa3d_data_folder = '/data5/xushihao/data/UWA3d_feeder_data'
        opt.uwa3d_train = ['12', '13', '14', '23', '24', '34', ]
        opt.uwa3d_test1 = ['3','2','2','1','1','1']
        opt.uwa3d_test2 = ['4','4','3','4','3','2']
        opt.fold = 5 # starts from index = 0 
        opt.uwa3d_test_view = '2'
        
        opt.train_data_path = '{}/train{}_test{}{}/training_{}/train_data.npy'.format(uwa3d_data_folder, opt.uwa3d_train[opt.fold], 
        opt.uwa3d_test1[opt.fold],opt.uwa3d_test2[opt.fold], opt.uwa3d_train[opt.fold])
        opt.train_label_path = '{}/train{}_test{}{}/training_{}/train_label.pkl'.format(uwa3d_data_folder, opt.uwa3d_train[opt.fold], 
        opt.uwa3d_test1[opt.fold],opt.uwa3d_test2[opt.fold],opt.uwa3d_train[opt.fold] )
        opt.hidden_units = 256
        opt.lstm_layer = 1
        opt.selected_frames = 60
        opt.learning_rate = 0.01
        opt.nce_k = 511
        opt.epochs = 60
        opt.lr_decay_epochs = '30,70,150'
    elif opt.dataset == 'ucla':
        opt.ucla_test_view = 1 
        if opt.ucla_test_view == 1:
            opt.train_data_path = r'/data5/xushihao/multiview_skeleton/view1_for_test/train_data.npy'
            opt.train_label_path = r'/data5/xushihao/multiview_skeleton/view1_for_test/train_label.pkl'
        if opt.ucla_test_view == 2:
            opt.train_data_path = r'/data5/xushihao/multiview_skeleton/view2_for_test/train_data.npy'
            opt.train_label_path = r'/data5/xushihao/multiview_skeleton/view2_for_test/train_label.pkl'
        if opt.ucla_test_view == 3:
            opt.train_data_path = r'/data5/xushihao/multiview_skeleton/view3_for_test/train_data.npy'
            opt.train_label_path = r'/data5/xushihao/multiview_skeleton/view3_for_test/train_label.pkl'
        opt.hidden_units = 256
        opt.lstm_layer = 1
        opt.selected_frames = 60
        opt.learning_rate = 0.01
        opt.nce_k = 500
        if opt.ucla_test_view == 1:
            opt.batch_size = 31
        opt.epochs = 60
        opt.lr_decay_epochs = '30,70,150'
    elif opt.dataset == 'sbu':
        opt.print_freq = 1
        opt.train_data_path = r'/data5/xushihao/data/SBU'
        # opt.train_label_path = r'/data5/xushihao/multiview_skeleton/train_label.pkl'
        opt.hidden_units = 128
        opt.lstm_layer = 2
        opt.selected_frames = 40
        opt.learning_rate = 0.01
        opt.nce_k = 200
        opt.epochs = 60
        opt.lr_decay_epochs = '30,70,150'
        opt.feeder = 'feeders.sbu_feeder.SBU_feeder'
        opt.fold = 0 #  choices: [0,1,2,3,4]


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if  'subtract' in opt.aug1:
        if opt.norm != "None":
            raise NotImplementedError('subtract does not need norm')


    opt.method = 'softmax' if opt.softmax else 'nce'



    flag = opt.train_data_path.split("/")[-2]
    if opt.dataset == 'ntu':
        ntu120 = ('120' in opt.train_data_path.split("/")[-3])
        if ntu120:
            d = '120'
        else:
            d = '60'
        flag = "{}_{}".format(flag, d)
    if opt.dataset == 'sbu':
        flag = 'sbu_{}fd'.format(opt.fold)
    elif opt.dataset == 'uwa3d':
        flag = 'uwa3d_train{}_test{}'.format(opt.uwa3d_train[opt.fold], opt.uwa3d_test_view)

    prefix = '{}_{}'.format(opt.norm, opt.alpha)
    hidden_units = opt.hidden_units

    if opt.head :
        head_name = "{}_{}".format(opt.head_dim, opt.head_flag[0:3])
    else:
        head_name = ''


    # if opt.dataset == 'sbu':
    #     tail_flag =  '{}_fold{}'.format(opt.lossType, opt.fold)
    # else:
    #     tail_flag = opt.lossType
    opt.model_name = '{}_{}_{}_{}_tprt{}_{}{}_layer{}_lr_{}_bsz_{}_epoch_{}_head{}{}_tap_{}_{}F_GLC{}_{}_nceM{}_{}'.format(  flag,
                                                                                                         prefix,
                                                                                                        opt.method,
                                                                                                        opt.nce_k,
                                                                                                        opt.nce_t,
                                                                                                        opt.model,
                                                                                                        hidden_units,
                                                                                                        opt.lstm_layer,
                                                                                                        opt.learning_rate,
                                                                                                        opt.batch_size,
                                                                                                        opt.epochs,
                                                                                                        opt.head,
                                                                                                        head_name,
                                                                                                        opt.tap,
                                                                                                        opt.selected_frames,
                                                                                                                global_seed,
                                                                                                                opt.mom_flag,
                                                                                                                opt.nce_m,
                                                                                                                opt.lossType
                                                                                                                 # flag_clip
                                                                                          )
    
    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)
        print("=================use amp")

    if opt.aug1 == opt.aug2:
        opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.aug1)
    else:
        opt.model_name = '{}_aug1_{}_aug2_{}'.format(opt.model_name, opt.aug1, opt.aug2)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






def main():
    args = parse_option()
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))




    args1 = [1, None, None, None, None, None, None, None, None, None, ]
    args.list_args = args1
    transform1 = aug_transfrom(args.aug1, args1, args.norm, None, args)

    # ==============================================
    args2 = args1
    transform2 = aug_transfrom(args.aug2, args2, args.norm, None, args)

    # =====================  feeder
    Feeder = import_class(args.feeder)
    if args.dataset == 'sbu':
        train_dataset = Feeder(args.train_data_path,
                               args.fold,
                               'train',
                               transform1,
                               transform2,
                               args.dataset,
                               False,
                               None
                               )  
    else:
        train_dataset = Feeder(args.train_data_path,
                args.train_label_path,
                transform1 = transform1,
                transform2 = transform2,
                dataset = args.dataset, 
                )
    print(len(train_dataset))
    train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # drop_last=True,
        worker_init_fn=init_seed(global_seed),
        # pin_memory=True,
        sampler=train_sampler)

    # create model and optimizer
    n_data = len(train_dataset)

    if 'lstm' in args.model or 'GRU' in args.model or 'RNN' in args.model:
        if args.dataset == 'uwa3d':
            input_size = 15 * 3
        elif args.dataset == 'ucla':
            input_size = 60
        elif args.dataset == 'sbu':
            input_size = 90
        else:
            input_size = max_body * joints * dim

        if args.head:
            model = Bi_lstm_with_head(input_size = input_size, args = args)
            if args.mom_flag != 'ins':
                model_ema = Bi_lstm_with_head(input_size=input_size, args = args)
                    
        else:
            if args.model == 'GRU':
                model = GRU_model(input_size = input_size, args = args)
                if args.mom_flag != 'ins':
                    model_ema = GRU_model(input_size=input_size, args = args)
            elif args.model == 'RNN':
                model = RNN_model(input_size = input_size, args = args)
                if args.mom_flag != 'ins':
                    model_ema = RNN_model(input_size=input_size, args = args)
            else:
                model = Bi_lstm(input_size = input_size, args = args)
                if args.mom_flag != 'ins':
                    model_ema = Bi_lstm(input_size=input_size, args = args)
            # model = LSTM(input_size=input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer, bidirectional=True)
            # model_ema = LSTM(input_size=input_size, hidden_size=args.hidden_units, num_layers=args.lstm_layer, bidirectional=True)


    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    sum_parameters = sum(param.numel() for param in model.parameters())
    print('# model parameters:', sum_parameters)
    args.sum_parameters = sum_parameters

    # copy weights from `model' to `model_ema'
    if args.mom_flag != 'ins':
        moment_update(model, model_ema, 0)
    # set the contrast memory and criterion
    if 'lstm' in args.model or 'GRU' in args.model or 'RNN' in args.model:
        if args.head:
            size = args.head_dim
        else:
            if 'bi' in args.model:
                size = args.hidden_units * 2
            else:
                size = args.hidden_units


    if args.mom_flag == 'moco':
        contrast = MemoryMoCo(size, n_data, args.nce_k, args.nce_t, args.softmax, args.mom_flag).cuda(args.gpu)
    elif args.mom_flag == 'end':
        contrast = End_to_end(size, n_data, args.nce_k, args.nce_t, args.softmax,  args.mom_flag, args).cuda(args.gpu)
    elif args.mom_flag == 'ins':
        contrast = MemoryInsDis(size, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax).cuda(args.gpu)



    if args.lossType == 'infonce':
        criterion = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    elif args.lossType == 'logit':
        criterion = NTLogistic()
    elif args.lossType == 'MTrip':
        criterion = TripletMarginLoss(1)

    criterion = criterion.cuda(args.gpu)

    model = model.cuda()


    if args.mom_flag == 'moco':
        model_ema = model_ema.cuda()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
    elif args.mom_flag == 'end':
        print("end to end")
        model_ema = model_ema.cuda()
        optimizer = torch.optim.SGD(list(model.parameters())+list(model_ema.parameters()),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
    elif args.mom_flag == 'ins':
        optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.learning_rate,
                            momentum=args.momentum,
                            nesterov=args.nesterov,
                            weight_decay=args.weight_decay)

    cudnn.benchmark = True


    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.mom_flag != 'ins':
                model_ema.load_state_dict(checkpoint['model_ema'])

            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    iteration = []
    loss_list = []
    for epoch in range(args.start_epoch, args.epochs + 2):
        iteration.append(epoch)
        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        if args.mom_flag == 'moco' or args.mom_flag == 'end':
            loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args)

        else:
            loss, prob = train_ins(epoch, train_loader, model, contrast, criterion, optimizer, args)
        loss_list.append(loss)
        
        time2 = time.time()
        # print('loss_list',loss_list)
        # break
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('ins_loss', loss, epoch)
        logger.log_value('ins_prob', prob, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.mom_flag != 'ins':
                state['model_ema'] = model_ema.state_dict()
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print('==> Saving...')

        state = {
            'opt': args,
            'model': model.state_dict(),
            'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if args.mom_flag != 'ins':
            state['model_ema'] = model_ema.state_dict()
        if args.amp:
            state['amp'] = amp.state_dict()
        save_file = os.path.join(args.model_folder, 'current.pth')
        torch.save(state, save_file)
        del state
        torch.cuda.empty_cache()

    #save loss in pdf and .npy
    print(args.model_name)
    print('\n loss: \n', loss_list)
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(iteration, loss_list, label='loss')
    plt.draw()
    plt.tight_layout()
    save_pdf_path = os.path.join(args.model_folder, "loss.pdf")
    plt.savefig(save_pdf_path, format='pdf', transparent=True, dpi=300, pad_inches=0,
                bbox_inches='tight')
    plt.close()

    loss_npy = os.path.join(args.model_folder, "loss.npy")
    np.save(loss_npy, np.array(loss_list))


    args.model_file_path = os.path.join(args.model_folder, 'ckpt_epoch_{}.pth'.format(args.epochs))
    return args




def train_ins(epoch, train_loader, model, contrast, criterion, optimizer, opt):
    """
    one epoch training for instance discrimination
    """

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
        index = index.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        feat = model(inputs)
        out = contrast(feat, index)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        if epoch == 1:
            pass
        else:
            if opt.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
            print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg



def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt, 
               ):
    """
    one epoch training for instance discrimination
    """

    model.train()
    if opt.mom_flag == 'moco':
        model_ema.eval()
    if opt.mom_flag == 'end':
        model_ema.train()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()

    # if opt.mom_flag == False:
    #     flag_queue = 512
    #     if opt.nce_k > flag_queue:
    #         query_store_for_batch = []
    #         key_store_for_batch = []
    #         flag_batch = 0
    #         times = int(opt.nce_k / opt.batch_size)

    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
        index = index.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        x1, x2 = torch.split(inputs, [3, 3], dim=1)
        # ids for ShuffleBN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        # print(x1.size())
        feat_q = model(x1)


        # original
        if opt.mom_flag == 'moco':
            with torch.no_grad():
                x2 = x2[shuffle_ids]
                feat_k = model_ema(x2)
                feat_k = feat_k[reverse_ids]
        elif opt.mom_flag == 'end':
            x2 = x2[shuffle_ids]
            feat_k = model_ema(x2)
            feat_k = feat_k[reverse_ids]

                    

        # =============================
        out = contrast(feat_q, feat_k)
        # print(list(out.size()))
        if len(list(out.size()) ) < 2:
            out = out.unsqueeze(0)
            print('fuck',out.size())
        loss = criterion(out)
        # print('loss',loss)
        prob = out[:, 0].mean()
        # print('loss', loss.item())
        # print('loss', loss.size())
        # ===================backward=====================
        optimizer.zero_grad()
        if epoch == 1:
            pass
        else:
            if opt.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # if opt.clip:
            #     nn.utils.clip_grad_value_(model.parameters(), clip_value=opt.clip_val)
            # if opt.dataset == 'uwa3d':
            #     nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        if opt.mom_flag == 'moco' and epoch != 1:
            moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=loss_meter, prob=prob_meter))
            print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg



############################# linear evaluation
def parse_option_lin_eval(args_pretrain):

    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
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
    parser.add_argument('--model_path', type=str, default=args_pretrain.model_file_path, help='the model to test')
    '''<<<<<<<<<===================================================='''
    # model definition
    parser.add_argument('--model', type=str, default=args_pretrain.model,
                        choices=['lstm','GRU','RNN'])

    parser.add_argument('--hidden_units', type=int, default=args_pretrain.hidden_units)
    parser.add_argument('--lstm_layer', type=int, default=args_pretrain.lstm_layer)
    # dataset
    parser.add_argument('--dataset', type=str, default=args_pretrain.dataset, choices=['ntu', 'uwa3d', 'ucla', 'sbu'])
    # augmentation
    parser.add_argument('--FTaug', type=str, default='subtract1')
    parser.add_argument('--Vaug', type=str, default= 'subtract1')
    # parser.add_argument('--lin_eval_aug', type=str, default= 'subtract1_randomFlip_shear')

    parser.add_argument('--semiFT', type=str, default= 'subtract1')

    if args_pretrain.dataset == 'ntu':
        if args_pretrain.ntu_dataset == 'sub60':
            parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/train_data.npy')
            parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/train_label.pkl')
            parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/val_data.npy')
            parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xsub/val_label.pkl')
        if args_pretrain.ntu_dataset == 'view60':
            parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/train_data.npy')
            parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/train_label.pkl')
            parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/val_data.npy')
            parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_60act/xview/val_label.pkl')
        if args_pretrain.ntu_dataset == 'sub120':
            parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/train_data.npy')
            parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/train_label.pkl')
            parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/val_data.npy')
            parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xsub/val_label.pkl')
        if args_pretrain.ntu_dataset == 'view120':
            parser.add_argument('--train_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/train_data.npy')
            parser.add_argument('--train_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/train_label.pkl')
            parser.add_argument('--val_data_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/val_data.npy')
            parser.add_argument('--val_label_path', type=str, default='/data5/xushihao/data/ntu_raw_data/st_gcn_preprocess_120act/xview/val_label.pkl')
        
        
    parser.add_argument('--tap',  default=args_pretrain.tap, help='avg on time for lstm')
    parser.add_argument('--mode',  default= 'eval', choices=['eval'])
    parser.add_argument('--semi_rate', type= int, default=0.1, )
    parser.add_argument('--epochs_semi_ft', type= int, default=50, )
    '''=============================================================>>>>>'''
    parser.add_argument('--norm', type=str, default='None', choices=['normalizeC', 'normalizeCV', "None"])
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--syncBN', action='store_true', help='enable synchronized BN')

    opt = parser.parse_args()

    opt.dataset = opt.model_path.split('/')[6].split('_')[0]
    # set the path according to the environment
    if hostname.startswith('ubuntu'):
        opt.save_path = '/data5/xushihao/data/MoCo/CMC/{}_linear'.format(opt.dataset)
        opt.tb_path = '/data5/xushihao/data/MoCo/CMC/{}_linear_tensorboard'.format(opt.dataset)



    opt.model_name = opt.model_path.split('/')[-2]
    opt.epoch_pth = opt.model_path.split('/')[-1]

    opt.head = (opt.model_name[opt.model_name.find('head')+4] == 'T')
    # model or ema
    opt.model_or_model_ema = 'model'

    if '_' in opt.epoch_pth:
        opt.epoch_pth = opt.epoch_pth.split('.')[0].split('_')[-1]
    else:
        opt.epoch_pth = 'current'
    if opt.dataset == 'uwa3d':
        
        uwa3d_data_folder = '/data5/xushihao/data/UWA3d_feeder_data'
        # opt.uwa3d_train = ['12', '13', '14', '23', '24', '34', ]
        # opt.uwa3d_test1 = ['3','2','2','1','1','1']
        # opt.uwa3d_test2 = ['4','4','3','4','3','2']
        # opt.fold = 1
        # opt.fold_test = 3

        opt.uwa3d_train = args_pretrain.uwa3d_train
        opt.uwa3d_test1 = args_pretrain.uwa3d_test1
        opt.uwa3d_test2 = args_pretrain.uwa3d_test2
        opt.fold =  args_pretrain.fold
        opt.uwa3d_test_view = args_pretrain.uwa3d_test_view


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
        opt.ucla_test_view = args_pretrain.ucla_test_view
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
        opt.fold = int(opt.model_path.split('/')[-2].split('_')[1][0])


    # =====================================
    if opt.mode == 'eval':
        opt.learning_rate = 1
        opt.epochs = 90
        opt.lr_decay_epochs = '15,35,60,75'
        opt.weight_decay = 0
        # opt.weight_decay = 0.0001


    elif opt.mode == 'semi':
        opt.learning_rate = 0.01
        opt.weight_decay = 0
        if opt.semi_rate == 0.01:
            opt.epochs_semi_ft = 30
            opt.lr_decay_epochs = '10,20'

        elif opt.semi_rate == 0.1:
            opt.epochs_semi_ft = 15
            opt.lr_decay_epochs = '8,16'

    global_seed_pretrain = int(opt.model_name[opt.model_name.index('GLC') + 3])
    print(global_seed_pretrain)
    if global_seed != global_seed_pretrain:
        raise  ValueError("wrong seed")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.lstm_layer = int(opt.model_name[opt.model_name.index('layer') + 5] )
    if opt.model_name[opt.model_name.index('F_') - 3] == '_':
        opt.selected_frames = int(opt.model_name[opt.model_name.index('F_') - 2: opt.model_name.index('F_')])
    else:
        opt.selected_frames = int(opt.model_name[opt.model_name.index('F_') - 3 : opt.model_name.index('F_') ])



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
    # if opt.FTaug != opt.model_name[opt.model_name.index('aug') + 4 :]:
    #     if opt.FTaug != opt.model_name[opt.model_name.find('aug') + 5 : opt.model_name.find('aug2') -1 ]:
    #         raise IndentationError('wrong aug match for training and test')
    if opt.model_name[opt.model_name.index('aug') + 4 :] == 'None':
        opt.FTaug = 'None'
        opt.Vaug = 'None'
    else:
        print("be careful the mismatch of aug")


    if opt.dataset == 'ucla' or opt.dataset == 'uwa3d':
        norm_name = opt.model_name.split('_')[2]
    else:
        norm_name = opt.model_name.split('_')[2]

    opt.norm = norm_name
    print(opt.norm)
    # if opt.norm != norm_name:
    #     raise IndentationError('wrong norm match for training and test')

    if opt.dataset == 'ntu':
        ntu_120 = ('120' in opt.train_data_path.split('/')[-3])
        ntu_120_self_sup = ('120' in opt.model_name.split('_')[1])
        if ntu_120 != ntu_120_self_sup:
            raise IndentationError('wrong dataset match for training and test')

        dataset_flag = opt.train_data_path.split('/')[-2] #xsub or xview
        if opt.model_name.split('_')[0] != dataset_flag:
            raise IndentationError('wrong dataset match for training and test')


    if opt.mode == 'semi':
        mode_flag = "{}{}_{}".format(opt.mode, opt.semi_rate, opt.epochs_semi_ft)
    else:
        mode_flag = opt.mode 


    # if opt.dataset == 'uwa3d':
    #     model_flag = model_flag 

    if opt.FTaug == opt.Vaug:
        aug_flag = '2aug_{}'.format(opt.FTaug)
    else:
        aug_flag = "FTaug_{}_Vaug_{}".format(opt.FTaug, opt.Vaug)
    if opt.mode == 'supervise' or opt.mode == 'finetuneR':
        ntu_120 = ('120' in opt.train_data_path.split('/')[-3])
        if ntu_120:
            flag_sup = 120
        else:
            flag_sup = 60
        opt.model_name = '{}{}_{}{}_layer{}_lr{}_bsz_{}_epoch_{}_wd{}_nstr{}_tap{}_{}F_GLC{}_{}_{}_{}'.format(opt.train_data_path.split('/')[-2],
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
                                                                                                                      opt.epoch_pth,
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






def main_lin_eval(args_pretrain):

    global best_acc1
    best_acc1 = 0

    args = parse_option_lin_eval(args_pretrain)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    # ====================================================
    Feeder = import_class(args.feeder)
    train_sampler = None
    # ===================================================
    if args.mode == 'semi':
        args_semi_ft = [1, None, None, None, None, None]
        transform_semi_ft = aug_transfrom(args.semiFT, args_semi_ft, args.norm, None, args)

        train_dataset_semi_ft = Feeder(args.train_data_path,
                                       args.train_label_path,
                                       True,
                                       transform_semi_ft,
                                       None,
                                       args.dataset, (args.mode == 'semi'), args.semi_rate
                                       )
    #==================================================
    args_train = [1, None, None, None]
    transformFT = aug_transfrom(args.FTaug, args_train, args.norm, None, args)




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
                               dataset = args.dataset,
                               )
    #==================================================
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
    print('len_val_loader',len(val_loader))

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
            input_size = 60
        elif args.dataset == 'uwa3d':
            input_size = 45
        elif args.dataset == 'sbu':
            input_size = 90
        else:
            input_size = max_body * joints * dim

        if args.mode == 'supervise' or args.mode == 'finetune' or args.mode == 'finetuneR':
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
            save_name = '{}_epoch_{}_bestAcc_{:.1f}.pth'.format(args.model, epoch, float(best_acc1.cpu().numpy()))
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
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(iteration, loss_list, label='loss')
        plt.draw()
        plt.tight_layout()
        save_pdf_path = os.path.join(args.save_folder, "semi_ft_loss.pdf")
        plt.savefig(save_pdf_path, format='pdf', transparent=True, dpi=300, pad_inches=0,
                    bbox_inches='tight')
        plt.close()

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
    for idx, (input, target) in enumerate(train_loader):
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
 
                   

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, classifier, criterion, opt, ):
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
    init_seed(global_seed)
    args_pretrain = main()
    best_acc1 = 0
    print('====== linear evaluation ===========')
    main_lin_eval(args_pretrain)