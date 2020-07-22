import random

import numpy as np
import math
from math import sin,cos
import random
import  torch
from torch import nn
import torch.nn.functional as F


def aug_look(name, args1 = None, args2 = None):
    if 'selectFrames' in name:
        return SelectFrames(args1)
    # elif 'normalizeC' == name:
    #     return NormalizeC(args1, args2)
    # elif 'normalizeCV' == name:
    #     return NormalizeCV(args1, args2)
    elif 'subtract' in name:
        return Subtract(args1)
    # elif 'subsample' in name:
    #     return Subsample(args1)  # subsample(data_numpy, time_range)
    elif 'randomFlip' in name:
        # return RandomHorizontalFlip(args1)  # subSampleFlip(data_numpy, time_range)
        return RandomHorizontalFlip()  # subSampleFlip(data_numpy, time_range)
    elif 'zeroOutAxis' in name:
        return Zero_out_axis(args1)  # zero_out_axis(data_numpy, axis)
    # elif 'diffOnAxis' in name:
    #     return Diff_on_axis(args1)  # diff_on_axis(data_numpy, axis)
    elif 'rotate' in name:
        return Rotate( args1, args2)  # rotate(data_numpy, axis, angle)
    elif 'zeroOutJoints' in name:
        return Zero_out_joints( args1, args2)  # zero_out_joints(data_numpy, joint_list, time_range)
    elif 'gausNoise' in name:
        # return Gaus_noise( args1, args2)  # gaus_noise(data_numpy, mean= 0, std = 0.01)
        return Gaus_noise()  # gaus_noise(data_numpy, mean= 0, std = 0.01)
    elif  'gausFilter' in name:
        # return Gaus_filter(args1, args2)  # gaus_filter(data_numpy)
        return Gaus_filter()  # gaus_filter(data_numpy)
    elif 'shear' in name:
        return Shear(args1, args2)
    # elif name == 'diff':
    #     return Diff()
    else:
        raise IndentationError("wrong")



class NormalizeC(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor




class NormalizeCV(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        self.mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        self.std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        out = (tensor - self.mean) / self.std
        return  out



# numpy
class Skeleton2Image(object):
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        data = data_numpy.reshape(C, T, V * M)
        return data

# tensor
class Image2skeleton(object):
    def __call__(self, tensor):
        C, T, W = tensor.size()
        return tensor.view(C, T, 25, 2)






class SelectFrames(object):
    def __init__(self, frames):
        self.frames = frames
    def __call__(self, data_numpy):
        return data_numpy[:, :self.frames, :, :]


class ToTensor(object):
    def __call__(self, data_numpy):
        return torch.from_numpy(data_numpy)


class Subtract(object):
    def __init__(self, joint = None):
        if joint == None:
            self.joint = random.randint(0, 24)
        else:
            self.joint = joint
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((C, T, V, M))
        for i in range(V):
            x_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, self.joint, :]
        return x_new

class Subsample(object):
    def __init__(self,time_range = None):
        self.time_range = time_range
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        # frames = random.randint(1, T)
        if self.time_range == None:
            self.time_range = random.randint(1, T)
        all_frames = [i for i in range(T)]
        time_range_list = random.sample(all_frames, self.time_range)
        time_range_list.sort()
        x_new = np.zeros((C, T, V, M))
        x_new[:, time_range_list, :, :] = data_numpy[:, time_range_list, :, :]
        return x_new



class Zero_out_axis(object):
    def __init__(self, axis = None):
        self.first_axis = axis


    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)

        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp

class Diff_on_axis(object):
    def __init__(self, axis = None):
        self.first_axis = axis
    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        for t in range(T - 1):
            temp[axis_next, t, :, :] = data_numpy[axis_next, t + 1, :, :] - data_numpy[axis_next, t, :, :]
            temp[axis_next, -1, :, :] = np.zeros((V, M))
        return temp


class RandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        if random.random() < self.p:
            time_range_order = [i for i in range(T)]
            time_range_reverse = list(reversed(time_range_order))
            return data_numpy[:, time_range_reverse, :, :]
        else:
            return data_numpy.copy()

class Rotate(object):
    def __init__(self, axis = None, angle = None, ):
        self.first_axis = axis
        self.first_angle = angle
    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)

        if self.first_angle != None:
            if isinstance(self.first_angle, list):
                angle_big = self.first_angle[0] + self.first_angle[1]
                angle_small = self.first_angle[0] - self.first_angle[1]
                angle_next = random.uniform(angle_small, angle_big)
            else:
                angle_next = self.first_angle
        else:
            # angle_list = [0, 90, 180, 270]
            # angle_next = random.sample(angle_list, 1)[0]
            angle_next = random.uniform(0, 30)

        temp = data_numpy.copy()
        angle = math.radians(angle_next)
        # x
        if axis_next == 0:
            R = np.array([[1, 0, 0],
                          [0, cos(angle), sin(angle)],
                          [0, -sin(angle), cos(angle)]])
        # y
        if axis_next == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                          [0, 1, 0],
                          [sin(angle), 0, cos(angle)]])

        # z
        if axis_next == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                          [-sin(angle), cos(angle), 0],
                          [0, 0, 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp


class Zero_out_joints(object):
    def __init__(self,joint_list = None, time_range = None):
        self.first_joint_list = joint_list
        self.first_time_range = time_range
    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape

        if self.first_joint_list != None:
            if isinstance(self.first_joint_list, int):
                all_joints = [i for i in range(V)]
                joint_list_ = random.sample(all_joints, self.first_joint_list)
                joint_list_ = sorted(joint_list_)
            else:
                joint_list_ = self.first_joint_list
        else:
            random_int =  random.randint(5, 15)
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, random_int)
            joint_list_ = sorted(joint_list_)

        if self.first_time_range != None:
            if isinstance(self.first_time_range, int):
                all_frames = [i for i in range(T)]
                time_range_ = random.sample(all_frames, self.first_time_range)
                time_range_ = sorted(time_range_)
            else:
                time_range_ = self.first_time_range
        else:
            if T < 100:
                random_int = random.randint(20, 50)
            else:
                random_int = random.randint(50, 100)
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, random_int)
            time_range_ = sorted(time_range_)

        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
        # print("data_numpy",data_numpy[:, time_range, joint_list, :].shape)
        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp

class Gaus_noise(object):
    def __init__(self, mean= 0, std = 0.05):
        self.mean = mean
        self.std = std
    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(self.mean, self.std, size=(C, T, V, M))
        return temp + noise

class Gaus_filter(object):
    def __init__(self, kernel = 15, sig_list =  [0.1, 2]):
        self.g = GaussianBlurConv(3, kernel, sig_list)
    def __call__(self, data_numpy):
        return self.g(data_numpy)


class Shear(object):
    def __init__(self, s1 = None, s2 = None):
        self.s1 = s1
        self.s2 = s2

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        if self.s1 != None:
            s1_list = self.s1
        else:
            s1_list = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]
            # print(s1_list[0])
        if self.s2 != None:
            s2_list = self.s2
        else:
            s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

        R = np.array([[1,     s1_list[0], s2_list[0]],
                      [s1_list[1], 1,     s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])

        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return  temp

class Diff(object):
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((C, T, V, M))
        for t in range(T - 1):
            x_new[:, t, :, :] = data_numpy[:, t + 1, :, :] - data_numpy[:, t, :, :]
        return x_new

'''========================================================'''


# ok
def subtract(data_numpy, joint):
    C, T, V, M = data_numpy.shape
    x_new = np.zeros((C, T, V, M))
    for i in range(V):
        x_new[:, :, i, :] = data_numpy[:, :, i, :] - data_numpy[:, :, joint, :]
    return x_new




# ok
# b: crop and resize
def subsample(data_numpy, time_range):
    C, T, V, M = data_numpy.shape
    if isinstance(time_range, int):
        all_frames = [i for i in range(T)]
        time_range = random.sample(all_frames, time_range)
        time_range.sort()
    x_new = np.zeros((C, T, V, M))
    x_new[:, time_range, :, :] = data_numpy[:, time_range, :, :]
    return x_new

# ok
# c: crop,resize (and flip)
def subSampleFlip(data_numpy, time_range):
    C, T, V, M = data_numpy.shape
    assert T >= time_range, "frames longer than data"
    if isinstance(time_range, int):
        all_frames = [i for i in range(T)]
        time_range = random.sample(all_frames, time_range)
        time_range_order = sorted(time_range)
        time_range_reverse =  list(reversed(time_range_order))
    x_new = np.zeros((C, T, V, M))
    x_new[:, time_range_order, :, :] = data_numpy[:, time_range_reverse, :, :]
    return x_new

# ok
# d: color distort.(drop)
def zero_out_axis(data_numpy, axis):
    # x, y, z -> axis : 0,1,2
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    x_new = np.zeros((T, V, M))
    temp[axis] = x_new
    return temp




# ok
# e: color distort. (jitter)
def diff_on_axis(data_numpy, axis):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    for t in range(T - 1):
        temp[axis, t, :, :] = data_numpy[axis, t+1, :, :] - data_numpy[axis, t, :, :]
        temp[axis, -1, :, :] = np.zeros((V, M))
    return temp



# ok
# f: rotate
def rotate(data_numpy, axis, angle):
    temp = data_numpy.copy()
    angle = math.radians(angle)
    # x
    if axis == 0:
        R = np.array([[1, 0, 0],
                      [0, cos(angle), sin(angle)],
                      [0, -sin(angle), cos(angle)]])
    # y
    if axis == 1:
        R = np.array([[cos(angle), 0, -sin(angle)],
                       [0,1,0],
                       [sin(angle), 0, cos(angle)]])

    # z
    if axis == 2:
        R = np.array([[cos(angle),sin(angle),0],
                       [-sin(angle),cos(angle),0],
                       [0,0,1]])
    R = R.transpose()
    temp = np.dot(temp.transpose([1,2,3,0]),R)
    temp = temp.transpose(3,0,1,2)
    return temp



# ok
# g: cutout
def zero_out_joints(data_numpy, joint_list, time_range):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    # print("joint_list" ,joint_list)
    # print("time_range" ,time_range)
    if isinstance(joint_list, int):
        all_joints = [i for i in range(V)]
        joint_list_ = random.sample(all_joints, joint_list)
        joint_list_ = sorted(joint_list_)
    else:
        joint_list_ = joint_list
    if isinstance(time_range, int):
        all_frames = [i for i in range(T)]
        time_range_ = random.sample(all_frames, time_range)
        time_range_ =  sorted(time_range_)
    else:
        time_range_ = time_range
    x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
    # print("data_numpy",data_numpy[:, time_range, joint_list, :].shape)
    temp2 = temp[:, time_range_, :, :].copy()
    temp2[:, :, joint_list_, :] = x_new
    temp[:, time_range_, :, :] = temp2
    return temp



# ok
# h: gaussian noise
def gaus_noise(data_numpy, mean= 0, std = 0.01):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    noise = np.random.normal(mean, std, size=(C, T, V, M ))
    return temp + noise



# i: gaussian blur
# def gaus_filter(data_numpy):
#     g = GaussianBlurConv(3)
#     return g(data_numpy)

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel = 15, sigma = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

        # kernel = [1,4,6,4,1]
        # kernel = [ i/16.0 for i in kernel]
        # kernel = torch.DoubleTensor(kernel).unsqueeze(0).unsqueeze(0) # (1,1,5)
        # kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0) # (1,1,5)
        # kernel = kernel.repeat(channels, 1, 1, 1) # (3,1,1,5)
        # self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):

        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1) # (3,1,1,5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        if prob < 0.5:
            x = x.permute(3,0,2,1) # M,C,V,T
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            x = x.permute(1,-1,-2, 0) #C,T,V,M

        return x.numpy()




# ok
# j: sobel filtering
def diff(data_numpy):
    C, T, V, M = data_numpy.shape
    x_new = np.zeros((C, T, V, M))
    for t in range(T - 1):
        x_new[:, t, :, :] = data_numpy[:, t+1, :, :] - data_numpy[:, t, :, :]
        x_new[:, -1, :, :] = np.zeros((C, V, M))
    return x_new




'''============================================================='''

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


