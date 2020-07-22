from __future__ import print_function

import torch
import numpy as np








def visual_sim(a,pdf_name):
   import matplotlib.pyplot as plt
   from matplotlib.backends.backend_pdf import PdfPages
   plt.switch_backend('agg')
   # plt.rcParams['figure.figsize'] = (20, 20)
   plt.axis('square')

   # =================
   def make_square_axes(ax):
      """Make an axes square in screen units.

      Should be called after plotting.
      """
      ax.set_aspect(1 / ax.get_data_ratio())

   # ===================
   fig = plt.figure()
   # fig.subplots_adjust(wspace=0.2, hspace=0)
   plt.rcParams['xtick.direction'] = 'in'
   plt.rcParams['ytick.direction'] = 'in'
   ax = fig.add_subplot(111)
   vmin = 0
   vmax = 100
   pt = a
#    im = ax.imshow(pt.T, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
   im = ax.imshow(pt.T, cmap=plt.cm.rainbow)

   make_square_axes(ax)
   fig.subplots_adjust(right=0.8)
   # cbar_ax = fig.add_axes([0.81, 0.328, 0.01, 0.334])
   cbar_ax = fig.add_axes([0.79, 0.15, 0.05, 0.7])
   cb = fig.colorbar(im, cax=cbar_ax)
   # plt.axis('off')
   # plt.subplots_adjust(wspace=0.02, hspace=0.02)
   # plt.show()
   plt.savefig('./contrast_fig/{}.pdf'.format(pdf_name), format='pdf', transparent=True, dpi=300, pad_inches=0,bbox_inches='tight')
   plt.close()
   return


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    meter = AverageMeter()
