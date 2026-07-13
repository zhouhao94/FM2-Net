import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from dataset import get_loader
from abc import ABCMeta, abstractmethod
import math
from Models.MambaNet import ImageDepthNet
import os
import pytorch_iou
import pytorch_ssim
import datetime
import time


criterion = nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=7, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass


class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num ' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss ' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_los s' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, learning_rate):
    update_lr_group = optimizer.param_groups
    first_group = True
    for param_group in update_lr_group:
        if first_group:
            param_group['lr'] = learning_rate * 0.1
            first_group = False
        else:
            param_group['lr'] = learning_rate
    return optimizer


def save_lr(save_dir, optimizer, epoch, iter_num, whole_iter_num, total_loss):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('At epoch: ' + str(epoch) + ', iter_num: ' + str(iter_num) + ', whole_iter_num: ' + str(
        whole_iter_num) + '\n')
    fh.write('total_loss: ' + str(total_loss.item()) + '\n')
    fh.write('encode:update:lr ' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr ' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


### bce_ssim_loss
def bce_ssim_loss(pred, target):
    bce_out = criterion(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    loss = bce_out + ssim_out
    return loss


### bce_iou_loss
def bce_iou_loss(pred, target):
    bce_out = criterion(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

### dice_loss
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def main(local_rank, num_gpus, args):
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)
    net = ImageDepthNet(args)
    net.train()
    net.cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    base_params = []
    other_params = []
    for name, p in net.named_parameters():
        if "backbone.vssm" in name and "outnorm" not in name:
            base_params.append(p)      

        else:
            other_params.append(p)    


    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])

    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')
    val_dataset = get_loader(args.validset, args.data_root, args.img_size, mode='train')

    combined_train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    sampler = torch.utils.data.distributed.DistributedSampler(
        #train_dataset,
        combined_train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=args.batch_size, num_workers=16,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    iter_num = math.floor(len(train_loader.dataset) / args.batch_size)
    total_iter_num = args.epochs * iter_num
    lr_policy = WarmUpPolyLR(args.lr, args.lr_power, total_iter_num, iter_num * args.warm_up_epoch)

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(total_iter_num, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0
        start_time = time.time()

        for i, data_batch in enumerate(train_loader):

            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                                     Variable(depths.cuda(local_rank, non_blocking=True)), \
                                                     Variable(label_224.cuda(local_rank, non_blocking=True)), \
                                                     Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()), \
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                              Variable(contour_28.cuda()), \
                                                              Variable(contour_56.cuda()), Variable(contour_112.cuda())

            outputs_saliency = net(images, depths)
            
            d1, d2, d3, d4, ud2, ud3, ud4 = outputs_saliency

            bce_loss1 = criterion(d1, label_224)
            bce_loss2 = criterion(d2, label_56)
            bce_loss3 = criterion(d3, label_28)
            bce_loss4 = criterion(d4, label_14)

            iou_loss1 = bce_iou_loss(d1,  label_224)
            iou_loss2 = bce_iou_loss(ud2, label_224)
            iou_loss3 = bce_iou_loss(ud3, label_224)
            iou_loss4 = bce_iou_loss(ud4, label_224)

            c_loss1 = bce_ssim_loss(d1,  label_224)
            c_loss2 = bce_ssim_loss(ud2, label_224)
            c_loss3 = bce_ssim_loss(ud3, label_224)
            c_loss4 = bce_ssim_loss(ud4, label_224)

            d_loss1 = dice_loss(d1,   label_224)
            d_loss2 = dice_loss(ud2,  label_224)
            d_loss3 = dice_loss(ud3,  label_224)
            d_loss4 = dice_loss(ud4,  label_224)

            BCE_total_loss = bce_loss1 + 0.5*bce_loss2 + 0.3*bce_loss3 + 0.2*bce_loss4
            IoU_total_loss = iou_loss1 + 0.5*iou_loss2 + 0.3*iou_loss3 + 0.2*iou_loss4
            Edge_total_loss = c_loss1 + 0.5*c_loss2 + 0.3*c_loss3 + 0.2*c_loss4
            Dice_total_loss = d_loss1 + 0.5*d_loss2 + 0.3*d_loss3 + 0.2*d_loss4
            total_loss = Edge_total_loss + BCE_total_loss + IoU_total_loss + Dice_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += bce_loss1.cpu().data.item()

            if (i*1)%10==0 or (i+1)==iter_num:
                print(
                    'epoch: {0} --- iter_num: {1}/{2} --- whole_iter_num: {3} --- total_loss: {4:.6f} --- bce loss: {5:.6f} --- e loss: {6:.6f}'.format(
                        epoch+1, i+1, iter_num, (whole_iter_num + 1), total_loss.item(), bce_loss1.item(), c_loss1.item()
                        ))

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            # update learning rate
            lr = lr_policy.get_lr(whole_iter_num)
            optimizer = adjust_learning_rate(optimizer, learning_rate=lr)

            if (i%50==0) or (i+1)==iter_num:
                save_dir = f'./loss_{current_time}.txt'
                save_lr(save_dir, optimizer, epoch+1, i+1, whole_iter_num, total_loss)

        if (epoch >= args.checkpoint_start_epoch) and (epoch % args.checkpoint_step==0) or (epoch == args.epochs - 1):
            if local_rank == 0:
                torch.save(net.state_dict(),
                           args.save_model_dir + 'UVST_epoch_{}.pth'.format(epoch + 1))

        end_time = time.time()
        epoch_duration = end_time - start_time  # 计算每个 epoch 的耗时
        print('Epoch finished ! Loss: {}, Epoch duration: {}'.format(epoch_total_loss / iter_num, epoch_duration))
        save_lossdir = f'./loss_{current_time}.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss / iter_num, epoch + 1)

