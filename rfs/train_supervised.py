from __future__ import print_function

import argparse
import json
import os
import sys
import time

import tensorboard_logger as tb_logger
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.imagenet import ImagenetFolder
from dataset.transform_cfg import transforms_options, transforms_list
from eval.cls_eval import validate
from models import model_pool
from models.util import create_model
from util import adjust_learning_rate, accuracy, AverageMeter, OrthogonalProjectionLoss, LabelSmoothing, \
    GuidedComplementEntropy, PerpetualOrthogonalProjectionLoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--opl', action='store_true', help='add opl as auxiliary loss')
    parser.add_argument('--popl', action='store_true', help='add popl as auxiliary loss')
    parser.add_argument('--opl_ratio', type=float, default=1, help='ratio for opl')
    parser.add_argument('--srl', action='store_true', help='add opl as auxiliary loss')
    parser.add_argument('--srl_ratio', type=float, default=1, help='ratio for opl')
    parser.add_argument('--label_smoothing', action='store_true', help='replace CE with label smoothing loss')
    parser.add_argument('--smoothing_ratio', type=float, default=0.1, help='ratio for label smoothing')
    parser.add_argument('--gce', action='store_true', help='replace CE with guided cross entropy')
    parser.add_argument('--gce_alpha', type=float, default=0.33, help='ratio for label smoothing')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100', 'imagenet'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    opt = parser.parse_args()

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.transform)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()

    return opt


def main():
    opt = parse_option()

    with open(f"{opt.tb_folder}/config.json", "w") as fo:
        fo.write(json.dumps(vars(opt), indent=4))

    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    elif opt.dataset == "imagenet":
        train_trans, test_trans = transforms_options["A"]
        train_dataset = ImagenetFolder(root=os.path.join(opt.data_root, "train"), transform=train_trans)
        val_dataset = ImagenetFolder(root=os.path.join(opt.data_root, "val"), transform=test_trans)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        n_cls = 1000
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = create_model(opt.model, n_cls, opt.dataset, use_srl=opt.srl)

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    if opt.label_smoothing:
        criterion = LabelSmoothing(smoothing=opt.smoothing_ratio)
    elif opt.gce:
        criterion = GuidedComplementEntropy(alpha=opt.gce_alpha, classes=n_cls)
    else:
        criterion = nn.CrossEntropyLoss()
    if opt.opl:
        auxiliary_loss = OrthogonalProjectionLoss(use_attention=True)
    elif opt.popl:
        auxiliary_loss = PerpetualOrthogonalProjectionLoss(feat_dim=640)
    else:
        auxiliary_loss = None

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        if auxiliary_loss is not None:
            auxiliary_loss = auxiliary_loss.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)
    else:
        scheduler = None

    # routine: supervised pre-training
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if auxiliary_loss is not None:
            train_acc, train_loss, [train_cel, train_opl] = train(
                epoch=epoch, train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, opt=opt,
                auxiliary=auxiliary_loss)
        else:
            train_acc, train_loss = train(
                epoch=epoch, train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, opt=opt)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('accuracy/train_acc', train_acc, epoch)
        logger.log_value('train_losses/loss', train_loss, epoch)
        if auxiliary_loss is not None:
            logger.log_value('train_losses/cel', train_cel, epoch)
            logger.log_value('train_losses/opl', train_opl, epoch)
        else:
            logger.log_value('train_losses/cel', train_loss, epoch)

        if auxiliary_loss is not None:
            test_acc, test_acc_top5, test_loss, [test_cel, test_opl] = \
                validate(val_loader, model, criterion, opt, auxiliary=auxiliary_loss)
        else:
            test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('accuracy/test_acc', test_acc, epoch)
        logger.log_value('accuracy/test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_losses/loss', test_loss, epoch)
        if auxiliary_loss is not None:
            logger.log_value('test_losses/cel', test_cel, epoch)
            logger.log_value('test_losses/opl', test_opl, epoch)
        else:
            logger.log_value('test_losses/cel', test_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


def train(epoch, train_loader, model, criterion, optimizer, opt, auxiliary=None):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    cel_holder = AverageMeter()
    opl_holder = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        # output = model(input)
        if opt.srl:
            output, srl_loss = model(input, labels=target)
        else:
            [f0, f1, f2, f3, feat], output = model(input, is_feat=True)

        cel = criterion(output, target)
        if auxiliary is not None:
            opl = auxiliary(feat, target)
            ratio = opt.opl_ratio
            loss = cel + ratio * opl
        elif opt.srl:
            loss = cel + srl_loss
        else:
            loss = cel

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        cel_holder.update(cel.item(), input.size(0))
        if auxiliary is not None:
            opl_holder.update(opl.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        if opt.popl:
            for param in auxiliary.parameters():
                # learning rate: 0.5
                param.grad.data *= (0.5 / (opt.opl_ratio * opt.learning_rate))
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            sys.stdout.flush()

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    sys.stdout.flush()

    if auxiliary is not None:
        return top1.avg, losses.avg, [cel_holder.avg, opl_holder.avg]
    else:
        return top1.avg, losses.avg


if __name__ == '__main__':
    main()
