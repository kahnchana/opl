# train.py
# !/usr/bin/env	python3

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from losses import OrthogonalProjectionLoss, DistillationOrthogonalProjectionLoss, CenterLoss, \
    PerpetualOrthogonalProjectionLoss, cam_loss_kd_topk
from utils import get_network, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_train_valid_loader

embedding_dim = {
    'resnet56': 64,
    'resnet110': 64,
}


def train(epoch, use_opl=False, distill=False):
    start = time.time()
    net.train()
    for batch_index, data in enumerate(training_loader):

        if args.aug:
            labels = torch.cat([data[2], data[2]])
            images = torch.cat([data[0], data[1]], dim=0)
        else:
            images, labels = data

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        # x = images
        # x_90 = x.transpose(2, 3).flip(2)
        # x_180 = x.flip(2).flip(3)
        # x_270 = x.flip(2).transpose(2, 3)
        # images = torch.cat((x, x_90, x_180, x_270), 0)
        # labels = labels.repeat(4)

        optimizer.zero_grad()
        if args.hnc:
            features, cams, outputs = net(images, get_hnc=True)
        else:
            features, outputs = net(images, get_feat=True)
        base_loss = loss_function(outputs, labels)
        op_loss, s, d = aux_loss(features, labels)
        # op_loss = aux_loss(features, labels)
        if args.hnc:
            aux_hnc, _ = hnc_loss(cams, labels)
            loss = base_loss + 0.06 * aux_hnc
        elif use_opl and args.cl:
            loss = base_loss + 0.1 * aux_loss(features, labels) + 0.1 * center_loss(features, labels)
        elif use_opl:
            # op_loss, s, d = aux_loss(features, labels)
            loss = base_loss + args.opl_ratio * op_loss
            # loss = base_loss + op_loss
        elif args.popl:
            loss = base_loss + args.opl_ratio * aux_loss(features, labels)
        elif distill:
            with torch.no_grad():
                teacher_features, _ = teacher_net(images, get_feat=True)
            loss = base_loss + args.distill_ratio * distill_loss(features, teacher_features.detach())
        else:
            loss = base_loss

        loss.backward()
        if args.cl:
            for param in center_loss.parameters():
                # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
                param.grad.data *= (0.5 / (0.1 * train_scheduler.get_last_lr()[0]))
        if args.popl:
            for param in aux_loss.parameters():
                # learning rate: 0.5
                param.grad.data *= (0.5 / (args.opl_ratio * train_scheduler.get_last_lr()[0]))
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/ce_loss', base_loss.item(), n_iter)
        # if use_opl:
        writer.add_scalar('Train/op_loss', op_loss.item(), n_iter)
        writer.add_scalar('Train/op_s', s.item(), n_iter)
        writer.add_scalar('Train/op_d', d.item(), n_iter)
        writer.add_scalar('Train/lr', train_scheduler.get_last_lr()[0], n_iter)
        if args.hnc:
            writer.add_scalar('Train/hnc_loss', aux_hnc.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True, data_loader=None, run="Test"):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    dataset_size = 0

    for (images, labels) in data_loader:
        dataset_size += labels.shape[0]
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print(dataset_size)
    print('Evaluating Network.....')
    print('{} set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        run,
        epoch,
        test_loss / dataset_size,
        correct.float() / dataset_size,
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar(f'{run}/Average loss', test_loss / dataset_size, epoch)
        writer.add_scalar(f'{run}/Accuracy', correct.float() / dataset_size, epoch)

    return correct.float() / dataset_size


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f} * Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, default="CIFAR-100", help='select dataset')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-opl', action='store_true', default=False, help='use OPL')
    parser.add_argument('-hnc', action='store_true', default=False, help='use HNC')
    parser.add_argument('-popl', action='store_true', default=False, help='use POPL')
    parser.add_argument('-rbf', action='store_true', default=False, help='use RBF kernel')
    parser.add_argument('-opl_ratio', type=float, default=0.1, help='opl ratio')
    parser.add_argument('-opl_gamma', type=float, default=2, help='opl ratio')
    parser.add_argument('-distill', action='store_true', default=False, help='distill')
    parser.add_argument('-distill_ratio', type=float, default=0.1, help='distill ratio')
    parser.add_argument('-teacher', type=str, default="", help='teacher network path')
    parser.add_argument('-cl', action='store_true', default=False, help='use centre loss')
    parser.add_argument('-aug', action='store_true', default=False, help='use contrastive augmentation')
    parser.add_argument('-imbalance', action='store_true', default=False, help='use imbalanced dataset')
    parser.add_argument('-eval', action='store_true', default=False, help='evaluate only')
    parser.add_argument('-pth', type=str, default=None, help='path to model folder')
    parser.add_argument('-ckpt', type=str, default=None, help='path to model .pth file')

    args = parser.parse_args()

    net = get_network(args)
    # net = torchvision.models.resnet50().cuda()
    if args.distill:
        teacher_net = get_network(args)
        distill_loss = DistillationOrthogonalProjectionLoss()
    else:
        teacher_net = None
        distill_loss = None

    training_loader, validation_loader = get_train_valid_loader(
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        valid_size=0.1
    )

    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        imbalance=args.imbalance
    )

    loss_function = nn.CrossEntropyLoss()
    params = net.parameters()
    if args.opl:
        aux_loss = OrthogonalProjectionLoss(no_norm=False, use_attention=False, gamma=args.opl_gamma)
        # aux_loss = OLELoss()
    elif args.popl:
        aux_loss = PerpetualOrthogonalProjectionLoss(num_classes=100, feat_dim=embedding_dim[args.net], no_norm=False,
                                                     use_attention=False)
        params = list(net.parameters()) + list(aux_loss.parameters())
    else:
        aux_loss = OrthogonalProjectionLoss(no_norm=False, use_attention=False)

    if args.hnc:
        hnc_loss = cam_loss_kd_topk()
    else:
        hnc_loss = None

    if args.cl:
        center_loss = CenterLoss(num_classes=100, feat_dim=2048, use_gpu=True)
        params = list(net.parameters()) + list(center_loss.parameters())
    else:
        optimizer = optim.SGD(params=params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        if args.pth is not None:
            recent_folder = args.pth
        else:
            recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net),
                                               fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # create checkpoint folder to save model
    if not args.eval:
        # use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)

        # since tensorboard can't overwrite old values
        # so the only way is to create a new tensorboard log
        writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
        input_tensor = torch.Tensor(1, 3, 32, 32)
        if args.gpu:
            input_tensor = input_tensor.cuda()
        writer.add_graph(net, input_tensor)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        json.dump(vars(args), open(f"{checkpoint_path}/config.json", "w"), indent=4)

    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.ckpt is not None:
        net.load_state_dict(torch.load(args.ckpt)['state_dict'])
    elif args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False, data_loader=test_loader, run='Test')
            print('best acc is {:0.3f}'.format(best_acc))
        if args.eval:
            pass
        else:
            recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
            if not recent_weights_file:
                raise Exception('no recent weights file were found')
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
            print('loading weights file {} to resume training.....'.format(weights_path))
            net.load_state_dict(torch.load(weights_path))

            resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    if teacher_net is not None:
        teacher_net.load_state_dict(torch.load(args.teacher))

    if args.eval:
        acc1, acc5 = validate(test_loader, net, loss_function)

    else:
        val_accuracy = []
        test_accuracy = []
        for epoch in range(1, settings.EPOCH + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue

            train(epoch, use_opl=args.opl, distill=args.distill)
            val_acc = eval_training(epoch, data_loader=validation_loader, run='Val')
            acc = eval_training(epoch, data_loader=test_loader, run="Test")
            val_accuracy.append(val_acc)
            test_accuracy.append(test_accuracy)

            # start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc
                if args.popl:
                    torch.save(aux_loss.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='centers'))
                continue

            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)

        writer.close()

        print(f"Best accuracy is: {best_acc}")
        print(f"Max val accuracy: {max(val_accuracy)}")
