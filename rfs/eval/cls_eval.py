from __future__ import print_function

import time

import torch

from .util import AverageMeter, accuracy


def validate(val_loader, model, criterion, opt, auxiliary=None):
    """One epoch validation"""
    batch_time = AverageMeter()
    cel_holder = AverageMeter()
    opl_holder = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, _) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            # output = model(input)
            [f0, f1, f2, f3, feat], output = model(input, is_feat=True)

            cel = criterion(output, target)
            if auxiliary is not None:
                opl = auxiliary(feat, target)
                ratio = opt.opl_ratio
                loss = cel + ratio * opl
            else:
                loss = cel

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            cel_holder.update(cel.item(), input.size(0))
            if auxiliary is not None:
                opl_holder.update(opl.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(f'Test: [{idx}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    if auxiliary is None:
        return top1.avg, top5.avg, losses.avg
    else:
        return top1.avg, top5.avg, losses.avg, [cel_holder.avg, opl_holder.avg]
