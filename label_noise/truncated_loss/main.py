from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import json
from models.resnet import *
from utils import *


from data.cifar import CIFAR10, CIFAR100
from TruncatedLoss import TruncatedLoss, OrthogonalProjectionLoss, CrossEntropyLoss
parser = argparse.ArgumentParser(
    description='PyTorch TruncatedLoss')

parser.add_argument('--resume', '-r', type=str, default=None, help='resume from checkpoint')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size', '-b', default=128,
                    type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_prune', default=40, type=int,
                    help='number of total epochs to run')
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--gamma', type = float, default = 0.1)
parser.add_argument('--schedule', nargs='+', type=int)
parser.add_argument('--opl', action='store_true', default=False, help='use OPL')
parser.add_argument('--ce', action='store_true', default=False, help='use CE')
parser.add_argument('--opl_ratio', type=float, default=0.1, help='opl ratio')

best_acc = 0
args = parser.parse_args()

os.makedirs(f"checkpoint/{args.sess}")
with open(f"checkpoint/{args.sess}/config.json", "w") as fo:
    fo.write(json.dumps(vars(args), indent=4))


def main():
    
    use_cuda = torch.cuda.is_available()
    global best_acc 
 
    # load dataset
        
    if args.dataset=='cifar10':
        print(f"Using {args.dataset}")
        num_classes=10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
        ])

        train_dataset = CIFAR10(root='/home/kanchanaranasinghe/data/cifar',
                                    download=True,  
                                    train=True, 
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                               )
        
        test_dataset = CIFAR10(root='/home/kanchanaranasinghe/data/cifar',
                                    download=True,  
                                    train=False, 
                                    transform=transform_test,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                              )

    elif args.dataset=='cifar100':
        print(f"Using {args.dataset}")
        num_classes=100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])
        train_dataset = CIFAR100(root='/home/kanchanaranasinghe/data/cifar',
                                    download=True,  
                                    train=True, 
                                    transform=transform_train,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )
        
        test_dataset = CIFAR100(root='/home/kanchanaranasinghe/data/cifar',
                                    download=True,  
                                    train=False, 
                                    transform=transform_test,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )

    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Model
    if args.resume is not None:
        # Load checkpoint.
        print(f'==> Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(f"{args.resume}")
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print(f'==> Building model.. (Default : {args.model})')
        start_epoch = 0
        if args.model == "resnet18":
            net = ResNet18(num_classes)
        elif args.model == "resnet34":
            net = ResNet34(num_classes)
        else:
            raise NotImplementedError(f"Invalid model: {args.model}")

    result_folder = f"checkpoint/{args.sess}"
    log_name = f"{result_folder}/{args.model}_{args.sess}.csv"

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')

    if args.ce:
        print(f"Using CE loss")
        criterion = CrossEntropyLoss()
    else:
        print(f"Using Truncated loss")
        criterion = TruncatedLoss(trainset_size=len(train_dataset)).cuda()

    if args.opl:
        print(f"Using OP loss")
        aux_criterion = OrthogonalProjectionLoss()
    else:
        aux_criterion = None
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.schedule, gamma=args.gamma)

    if not os.path.exists(log_name):
        with open(log_name, 'w') as logfile:
            log_writer = csv.writer(logfile, delimiter=',')
            log_writer.writerow(
                ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

    for epoch in range(start_epoch, args.epochs):
        
        train_loss, train_acc = train(epoch, trainloader, net, criterion, optimizer, aux_criterion, args)
        test_loss, test_acc = test(epoch, testloader, net, criterion)

        with open(log_name, 'a') as logfile:
            log_writer = csv.writer(logfile, delimiter=',')
            log_writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        scheduler.step()


# Training
def train(epoch, trainloader, net, criterion, optimizer, aux=None, opt=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 1

    if not args.ce and (epoch+1) >= args.start_prune and (epoch+1) % 10 == 0:
        checkpoint = torch.load(f'checkpoint/{args.sess}/best.pth')
        net = checkpoint['net']
        net.eval()
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            criterion.update_weight(outputs, targets, indexes)
        now = torch.load(f'checkpoint/{args.sess}/current_net.pth')
        net = now['current_net']
        net.train()
    
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        features, outputs = net(inputs, get_feat=True)
        if aux is not None:
            loss = criterion(outputs, targets, indexes) + opt.opl_ratio * aux(features, targets)
        else:
            loss = criterion(outputs, targets, indexes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / batch_idx, 100. * correct / total


def test(epoch, testloader, net, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets, indexes)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(acc, epoch, net)

    state = {
        'current_net': net,
    }
    torch.save(state, f'checkpoint/{args.sess}/current_net.pth')
    return test_loss / batch_idx, 100. * correct / total


def save_checkpoint(acc, epoch, net):
    # Save checkpoint.
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, f'checkpoint/{args.sess}/best.pth')


if __name__ == '__main__':
    main()