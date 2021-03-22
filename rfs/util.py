import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight,
                                              size_average=size_average,
                                              reduce=reduce,
                                              reduction=reduction,
                                              pos_weight=pos_weight)

    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)


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


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, no_norm=False, use_attention=False):
        super(OrthogonalProjectionLoss, self).__init__()
        self.no_norm = no_norm
        self.use_attention = use_attention

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if self.use_attention:
            features_weights = torch.matmul(features, features.T)
            features_weights = F.softmax(features_weights, dim=1)
            features = torch.matmul(features_weights, features)

        #  features are normalized
        if not self.no_norm:
            features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (2.0 * neg_pairs_mean)

        return loss


class ClassCentreSimilarity(nn.Module):
    def __init__(self, centres):
        self.class_centres = centres
        super(ClassCentreSimilarity, self).__init__()

    def forward(self, features, labels=None):
        with torch.no_grad():
            adjusted_centres = torch.index_select(self.class_centres, dim=0, index=labels)
            # adjusted_centres = torch.stack([self.class_centres[x] for x in labels], dim=0)
        return (adjusted_centres * features).sum()


class GuidedComplementEntropy(nn.Module):

    def __init__(self, alpha, classes):
        super(GuidedComplementEntropy, self).__init__()
        self.alpha = alpha
        self.classes = classes
        self.batch_size = None

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, y_hat, y):
        self.batch_size = len(y)
        y_hat = F.softmax(y_hat, dim=1)
        y_g = torch.gather(y_hat, 1, torch.unsqueeze(y, 1))
        y_g_ = (1 - y_g) + 1e-7  # avoiding numerical issues (first)
        # avoiding numerical issues (second)
        guided_factor = (y_g + 1e-7) ** self.alpha
        px = y_hat / y_g_.view(len(y_hat), 1)
        px_log = torch.log(px + 1e-10)  # avoiding numerical issues (third)
        y_zero_hot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = px * px_log * y_zero_hot.cuda()
        guided_output = guided_factor.squeeze() * torch.sum(output, dim=1)
        loss = torch.sum(guided_output)
        loss /= float(self.batch_size)
        loss /= np.log(float(self.classes))
        return loss


def soft_representation_loss(features, labels, srl_weights):  # TODO: improve this @kahnchana
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
    features = F.normalize(features, p=2, dim=1)

    loss = torch.zeros(1, requires_grad=True).to(device)
    for idx, cls_i in enumerate(labels):
        for jdx, cls_j in enumerate(labels):
            loss = loss + torch.abs(features[idx].T @ features[jdx] - srl_weights[cls_i, cls_j])
    # torch.index_select(
    # torch.index_select(srl_weights, torch.tensor(0), torch.tensor(cls_i)), torch.tensor(1), torch.tensor(cls_j))[0])
    return loss


class PerpetualOrthogonalProjectionLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2048, no_norm=False, use_attention=False, use_gpu=True):
        super(PerpetualOrthogonalProjectionLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.no_norm = no_norm
        self.use_attention = use_attention

        if self.use_gpu:
            self.class_centres = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.class_centres = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        if self.use_attention:
            features_weights = torch.matmul(features, features.T)
            features_weights = F.softmax(features_weights, dim=1)
            features = torch.matmul(features_weights, features)

        #  features are normalized
        if not self.no_norm:
            features = F.normalize(features, p=2, dim=1)
        normalized_class_centres = F.normalize(self.class_centres, p=2, dim=1)

        labels = labels[:, None]  # extend dim
        class_range = torch.arange(self.num_classes, device=device).long()
        class_range = class_range[:, None]  # extend dim
        label_mask = torch.eq(labels, class_range.t()).float().to(device)
        feature_centre_variance = torch.matmul(features, normalized_class_centres.t())
        same_class_loss = (label_mask * feature_centre_variance).sum() / (label_mask.sum() + 1e-6)
        diff_class_loss = ((1 - label_mask) * feature_centre_variance).sum() / ((1 - label_mask).sum() + 1e-6)

        loss = 0.5 * (1.0 - same_class_loss) + torch.abs(diff_class_loss)

        return loss