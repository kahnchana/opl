import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, no_norm=False, weights_path=None, use_attention=False, gamma=2):
        super(OrthogonalProjectionLoss, self).__init__()
        self.weights_dict = None
        self.no_norm = no_norm
        self.gamma = gamma
        self.use_attention = use_attention
        if weights_path is not None:
            self.weights_dict = json.load(open(weights_path, "r"))

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

        loss = (1.0 - pos_pairs_mean) + (self.gamma * neg_pairs_mean)
        # loss = neg_pairs_mean

        return loss, pos_pairs_mean, neg_pairs_mean


class DistillationOrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(DistillationOrthogonalProjectionLoss, self).__init__()

    @staticmethod
    def forward(features, features_teacher):
        #  features are normalized
        features = F.normalize(features, p=2, dim=1)
        features_teacher = F.normalize(features_teacher, p=2, dim=1)
        # dot products calculated
        dot_prod = torch.matmul(features, features.t())
        dot_prod_teacher = torch.matmul(features_teacher, features_teacher.t())
        tau = 1
        loss = F.kl_div(
            dot_prod / tau,
            dot_prod_teacher / tau,
            reduction='sum',
            log_target=True
        ) * (tau * tau) / dot_prod_teacher.numel()

        # diff = torch.abs(dot_prod - dot_prod_teacher)
        # count = torch.ones_like(diff, device=diff.device)
        # loss = diff.sum() / count.sum()

        return loss


def covariance_loss(source, labels):
    device = (torch.device('cuda') if source.is_cuda else torch.device('cpu'))

    source = torch.div(source, source.norm(p=2, dim=1).view(source.size(0), 1))
    batch_size = source.size(0)
    classes = torch.arange(10).long()
    classes = classes.cuda()
    labels = labels.unsqueeze(1).expand(batch_size, 10)
    mask = labels.eq(classes.expand(batch_size, 10)).float()

    # Covariance of True Classes:
    xm = source
    xm_org = xm * mask
    xc_org = torch.mm(torch.t(xm_org), xm_org)
    z_org = torch.ones_like(xc_org, device=device)

    # Covariance of True Class with Other classes
    xm_other = xm * (1 - mask)
    xc_other = torch.mm(torch.t(xm_other), xm_other)
    z_others = torch.zeros_like(xc_other, device=device)

    loss = F.mse_loss(xc_org, z_org) + F.mse_loss(xc_other, z_others)
    # loss = torch.mean(xc_other-xc_org)

    return loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

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


class PerpetualOrthogonalProjectionLossV2(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2048, no_norm=False, use_attention=False, use_gpu=True):
        super(PerpetualOrthogonalProjectionLossV2, self).__init__()
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

        batch_size = labels.shape[0]
        labels = labels[:, None]  # extend dim
        class_range = torch.arange(self.num_classes, device=device).long()
        class_range = class_range[:, None]  # extend dim
        label_mask = torch.eq(labels, class_range.t()).float().to(device)
        rearranged_centres = torch.matmul(label_mask, normalized_class_centres)
        # 0.5 * |1 - dot_product(rearranged_centres,features)|
        intra_class_loss = 0.5 * torch.abs(1 - (rearranged_centres * features).sum(1)).sum() / batch_size

        eye = torch.eye(self.num_classes).float().to(device)
        class_centre_covariance = torch.matmul(normalized_class_centres, normalized_class_centres.t())
        inter_class_loss = torch.abs(class_centre_covariance - eye).sum() / (self.num_classes * self.num_classes)

        loss = intra_class_loss + inter_class_loss

        return loss


class RBFLogits(nn.Module):
    def __init__(self, feature_dim, class_num, scale, gamma):
        super(RBFLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weight = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        self.bias = nn.Parameter(torch.FloatTensor(class_num))
        self.scale = scale
        self.gamma = gamma
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, label=None):
        diff = torch.unsqueeze(self.weight, dim=0) - torch.unsqueeze(feat, dim=1)
        diff = torch.mul(diff, diff)
        metric = torch.sum(diff, dim=-1)
        kernal_metric = torch.exp(-1.0 * metric / self.gamma)
        if self.training:
            train_logits = self.scale * kernal_metric
            # ###
            # Add some codes to modify logits, e.g. margin, temperature and etc.
            # ###
            return train_logits
        else:
            test_logits = self.scale * kernal_metric
            return test_logits


class cam_loss_kd_topk(nn.Module):
    ## HNC_kd loss: distill uniform distribution to top k negative cam
    def __init__(self):
        super(cam_loss_kd_topk, self).__init__()
        print("cam_loss_kd_topk is used")

    def forward(self, x, y):
        x1 = x.clone()
        # x1 = Variable(x1)

        T = 1.0
        x = x.reshape(x.size(0), x.size(1), -1)
        b = -F.log_softmax(x / T, dim=2) / x.size(2)
        b = b.sum(2)

        x1 = x1.sum(2).sum(2)
        index = torch.zeros(x1.size())
        x1[range(x.size(0)), y] = -float("Inf")
        topk_ind = torch.topk(x1, 100, dim=1)[1]
        index[torch.tensor(range(x1.size(0))).unsqueeze(1), topk_ind] = 1
        index = index > 0.5
        # print(index.size(),index.sum())
        # ind=torch.ones(b.size())
        # ind[range(x.size(0)),y]=0
        # ind=ind>0.5
        # b=b[ind]

        index2 = x > 0
        index2[range(x.size(0)), y] = 0
        num_posi = index2.sum()
        return b[index].sum() / b.size(0), num_posi
