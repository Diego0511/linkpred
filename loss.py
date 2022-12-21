import torch
import torch.nn as nn
import torch.nn.functional as F


def auc_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()


def hinge_auc_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()


def weighted_auc_loss(pos_out, neg_out, num_neg, weight):
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (weight*torch.square(1 - (pos_out - neg_out))).sum()


def adaptive_auc_loss(pos_out, neg_out, num_neg, margin):
    margin = torch.reshape(margin, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(margin - (pos_out - neg_out))).sum()


def weighted_hinge_auc_loss(pos_out, neg_out, num_neg, weight):
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (weight*torch.square(torch.clamp(weight - (pos_out - neg_out), min=0))).sum()


def adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, weight):
    weight = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(torch.clamp(weight - (pos_out - neg_out), min=0))).sum()


def log_rank_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()


def ce_loss(pos_out, neg_out):
    pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
    return pos_loss + neg_loss


def info_nce_loss(pos_out, neg_out, num_neg):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, pos_out, neg_out):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
def focal_loss(pos_out, neg_out, alpha=0.25, gamma=2, reduce=True):
    def cal_loss(inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)**gamma * BCE_loss
        if reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
    pos_loss = cal_loss(pos_out, torch.ones_like(pos_out))
    neg_loss = cal_loss(neg_out, torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def adaptive_hinge_loss(pos_out, neg_out, num_neg, weight):
    margin = torch.reshape(weight, (-1, 1))
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return (torch.square(torch.clamp(weight - (pos_out - neg_out), min=0))).sum()
    