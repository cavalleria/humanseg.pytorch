import torch
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

def ce_loss(logits, targets):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	targets = targets.type(torch.int64)
	ce_loss = F.cross_entropy(logits, targets)
	return ce_loss

def dice_loss(logits, targets, smooth=1.0):
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	outputs = F.softmax(logits, dim=1)
	targets = torch.unsqueeze(targets, dim=1)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.tensor(1.0))
	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(2,3)) + smooth) / (outputs.sum(dim=(2,3))+targets.sum(dim=(2,3)) + smooth))
	return dice.mean()

def dice_loss_with_sigmoid(sigmoid, targets, smooth=1.0):
	"""
	sigmoid: (torch.float32)  shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
	outputs = torch.squeeze(sigmoid, dim=1)

	inter = outputs * targets
	dice = 1 - ((2*inter.sum(dim=(1,2)) + smooth) / (outputs.sum(dim=(1,2))+targets.sum(dim=(1,2)) + smooth))
	dice = dice.mean()
	return dice

def bce_loss(logits, targets):
	"""
	logits: (torch.float32)  shape (N, C, H, W)  (16, 2, 160, 160)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}  (16, 160, 160)
	"""
	targets = torch.unsqueeze(targets, dim=1)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets.type(torch.int64), src=torch.tensor(1.0))
	loss = F.binary_cross_entropy_with_logits(logits, targets)
	return loss

#==============================lovasz loss==================================
# adapted from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax(logits, targets, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(logits, targets))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(logits, targets, ignore), classes=classes)
    return loss

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def isnan(x):
    return x != x
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
#==============================lovasz loss==================================

def custom_bisenet_loss(logits, targets):
	"""
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		main_loss = ce_loss(logits[0], targets)
		os16_loss = ce_loss(logits[1], targets)
		os32_loss = ce_loss(logits[2], targets)
		return main_loss + os16_loss + os32_loss
	else:
		return ce_loss(logits, targets)

def custom_pspnet_loss(logits, targets, alpha=0.4):
	"""
	logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		with torch.no_grad():
			_targets = torch.unsqueeze(targets, dim=1)
			aux_targets = F.interpolate(_targets, size=logits[1].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]

		main_loss = ce_loss(logits[0], targets)
		aux_loss = ce_loss(logits[1], aux_targets)
		return main_loss + alpha*aux_loss
	else:
		return ce_loss(logits, targets)

def custom_icnet_loss(logits, targets, alpha=[0.4, 0.16]):
	"""
	logits: (torch.float32)
		[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
						(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

		[valid_mode] x_124_cls of shape (N, C, H, W)

	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		with torch.no_grad():
			targets = torch.unsqueeze(targets, dim=1)
			target1 = F.interpolate(targets, size=logits[0].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]
			target2 = F.interpolate(targets, size=logits[1].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]
			target3 = F.interpolate(targets, size=logits[2].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]

		loss1 = ce_loss(logits[0], target1)
		loss2 = ce_loss(logits[1], target2)
		loss3 = ce_loss(logits[2], target3)
		return loss1 + alpha[0]*loss2 + alpha[1]*loss3

	else:
		return ce_loss(logits, targets)