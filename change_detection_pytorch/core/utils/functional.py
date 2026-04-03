import numpy as np
import torch


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _fast_hist(label_gt, label_pred, num_classes):
    """Compute confusion matrix for a single sample."""
    mask = (label_gt >= 0) & (label_gt < num_classes)
    hist = np.bincount(
        num_classes * label_gt[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """Compute confusion matrix for a batch.
    Args:
        num_classes: number of classes
        label_gts: list or array of ground truth labels (H, W) each
        label_preds: list or array of predicted labels (H, W) each
    Returns:
        confusion_matrix: (num_classes, num_classes) confusion matrix
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    return confusion_matrix


def cm2score(confusion_matrix):
    """Calculate scores from confusion matrix (like CASP metric_tool.py).
    Args:
        confusion_matrix: (num_classes, num_classes) confusion matrix
    Returns:
        dict: scores including acc, miou, mf1, per-class metrics
    """
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # Accuracy
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # Recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)

    # Precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    # IoU
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    # Frequency weighted accuracy
    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    cls_iou = dict(zip([f'iou_{i}' for i in range(n_class)], iu))
    cls_precision = dict(zip([f'precision_{i}' for i in range(n_class)], precision))
    cls_recall = dict(zip([f'recall_{i}' for i in range(n_class)], recall))
    cls_F1 = dict(zip([f'F1_{i}' for i in range(n_class)], F1))

    score_dict = {
        'acc': acc,
        'miou': mean_iu,
        'mf1': mean_F1,
        'fwavacc': fwavacc
    }
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp_tn = torch.sum(gt == pr, dtype=pr.dtype)

    score = tp_tn / gt.view(-1).shape[0]

    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


def kappa(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate kappa score between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: kappa score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    tn = torch.sum((1 - gt)*(1 - pr))

    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (N * N)

    score = (p0 - pe) / (1 - pe)

    return score


def dice(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate dice score between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: dice score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    _precision = precision(pr, gt, eps=eps, threshold=threshold, ignore_channels=ignore_channels)
    _recall = recall(pr, gt, eps=eps, threshold=threshold, ignore_channels=ignore_channels)

    score = 2 * _precision * _recall / (_precision + _recall)

    return score