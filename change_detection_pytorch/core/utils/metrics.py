import numpy as np
import torch
from collections import OrderedDict

from . import base
from . import functional as F


class ConfusionMatrixMetric:
    """Metric using confusion matrix, like CASP metric_tool.py.

    Expects y_pr and y_gt as class indices (B, H, W) shape.
    """

    def __init__(self, n_classes=2, activation=None, **kwargs):
        self.n_classes = n_classes
        self.activation = activation
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, y_pr, y_gt):
        """Update confusion matrix with predictions and labels as numpy arrays (B, H, W)."""
        # If activation is set (e.g., 'argmax2d'), apply it
        if self.activation == 'argmax' or self.activation == 'argmax2d':
            y_pr = torch.argmax(y_pr, dim=1)

        y_pr = y_pr.detach().cpu().numpy()
        y_gt = y_gt.detach().cpu().numpy()

        # Iterate over batch samples
        for i in range(y_gt.shape[0]):
            gt_i = y_gt[i]
            pr_i = y_pr[i]
            batch_cm = F.get_confuse_matrix(self.n_classes, [gt_i], [pr_i])
            if self.confusion_matrix is None:
                self.confusion_matrix = batch_cm
            else:
                self.confusion_matrix += batch_cm

    def __call__(self, y_pr, y_gt):
        if self.confusion_matrix is None:
            self.reset()
        self.update(y_pr, y_gt)
        return torch.tensor(self.compute()[self.__name__]).detach()


class IoU(ConfusionMatrixMetric):
    __name__ = 'iou_score'

    def compute(self):
        if self.confusion_matrix is None:
            return {'iou_score': 0.0}
        scores = F.cm2score(self.confusion_matrix)
        return {'iou_score': scores['miou']}


class Fscore(ConfusionMatrixMetric):
    __name__ = 'f_score'

    def compute(self):
        if self.confusion_matrix is None:
            return {'f_score': 0.0}
        scores = F.cm2score(self.confusion_matrix)
        return {'f_score': scores['mf1']}


class Precision(ConfusionMatrixMetric):
    __name__ = 'precision'

    def compute(self):
        if self.confusion_matrix is None:
            return {'precision': 0.0}
        scores = F.cm2score(self.confusion_matrix)
        # Mean precision across classes
        precisions = [scores[f'precision_{i}'] for i in range(self.n_classes)]
        return {'precision': np.nanmean(precisions)}


class Recall(ConfusionMatrixMetric):
    __name__ = 'recall'

    def compute(self):
        if self.confusion_matrix is None:
            return {'recall': 0.0}
        scores = F.cm2score(self.confusion_matrix)
        # Mean recall across classes
        recalls = [scores[f'recall_{i}'] for i in range(self.n_classes)]
        return {'recall': np.nanmean(recalls)}


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = activation
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        return F.accuracy(y_pr, y_gt, threshold=self.threshold, ignore_channels=self.ignore_channels)


class Dice(ConfusionMatrixMetric):
    __name__ = 'dice'

    def compute(self):
        if self.confusion_matrix is None:
            return {'dice': 0.0}
        scores = F.cm2score(self.confusion_matrix)
        dice_scores = []
        for i in range(self.n_classes):
            prec = scores[f'precision_{i}']
            rec = scores[f'recall_{i}']
            dice = 2 * prec * rec / (prec + rec + 1e-7)
            dice_scores.append(dice)
        return {'dice': np.nanmean(dice_scores)}


class Kappa(ConfusionMatrixMetric):
    __name__ = 'kappa'

    def compute(self):
        if self.confusion_matrix is None:
            return {'kappa': 0.0}
        hist = self.confusion_matrix
        n_class = hist.shape[0]
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        tn = hist.sum() - tp - fp - fn
        N = tp + tn + fp + fn
        p0 = (tp + tn) / (N + 1e-7)
        pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (N * N + 1e-7)
        kappa = (p0 - pe) / (1 - pe + 1e-7)
        return {'kappa': np.nanmean(kappa)}
