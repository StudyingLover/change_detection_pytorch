#!/usr/bin/env python
"""Test metrics directly by copying the key functions."""

import numpy as np
import torch


def _fast_hist(label_gt, label_pred, num_classes):
    """Compute confusion matrix for a single sample."""
    mask = (label_gt >= 0) & (label_gt < num_classes)
    hist = np.bincount(
        num_classes * label_gt[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """Compute confusion matrix for a batch."""
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    return confusion_matrix


def cm2score(confusion_matrix):
    """Calculate scores from confusion matrix."""
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    return {
        'acc': acc,
        'miou': mean_iu,
        'mf1': mean_F1,
        'precision': precision.mean(),
        'recall': recall.mean()
    }


class ConfusionMatrixMetric:
    def __init__(self, n_classes=2, activation='argmax'):
        self.n_classes = n_classes
        self.activation = activation
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, y_pr, y_gt):
        # Apply activation
        if self.activation == 'argmax':
            y_pr = torch.argmax(y_pr, dim=1)
        y_pr = y_pr.detach().cpu().numpy()
        y_gt = y_gt.detach().cpu().numpy()
        batch_cm = get_confuse_matrix(self.n_classes, [y_gt], [y_pr])
        if self.confusion_matrix is None:
            self.confusion_matrix = batch_cm
        else:
            self.confusion_matrix += batch_cm

    def __call__(self, y_pr, y_gt):
        if self.confusion_matrix is None:
            self.reset()
        self.update(y_pr, y_gt)
        scores = cm2score(self.confusion_matrix)
        return torch.tensor(scores['miou'])


print("Testing confusion matrix based metrics...")
print("=" * 50)

# Shape notation: [B, C, H, W]
# y_pr is model output logits [B, C, H, W]
# y_gt is ground truth class indices [B, H, W]

# Test 1: Perfect prediction (4 pixels: 2 class 0, 2 class 1)
print("\nTest 1: Perfect prediction")
iou = ConfusionMatrixMetric(n_classes=2, activation='argmax')

# Model output: 2 classes, 1x4 image (treated as 1x1x1x4 after squeeze)
# Class 0 has high logits at positions 0,1; Class 1 at positions 2,3
y_pr = torch.tensor([[[[2.0, 2.0, -2.0, -2.0]], [[-2.0, -2.0, 2.0, 2.0]]]])  # (1, 2, 1, 4)
y_gt = torch.tensor([[[0, 0, 1, 1]]])  # (1, 1, 4) - class indices

print(f'  y_pr shape: {y_pr.shape}, y_gt shape: {y_gt.shape}')
result = iou(y_pr, y_gt)
print(f'  IoU: {result.item():.4f} (expected: 1.0)')

# Test 2: Imperfect prediction (50% of class 0 predicted wrong)
print("\nTest 2: Imperfect prediction (50% of class 0 predicted wrong)")
iou.reset()
y_pr_bad = torch.tensor([[[[2.0, -2.0, -2.0, -2.0]], [[-2.0, 2.0, 2.0, 2.0]]]])  # wrong at position 1
result_bad = iou(y_pr_bad, y_gt)
print(f'  IoU: {result_bad.item():.4f} (expected: ~0.33)')

# Test 3: All wrong for class 0
print("\nTest 3: All class 0 predicted as class 1")
iou.reset()
y_pr_allwrong = torch.tensor([[[[-2.0, -2.0, -2.0, -2.0]], [[2.0, 2.0, 2.0, 2.0]]]])  # all inverted
result_allwrong = iou(y_pr_allwrong, y_gt)
print(f'  IoU: {result_allwrong.item():.4f} (expected: 0.0 since class 0 never predicted)')

# Test 4: F1 Score
print("\nTest 4: F1 Score")
f1 = ConfusionMatrixMetric(n_classes=2, activation='argmax')
f1.reset()
result_f1 = f1(y_pr, y_gt)
print(f'  F1: {result_f1.item():.4f} (expected: 1.0)')

# Test 5: Multiple batches accumulation
print("\nTest 5: Multiple batches accumulation")
iou = ConfusionMatrixMetric(n_classes=2, activation='argmax')

# Batch 1: 50% change
y_pr_b1 = torch.tensor([[[[2.0, 2.0, -2.0, -2.0]], [[-2.0, -2.0, 2.0, 2.0]]]])
y_gt_b1 = torch.tensor([[[0, 0, 1, 1]]])

# Batch 2: 75% change
y_pr_b2 = torch.tensor([[[[2.0, 2.0, 2.0, -2.0]], [[-2.0, -2.0, -2.0, 2.0]]]])
y_gt_b2 = torch.tensor([[[0, 0, 0, 1]]])

result_b1 = iou(y_pr_b1, y_gt_b1)
result_b2 = iou(y_pr_b2, y_gt_b2)
print(f'  After batch 1: IoU = {result_b1.item():.4f}')
print(f'  After batch 2: IoU = {result_b2.item():.4f}')

# Manually compute expected IoU for combined
combined_cm = np.zeros((2, 2))
combined_cm += _fast_hist(y_gt_b1.squeeze().numpy(), torch.argmax(y_pr_b1, dim=1).squeeze().numpy(), 2)
combined_cm += _fast_hist(y_gt_b2.squeeze().numpy(), torch.argmax(y_pr_b2, dim=1).squeeze().numpy(), 2)
print(f'  Combined confusion matrix:\n{combined_cm}')
combined_scores = cm2score(combined_cm)
print(f'  Expected combined IoU: {combined_scores["miou"]:.4f}')

print("\n" + "=" * 50)
print("All tests completed!")