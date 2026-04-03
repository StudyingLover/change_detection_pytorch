#!/usr/bin/env python
"""通用训练脚本，支持 applications/ 下所有已注册的分割模型。

用法:
    python train_app.py --model casp --dataset WHU-CD --data_dir ./data
    python train_app.py --model unet --dataset LEVIR-CD --encoder_name resnet50
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from change_detection_pytorch.datasets import WHU_CD_Dataset, GZ_CD_Dataset, LEVIR_CD_Dataset
from change_detection_pytorch.core.utils.train import TrainEpoch, ValidEpoch
from change_detection_pytorch.core.utils.metrics import IoU, Fscore, Precision, Recall
from change_detection_pytorch.core.utils.logger import Logger
from change_detection_pytorch.applications import get_model, list_models


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(dataset_name, img_dir, split='train', size=256):
    dataset_name = dataset_name.upper()

    if dataset_name == 'WHU-CD' or dataset_name == 'WHU_CD':
        base_cls = WHU_CD_Dataset
    elif dataset_name == 'GZ-CD' or dataset_name == 'GZ_CD':
        base_cls = GZ_CD_Dataset
    elif dataset_name == 'LEVIR-CD' or dataset_name == 'LEVIR_CD':
        base_cls = LEVIR_CD_Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if split == 'train':
        return base_cls(img_dir=os.path.join(img_dir, 'train'),
                       ann_dir=os.path.join(img_dir, 'train', 'label'),
                       size=size)
    elif split == 'val':
        return base_cls(img_dir=os.path.join(img_dir, 'val'),
                       ann_dir=os.path.join(img_dir, 'val', 'label'),
                       size=size)
    elif split == 'test':
        return base_cls(img_dir=os.path.join(img_dir, 'test'),
                       ann_dir=os.path.join(img_dir, 'test', 'label'),
                       size=size)


def main():
    parser = argparse.ArgumentParser(description='Train Change Detection Models')
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='WHU-CD',
                        help='Dataset name: WHU-CD, GZ-CD, LEVIR-CD')
    parser.add_argument('--data_dir', type=str, default='./data/WHU-CD',
                        help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    # 模型参数
    parser.add_argument('--model', type=str, default='casp',
                        help=f'Model name (see applications/__init__.py for available models)')
    parser.add_argument('--encoder_name', type=str, default='resnet34',
                        help='Encoder name (used by models that need encoder)')
    parser.add_argument('--classes', type=int, default=2, help='Number of output classes')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained')
    # 其他
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--save_dir', type=str, default='./runs', help='Save dir')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # 初始化 Logger (创建 runs/{task}_{dataset}_{timestamp}/ 目录)
    logger = Logger(task=args.model, dataset=args.dataset, save_dir=args.save_dir)
    logger.print(f"Using device: {device}")
    logger.print(f"Model: {args.model}, Dataset: {args.dataset}")
    logger.print(f"Image size: {args.img_size}, Batch size: {args.batch_size}")
    logger.print(f"Epochs: {args.epochs}, LR: {args.lr}")

    # 加载数据集
    logger.print(f'Loading {args.dataset} dataset...')
    train_dataset = get_dataset(args.dataset, args.data_dir, split='train', size=args.img_size)
    val_dataset = get_dataset(args.dataset, args.data_dir, split='val', size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    logger.print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    # 创建模型（通过注册表自动获取）
    logger.print(f'Creating {args.model} model...')
    try:
        model = get_model(args.model, in_ch=3, pretrained=args.pretrained)
    except Exception as e:
        logger.print(f'Error creating model: {e}')
        logger.print(f'Available models: {list_models()}')
        return
    model = model.to(device)

    # 损失函数
    loss = nn.CrossEntropyLoss()

    # Metrics
    metrics = [
        IoU(threshold=0.5, activation='argmax'),
        Fscore(threshold=0.5, activation='argmax'),
        Precision(threshold=0.5, activation='argmax'),
        Recall(threshold=0.5, activation='argmax'),
    ]

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 使用 core/utils/train.py 的 TrainEpoch/ValidEpoch
    train_epoch = TrainEpoch(
        model, loss=loss, metrics=metrics,
        optimizer=optimizer, device=device, verbose=True,
    )
    valid_epoch = ValidEpoch(
        model, loss=loss, metrics=metrics,
        device=device, verbose=True,
    )

    # 恢复训练
    start_epoch = 0
    best_val_iou = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_iou = checkpoint.get('best_val_iou', 0)
            logger.print(f'Resumed from epoch {start_epoch}')

    logger.print(f'\nTraining for {args.epochs} epochs...')
    logger.print('=' * 80)

    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_logs = train_epoch.run(train_loader)

        # 验证
        valid_logs = valid_epoch.run(val_loader)

        # 更新学习率
        scheduler.step()

        # 记录到 Logger
        lr = scheduler.get_last_lr()[0]
        logger.log_epoch(epoch, train_logs, valid_logs, lr)

        # 保存模型到 run 目录
        val_iou = valid_logs['iou_score']
        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
            }, os.path.join(logger.run_path, 'best_model.pth'))
            logger.print(f'>>> Saved best model with IoU: {best_val_iou:.4f}')

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_iou': best_val_iou,
        }, os.path.join(logger.run_path, 'latest_model.pth'))

    # 训练结束，绘制曲线
    logger.finish()
    logger.print(f'\nTraining done! Best Val IoU: {best_val_iou:.4f}')


if __name__ == '__main__':
    main()
