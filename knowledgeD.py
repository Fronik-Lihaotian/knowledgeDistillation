import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
import argparse


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    dataset_path = os.path.join(args.data_path, args.dataset)
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomErasing(),

        ]),
        'val': transforms.Compose([
            transforms.RandomCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'),
                                         transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'),
                                       transform=data_transform['val'])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)

    # teacher training
    teacher_model = models.Teacher(num_class=args.num_classes).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=teacher_model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer)
    best_acc = 0
    for epoch in range(args.t_epochs):
        teacher_model.train()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = teacher_model(images.to(device))
            loss_value = loss(logits, labels.to(device))
            loss_value.backward()
            optimizer.step()
            train_bar.desc = "training epoch[{}/{}], loss: {}, lr: {}".format(epoch, args.t_epochs, loss_value,
                                                                              optimizer.param_groups[0]['lr'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--t_epochs', type=int, default=300)
    parser.add_argument('--s_epochs', type=int, default=120)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--data_path', type=str, default='./knowledgeDistillation/dataset')
    parser.add_argument('--dataset', type=str, default='caltech-101')
    parser.add_argument('--saved_path', type=str, default='./knowledgeDistillation')

    opt = parser.parse_args()
    main(opt)
