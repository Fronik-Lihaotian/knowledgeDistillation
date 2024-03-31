import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import models
import argparse


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


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
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),  # keep size as 224*224
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'),
                                         transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'),
                                       transform=data_transform['val'])
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    # teacher training
    teacher_logger = get_logger('./logfile/explog_teacher_network_{}epochs_on_{}'.format(args.t_epochs, args.dataset))
    teacher_model = models.Teacher(num_class=args.num_classes).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=teacher_model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer)
    best_acc = 0
    for epoch in range(args.t_epochs):
        loss_sum = 0.0
        teacher_model.train()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = teacher_model(images.to(device))
            loss_value = loss(logits, labels.to(device))
            loss_value.backward()
            optimizer.step()
            loss_sum += loss_value.item()
            train_bar.desc = "training epoch[{}/{}], loss: {}, lr: {}".format(epoch, args.t_epochs, loss_value,
                                                                              optimizer.param_groups[0]['lr'])
        teacher_model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_logits = teacher_model(val_images.to(device))
                predict = torch.max(val_logits, dim=1)[1]  # torch.max()[1] returns the index of max value in tensor
                acc += torch.eq(predict, val_labels.to(device)).sum().item()  # item seems more precise
                val_bar.desc = 'validate epoch[{}/{}]'.format(epoch, args.t_epochs)

        acc = acc / val_num
        average_loss = loss_sum / train_num
        print('epoch[{}/{}]: val_acc: {}, average_loss: {}, lr: {}'.format(epoch, args.t_epochs, acc, average_loss,
                                                                           optimizer.param_groups[0]['lr']))
        teacher_logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}\t lr={:.5f}'.format(epoch, args.t_epochs,
                                                                                          average_loss,
                                                                                          acc,
                                                                                          optimizer.param_groups[0][
                                                                                              'lr']))
        if acc >= best_acc:
            best_acc = acc
            torch.save(teacher_model.state_dict(), args.teacher_path)

    # student stage
    student_logger = get_logger('./logfile/explog_student_network_{}epochs_on_{}'.format(args.t_epochs, args.dataset))
    student_model = models.Student(num_class=args.num_classes_s).to(device)
    kd_loss = nn.KLDivLoss(reduction='batchmean')
    hard_loss = nn.CrossEntropyLoss()
    student_optimizer = torch.optim.AdamW(params=student_model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer)
    student_best_acc = 0.0
    teacher_model.eval()
    for epoch in range(args.s_epochs):
        student_model.train()
        kd_loss_sum = 0.0
        hard_loss_sum = 0.0
        train_bar_student = tqdm(train_loader)
        for step, data in enumerate(train_bar_student):
            images, labels = data
            student_optimizer.zero_grad()
            with torch.no_grad():
                out_t = teacher_model(images.to(device))
            out_s = student_model(images.to(device))
            hard_loss_value = hard_loss(out_s, labels.to(device))
            kd_loss_value = kd_loss(F.log_softmax(out_s / args.kl_tmp, dim=1), F.softmax(out_t / args.kl_tmp, dim=1))
            total_loss = args.kl_alpha * hard_loss_value + (1 - args.kl_alpha) * kd_loss_value
            total_loss.backward()
            student_optimizer.step()
            hard_loss_sum += hard_loss_value
            kd_loss_sum += kd_loss_value
            train_bar_student.desc = "student training epoch[{}/{}], loss: {}, kd_loss: {}, hard_loss{}, lr: {}".format(
                epoch,
                args.t_epochs,
                total_loss,
                kd_loss_value,
                hard_loss_value,
                optimizer.param_groups[0]['lr'])

        student_model.eval()
        student_acc = 0.0
        with torch.no_grad():
            val_bar_student = tqdm(val_loader)
            for val_images_s, val_labels_s in val_bar_student:
                val_out_s = student_model(val_images_s.to(device))
                predict_s = torch.max(val_out_s, dim=1)[1]
                student_acc += torch.eq(predict_s, val_labels_s.to(device)).sum().item()  # item seems more precise
                val_bar.desc = 'validate epoch[{}/{}]'.format(epoch, args.s_epochs)

        student_acc = student_acc / val_num
        average_loss = (kd_loss_sum + hard_loss_sum) / train_num
        avg_hard_loss = hard_loss_sum / train_num
        avg_kd_loss = kd_loss_sum / train_num
        print('epoch[{}/{}]: val_acc: {}, avg_loss: {}, avg_kd_loss: {}, avg_hard_loss: {}, lr: {}'.format(epoch,
                                                                                                           args.s_epochs,
                                                                                                           student_acc,
                                                                                                           average_loss,
                                                                                                           avg_kd_loss,
                                                                                                           avg_hard_loss,
                                                                                                           optimizer.param_groups[
                                                                                                               0][
                                                                                                               'lr']))
        student_logger.info('Epoch:[{}/{}]\t loss={:.5f}\t kd_loss: {}\t hard_loss{}\t acc={:.3f}\t lr={:.5f}'.format(
            epoch,
            args.t_epochs,
            average_loss,
            avg_kd_loss,
            avg_hard_loss,
            student_acc,
            optimizer.param_groups[0][
                'lr']))

        if student_acc >= student_best_acc:
            student_best_acc = student_acc
            torch.save(student_model.state_dict(), args.student_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=257)
    parser.add_argument('--num_classes_s', type=int, default=257)
    parser.add_argument('--t_epochs', type=int, default=300)
    parser.add_argument('--s_epochs', type=int, default=120)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--data_path', type=str, default='E:/学习/去雾/data/dataset')
    parser.add_argument('--dataset', type=str, default='caltech-256')
    parser.add_argument('--teacher_path', type=str, default='./teacher_weights/MobileNetv2.pth')
    parser.add_argument('--student_path', type=str, default='./student_weights/MobileNets.pth')
    parser.add_argument('--kl_tmp', type=float, default=5.)
    parser.add_argument('--kl_alpha', type=float, default=.3)

    opt = parser.parse_args()
    main(opt)
