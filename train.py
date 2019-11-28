import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from model import my_model
from model import MyLoss_l1
from dataset import FaceDataset_morph2
from defaults import _C as cfg
import os
import smtp
import time


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard logs directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device, l1loss=0.0):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    criterion_l1 = MyLoss_l1()

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)

            # compute output
            outputs, ouput1val = model(x)
            # outputs = model(x)
            # print(outputs)  # 2*numclasses
            # print(ouput1val)  # 2*batchsize

            # calc loss
            # loss再加一个mae损失
            # print("criterion(outputs, y):")
            # print(criterion(outputs, y)) #tensor(4.6222, device='cuda:0', grad_fn=<NllLossBackward>)
            # print("criterion_l1(ouput1val, y.float()):")
            # print(criterion_l1(ouput1val, y.float()))  # Long
            loss = criterion(outputs, y) + criterion_l1(ouput1val, y.float()) * l1loss
            cur_loss = loss.item()

            # calc accuracy
            _, predicted = outputs.max(1)  # 返回每一行最大值组成的一维数组
            correct_num = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device, l1loss=0.0):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    preds_1val = []
    gt = []  # ground truth

    criterion_l1 = MyLoss_l1()

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                # compute output
                outputs, ouput1val = model(x)
                # outputs = model(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())  # ? * 101 ? 方便后面求期望
                preds_1val.append(ouput1val.cpu().numpy())
                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y) + criterion_l1(ouput1val, y.float()) * l1loss
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)  # 展开
    preds_1val = np.concatenate(preds_1val, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)  # softmax后求期望,得出预测的年龄 DEX!
    ave_preds = (preds * ages).sum(axis=-1)  # axis=0结果一样?
    # 分类和回归的结果取平均
    # ave_preds = (ave_preds + preds_1val) / 2.0
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def main(mydict):
    print("开始训练时间：")
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(start_time)
    # py脚本额外参数
    args = get_args()
    # main函数传入参数
    my_data_dir = mydict["data_dir"]
    my_tensorboard = mydict["tensorboard"]
    my_checkpoint = mydict["checkpoint"]
    my_ifSE = mydict["ifSE"]
    my_l1loss = mydict["l1loss"]
    if my_l1loss:
        l1loss = 0.1
    else:
        l1loss = 0.0

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0

    # checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir = Path(my_checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model_dir
    print("=> creating model_dir '{}'".format(cfg.MODEL.ARCH))
    # model_dir = get_model(model_name=cfg.MODEL.ARCH)
    model = my_model(my_ifSE)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume
    if resume_path:
        print(Path(resume_path).is_file())
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    # 损失计算准则
    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset_morph2(my_data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                       age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset_morph2(my_data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None
    val_mae_list = []

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=my_tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=my_tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device, l1loss)

        # validate
        val_loss, val_acc, val_mae = validate(val_loader, model, criterion, epoch, device, l1loss)
        val_mae_list.append(val_mae)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            best_val_mae = val_mae
            # checkpoint
            if val_mae < 3.2:
                model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'arch': cfg.MODEL.ARCH,
                        'state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict()
                    },
                    str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
                )
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")
    print("结束训练时间：")
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(end_time)
    print("训练耗时: " + smtp.date_gap(start_time, end_time))
    # 发邮件
    smtp.main(dict_={"共训练epochs: ": cfg.TRAIN.EPOCHS,
                     "训练耗时: ": smtp.date_gap(start_time, end_time),
                     "最低val_mae: ": best_val_mae,
                     "平均val_mae: ": np.array(val_mae_list).mean(),
                     "MODEL.IMG_SIZE: ": cfg.MODEL.IMG_SIZE,
                     "BATCH_SIZE: ": cfg.BATCH_SIZE,
                     "LOSS.l1: ": l1loss,
                     "TRAIN.LR: ": cfg.TRAIN.LR,
                     "TRAIN.LR_DECAY_STEP: ": cfg.TRAIN.LR_DECAY_STEP,
                     "TRAIN.LR_DECAY_RATE:": cfg.TRAIN.LR_DECAY_RATE,
                     "TRAIN.OPT: ": cfg.TRAIN.OPT,
                     "MODEL.ARCH:": cfg.MODEL.ARCH})
    return best_val_mae


if __name__ == '__main__':
    # morph2和morph2_align数据集训练 各训练4次共8次
    start_time = smtp.print_time("全部开始训练!!!")
    # get_args()
    tf_log = cfg.tf_log[0]
    ckpt = cfg.ckpt[0]
    data_dir = {"morph2": cfg.dataset.morph2, "morph2_align": cfg.dataset.morph2_align}

    main({"data_dir": data_dir["morph2"], "tensorboard": tf_log["morph2"], "checkpoint": ckpt["morph2"],
          "ifSE": False, "l1loss": False})
    time.sleep(600)  # sleep 10 min
    main({"data_dir": data_dir["morph2"], "tensorboard": tf_log["morph2_l1"], "checkpoint": ckpt["morph2_l1"],
          "ifSE": False, "l1loss": True})
    time.sleep(600)  # sleep 10 min
    main({"data_dir": data_dir["morph2"], "tensorboard": tf_log["morph2_sfv2"], "checkpoint": ckpt["morph2_sfv2"],
          "ifSE": True, "l1loss": False})
    time.sleep(600)  # sleep 10 min
    main(
        {"data_dir": data_dir["morph2"], "tensorboard": tf_log["morph2_sfv2_l1"], "checkpoint": ckpt["morph2_sfv2_l1"],
         "ifSE": True, "l1loss": True})
    time.sleep(600)  # sleep 10 min

    main({"data_dir": data_dir["morph2_align"], "tensorboard": tf_log["morph2_align"],
          "checkpoint": ckpt["morph2_align"], "ifSE": False, "l1loss": False})
    time.sleep(600)  # sleep 10 min
    main({"data_dir": data_dir["morph2_align"], "tensorboard": tf_log["morph2_align_l1"],
          "checkpoint": ckpt["morph2_align_l1"], "ifSE": False, "l1loss": True})
    time.sleep(600)  # sleep 10 min
    main({"data_dir": data_dir["morph2_align"], "tensorboard": tf_log["morph2_align_sfv2"],
          "checkpoint": ckpt["morph2_align_sfv2"], "ifSE": True, "l1loss": False})
    time.sleep(600)  # sleep 10 min
    main({"data_dir": data_dir["morph2_align"], "tensorboard": tf_log["morph2_align_sfv2_l1"],
          "checkpoint": ckpt["morph2_align_sfv2_l1"], "ifSE": True, "l1loss": True})

    end_time = smtp.print_time("全部训练结束!!!")
    print(smtp.date_gap(start_time, end_time))
    smtp.main(dict_={"morph2全部训练耗时: ": smtp.date_gap(start_time, end_time)})
