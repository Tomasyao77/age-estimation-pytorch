import argparse
import better_exceptions
from pathlib import Path
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from model import my_model
from dataset import FaceDataset_FGNET
from dataset import FaceDataset_morph2
from defaults import _C as cfg
from train import validate
from train import validate_cs
import shutil
import os
import smtp
import sys
from tqdm import tqdm


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    # parser.add_argument("--resume", type=str, required=True, help="Model weight to be tested")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main(mydict):
    # py脚本额外参数
    args = get_args()
    # main函数传入参数
    my_data_dir = mydict["data_dir"]
    my_ifSE = mydict["ifSE"]
    my_l1loss = mydict["l1loss"]
    my_resume = mydict["resume"]
    if my_l1loss:
        l1loss = 0.1
    else:
        l1loss = 0.0

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # create model_dir
    print("=> creating model_dir '{}'".format(cfg.MODEL.ARCH))
    # model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    model = my_model(my_ifSE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = my_resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=>ckpt loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoTrueint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    test_dataset = FaceDataset_morph2(my_data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    print("=> start testing")
    _, _, test_mae = validate(test_loader, model, None, 0, device, l1loss)
    print(f"test mae: {test_mae:.3f}")
    return test_mae

def main_cs(mydict):
    # py脚本额外参数
    args = get_args()
    # main函数传入参数
    my_data_dir = mydict["data_dir"]
    my_ifSE = mydict["ifSE"]
    my_l1loss = mydict["l1loss"]
    my_resume = mydict["resume"]
    if my_l1loss:
        l1loss = 0.1
    else:
        l1loss = 0.0

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # create model_dir
    print("=> creating model_dir '{}'".format(cfg.MODEL.ARCH))
    # model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    model = my_model(my_ifSE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = my_resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=>ckpt loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoTrueint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    test_dataset = FaceDataset_morph2(my_data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    print("=> start testing")
    _, _, test_cs = validate_cs(test_loader, model, None, 0, device, l1loss)
    print(f"test cs list: {test_cs}")
    return test_cs

#自定义格式化输出 方便复用
def test_mae_morph2_print(test_mae_morph2, modelname):
    print(f"{modelname}_test_mae_morph2:{test_mae_morph2}")
    test_mae_morph2.sort()
    print(f"{modelname}_min_test_mae_morph2:{test_mae_morph2[0]}")

def log_refine():
    f = open("./logs/20191203_143049_morph2_all_test_log", "r")
    w = open("./logs/20191203_143049_morph2_all_test_log1", "a")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        # if 'test_mae_morph2:[' in line:
        if 'min_test_mae_morph2' in line:
            w.write(line)
            # print(line)

    f.close()
    w.close()

def testall():
    start_time = smtp.print_time("全部开始测试!!!")
    tf_log = cfg.tf_log[0]
    ckpt = cfg.ckpt[0]
    data_dir = {"morph2": cfg.dataset.morph2, "morph2_align": cfg.dataset.morph2_align}
    test_mae_morph2 = []
    ###########################################################################################################
    ##################morph2##################
    ckpt_morph2 = os.listdir(ckpt["morph2"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2"], "ifSE": False, "l1loss": False,
                  "resume": ckpt["morph2"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2")
    ##################morph2_l1##################
    ckpt_morph2 = os.listdir(ckpt["morph2_l1"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2"], "ifSE": False, "l1loss": True,
                  "resume": ckpt["morph2_l1"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2_l1")
    ##################morph2_sfv2##################
    ckpt_morph2 = os.listdir(ckpt["morph2_sfv2"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2"], "ifSE": True, "l1loss": False,
                  "resume": ckpt["morph2_sfv2"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2_sfv2")
    ##################morph2_sfv2_l1##################
    ckpt_morph2 = os.listdir(ckpt["morph2_sfv2_l1"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2"], "ifSE": True, "l1loss": True,
                  "resume": ckpt["morph2_sfv2_l1"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2_sfv2_l1")
    ###########################################################################################################
    ##################morph2_align##################
    ckpt_morph2 = os.listdir(ckpt["morph2_align"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2_align"], "ifSE": False, "l1loss": False,
                  "resume": ckpt["morph2_align"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2_align")
    ##################morph2_align_l1##################
    ckpt_morph2 = os.listdir(ckpt["morph2_align_l1"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2_align"], "ifSE": False, "l1loss": True,
                  "resume": ckpt["morph2_align_l1"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2_align_l1")
    ##################morph2_align_sfv2##################
    ckpt_morph2 = os.listdir(ckpt["morph2_align_sfv2"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2_align"], "ifSE": True, "l1loss": False,
                  "resume": ckpt["morph2_align_sfv2"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2_align_sfv2")
    ##################morph2_align_sfv2_l1##################
    ckpt_morph2 = os.listdir(ckpt["morph2_align_sfv2_l1"])
    ckpt_morph2.sort()
    for name in ckpt_morph2:
        test_mae_morph2.append(
            main({"data_dir": data_dir["morph2_align"], "ifSE": True, "l1loss": True,
                  "resume": ckpt["morph2_align_sfv2_l1"] + "/" + name}))
    test_mae_morph2_print(test_mae_morph2, "morph2_align_sfv2_l1")
    ###########################################################################################################
    end_time = smtp.print_time("全部测试结束!!!共耗时:")
    print(smtp.date_gap(start_time, end_time))

def test_single():
    start_time = smtp.print_time("全部开始测试!!!")
    tf_log = cfg.tf_log[0]
    ckpt = cfg.ckpt[0]
    data_dir = {"morph2": cfg.dataset.morph2, "morph2_align": cfg.dataset.morph2_align}
    test_mae_morph2 = []
    ###########################################################################################################
    ##################morph2_align_sfv2##################
    ckpt_morph2 = os.listdir(ckpt["morph2_align_sfv2"])
    ckpt_morph2.sort()
    name = "epoch079_0.02094_2.6708.pth"
    main({"data_dir": data_dir["morph2_align"], "ifSE": True, "l1loss": False,
          "resume": ckpt["morph2_align_sfv2"] + "/" + name})
    # test_mae_morph2_print(test_mae_morph2, "morph2_align_sfv2")
    ###########################################################################################################
    end_time = smtp.print_time("全部测试结束!!!共耗时:")
    print(smtp.date_gap(start_time, end_time))

def test_cs_curve():
    ##################morph2_align_sfv2_l1##################
    ckpt = cfg.ckpt[0]
    data_dir = {"morph2": cfg.dataset.morph2, "morph2_align": cfg.dataset.morph2_align}
    ckpt_morph2 = "epoch079_0.02798_2.6244.pth"
    main_cs({"data_dir": data_dir["morph2_align"], "ifSE": True, "l1loss": True,
             "resume": ckpt["morph2_align_sfv2_l1"] + "/" + ckpt_morph2})

if __name__ == '__main__':
    # morph2和morph2_align数据集测试
    # get_args()

    # log_refine()
    # sys.exit(1)
    #
    # testall()
    # sys.exit(1)

    # test_cs_curve()
    # sys.exit(1)

    test_single()
    sys.exit(1)

