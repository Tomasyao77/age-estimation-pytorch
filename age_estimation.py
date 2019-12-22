import argparse
from pathlib import Path
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import pretrainedmodels
import pretrainedmodels.utils
from model import my_model
from defaults import _C as cfg
from train import validate_age_estimation
import os
import smtp


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img_path", type=str, required=True, help="img path to be predicted")
    parser.add_argument("--my_resume", type=str, required=True, help="Model weight to be tested")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main():
    start_time = smtp.print_time("开始测试!!!")
    # py脚本额外参数
    args = get_args()
    # main函数传入参数
    img_path = args.img_path
    my_resume = args.my_resume

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # create model_dir
    print("=> creating model_dir '{}'".format(cfg.MODEL.ARCH))
    model = my_model()
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

    print("=> start testing")
    # img_path, img_size, model, device
    predict_age = validate_age_estimation(img_path, cfg.MODEL.IMG_SIZE, model, device)
    print(f"predict_age: {predict_age:.2f}")
    end_time = smtp.print_time("测试结束!!!")
    print(smtp.date_gap_abs(start_time, end_time))
    return int(round(predict_age))


if __name__ == '__main__':
    print(main())
    #
    # f = os.listdir("/media/zouy/workspace/gitcloneroot/age-estimation-pytorch/data_dir/morph2-align/morph2_align")
    # img_path = f[int(f.__len__() * 0.9):]
    # print(img_path)
