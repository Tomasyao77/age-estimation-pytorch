import torch
import torchvision.models as models
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils


def get_model(model_path, ckp_path):
    model = torch.load(model_path)
    model.load_state_dict(get_checkpoint(ckp_path)['state_dict'])
    return model


def get_checkpoint(ckp_path):
    ckp = torch.load(ckp_path, map_location="cpu")
    return ckp


if __name__ == '__main__':
    # print(get_model(
    #     '/media/zouy/workspace/gitcloneroot/age-estimation-pytorch/model_dir/se_resnext50_32x4d-a260b3a4.pth',
    #     '/media/zouy/workspace/gitcloneroot/age-estimation-pytorch/misc/epoch044_0.02343_3.9984.pth'))
    # print(get_checkpoint(
    #     '/media/zouy/workspace/gitcloneroot/age-estimation-pytorch/misc/epoch044_0.02343_3.9984.pth'))

    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)
    print(model)

    # resnet18 = models.resnet18(pretrained=True)
    # vgg16 = models.vgg16(pretrained=True)
    # print(vgg16)
    # alexnet = models.alexnet(pretrained=True)
    # squeezenet = models.squeezenet1_0(pretrained=True)
