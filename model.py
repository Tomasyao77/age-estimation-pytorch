import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import torch
import torchvision.models as models

def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    # se_resnext50_32x4d
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)

    # model.load_state_dict(torch.load(model_path)) !
    # model = torch.load("/home/zouy/.cache/torch/checkpoints/se_resnext50_32x4d-a260b3a4.pth")

    # vgg16
    # model = models.vgg16(pretrained=True)
    # num_fc = model.classifier[6].in_features
    # model.classifier[6] = torch.nn.Linear(num_fc, num_classes)


    return model


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
