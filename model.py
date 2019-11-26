import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import torch
import torchvision.models as models
from model_dir.mymodel import ShuffleNetV2
from model_dir import cbam



def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    # se_resnext50_32x4d
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    print(model)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    # model_dir.add_moudule("last-value", nn.Linear(num_classes, 1))

    # model_dir.load_state_dict(torch.load(model_path)) !
    # model_dir = torch.load("/home/zouy/.cache/torch/checkpoints/se_resnext50_32x4d-a260b3a4.pth")

    # vgg16
    # model_dir = models.vgg16(pretrained=True)
    # num_fc = model_dir.classifier[6].in_features
    # model_dir.classifier[6] = torch.nn.Linear(num_fc, num_classes)


    return model


def my_model():
    # ShuffleNetV2
    return ShuffleNetV2()
    # return cbam.resnet18_cbam(num_classes=101)

def main():
    model = get_model()
    print(model.state_dict())


# loss
class MyLoss_l1(nn.Module):
    def __init__(self):
        super(MyLoss_l1, self).__init__()

    def forward(self, pred, truth):
        return torch.mean(torch.abs(pred - truth))


if __name__ == '__main__':
    main()
