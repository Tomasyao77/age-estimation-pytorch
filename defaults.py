from yacs.config import CfgNode as CN

_C = CN()

# Model
_C.MODEL = CN()
# se_resnext50_32x4d 19G
# ShuffleNetV2 4G
# resnet18_cbam 6G
_C.MODEL.ARCH = "ShuffleNetV2"  # check python train.py -h for available models
_C.MODEL.IMG_SIZE = 224

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "adam"  # adam or sgd
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_STEP = 20 #20
_C.TRAIN.LR_DECAY_RATE = 0.5  # 0.2
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.BATCH_SIZE = 128  # 极客云服务器使用32，否则内存溢出
_C.TRAIN.EPOCHS = 80  # senet: 70(morph2) 20足以(fgnet) 20足以(fgnet_align)
_C.TRAIN.AGE_STDDEV = 1.0  # 年龄标准差

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
_C.TEST.BATCH_SIZE = 128

# loss
_C.LOSS = CN()
_C.LOSS.l1 = 0.1
