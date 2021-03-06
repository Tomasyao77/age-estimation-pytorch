from yacs.config import CfgNode as CN

_C = CN()

# Model
_C.MODEL = CN()
# se_resnext50_32x4d 19G
# ShuffleNetV2 4G
# resnet18_cbam 6G
_C.MODEL.ARCH = "ShuffleNetV2"  # check python train.py -h for available models
_C.MODEL.IMG_SIZE = 224
# base
_C.BATCH_SIZE = 128
_C.BASE = "/media/d9lab/data11/tomasyao/workspace/pycharm_ws/age-estimation-pytorch"  # 项目根目录
# _C.BASE = "/media/zouy/workspace/gitcloneroot/age-estimation-pytorch"  # 项目根目录
_C.DATASET = _C.BASE + "/data_dir"
_C.TF_LOG = _C.BASE + "/tf_log"
#symmetry1 sfv2 ma_sfv2没有aug align_sfv2 align_ma_sfv2有aug
#symmetry2 都有aug #symmetry2/morph2_align_sfv2_l1奇怪(因为只训练到40epoch就停掉了哈哈)
#symmetry3 都没有aug 40epoch 因为好像过拟合了
#symmetry4 只训练sfv2_l1 sfv2_ma_l1 align_sfv2_l1 align_sfv2_ma_l1,都有aug因为不用就过拟合,40epoch之后都保存
_C.TF_LOG_l1 = _C.BASE + "/tf_log_l1"
_C.TF_LOG_decay = _C.BASE + "/tf_log_decay"
_C.TF_LOG_decay_step = _C.BASE + "/tf_log_step"
_C.checkpoint = _C.BASE + "/checkpoint"

# dataset
_C.dataset = CN()
# morph2
_C.dataset.morph2 = _C.DATASET + "/morph2"
_C.dataset.morph2_align = _C.DATASET + "/morph2_align"
# FG-NET
_C.dataset.fgnet_leave1out = _C.DATASET + "/FG-NET-leave1out"
_C.dataset.fgnet_align_leave1out = _C.DATASET + "/FG-NET_align-leave1out"
# ceface
_C.dataset.ce_base = "/media/d9lab/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/CE_FACE"
_C.dataset.ceface_align = _C.dataset.ce_base + "/CE_FACE_align_224"
_C.dataset.ceface_align_tflog = _C.TF_LOG + "/ceface/2020-1-2"
_C.dataset.ceface_align_ckpt = _C.checkpoint + "/ceface/2020-1-2"

# tf_log
_C.tf_log = [{"morph2": _C.TF_LOG + "/morph2", "morph2_align": _C.TF_LOG + "/morph2_align",
              "morph2_l1": _C.TF_LOG + "/morph2_l1", "morph2_align_l1": _C.TF_LOG + "/morph2_align_l1",
              "morph2_sfv2": _C.TF_LOG + "/morph2_sfv2", "morph2_align_sfv2": _C.TF_LOG + "/morph2_align_sfv2",
              "morph2_sfv2_l1": _C.TF_LOG + "/morph2_sfv2_l1",
              "morph2_align_sfv2_l1": _C.TF_LOG + "/morph2_align_sfv2_l1"}]
# checkpoint
_C.ckpt = [{"morph2": _C.checkpoint + "/morph2", "morph2_align": _C.checkpoint + "/morph2_align",
            "morph2_l1": _C.checkpoint + "/morph2_l1", "morph2_align_l1": _C.checkpoint + "/morph2_align_l1",
            "morph2_sfv2": _C.checkpoint + "/morph2_sfv2", "morph2_align_sfv2": _C.checkpoint + "/morph2_align_sfv2",
            "morph2_sfv2_l1": _C.checkpoint + "/morph2_sfv2_l1",
            "morph2_align_sfv2_l1": _C.checkpoint + "/morph2_align_sfv2_l1"}]

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "adam"  # adam or sgd
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_STEP = 20  # 20
_C.TRAIN.LR_DECAY_RATE = 0.2  # 0.2
_C.TRAIN.EPOCHS = 80  # senet: 70(morph2) 20足以(fgnet) 20足以(fgnet_align)
# sgd
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.AGE_STDDEV = 1.0  # 年龄标准差

# Test
_C.TEST = CN()
_C.TEST.logs = _C.BASE + "/test_logs"

# loss
_C.LOSS = CN()
_C.LOSS.l1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
