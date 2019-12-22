from easydict import EasyDict as edict
import sys

# sys.path.append(".")
# sys.path.append("..")
cfg = edict()

cfg.BASE = "/media/zouy/workspace/gitcloneroot/age-estimation-pytorch"  # 项目根目录
cfg.dlib68dat = cfg.BASE + "/util/mydlib/shape_predictor_68_face_landmarks.dat"
