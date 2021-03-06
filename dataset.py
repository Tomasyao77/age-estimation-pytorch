import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([  # OneOf 其中一个
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
            ]),
            iaa.Affine(  # 仿射变换
                rotate=(-10, 10),  # 旋转
                mode="edge",  # 定义填充图像外区域的方法
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},  # 缩放
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}  # 平移
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),  # 对比度
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):  # 示例可以像函数那样执行
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert (data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert (img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev  # why?

        img = cv2.imread(str(img_path), 1)  # 读彩色图
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)
        # transpose(img, (2,0,1))不好理解 clip是越界则取边界


class FaceDataset_FGNET(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert (data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        # ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        # ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            # if img_name in ignore_img_names:
            #     continue

            img_path = img_dir.joinpath(img_name)
            assert (img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev  # why?

        img = cv2.imread(str(img_path), 1)  # 读彩色图
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)
        # transpose(img, (2,0,1))不好理解 clip是越界则取边界


class FaceDataset_morph2(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert (data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir_suffix = data_dir[data_dir.rindex("/") + 1:]
        img_dir = Path(data_dir).joinpath(img_dir_suffix)  # 各种类型图片都在一个目录下
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        # ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        # ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            # if img_name in ignore_img_names:
            #     continue

            img_path = img_dir.joinpath(img_name)
            assert (img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev  # why?

        img = cv2.imread(str(img_path), 1)  # 读彩色图
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)
        # ------------np.ndarray转为torch.Tensor------------------------------------
        # numpy image: H x W x C
        # torch image:(nSample) x C x H x W
        # np.transpose(xxx, (2, 0, 1))   # 将 H x W x C 转化为 C x H x W

        # torch image: C x H x W
        # clip是越界则取边界


class FaceDataset_ceface(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert (data_type in ("train", "valid", "test"))
        img_dir_suffix = data_dir[data_dir.rindex("/") + 1:]
        data_dir = data_dir[:data_dir.rindex("/")]
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")

        img_dir = Path(data_dir).joinpath(img_dir_suffix)  # ce的只有align后的，因为图片太大了
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        # ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        # ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            # if img_name in ignore_img_names:
            #     continue

            img_path = img_dir.joinpath(img_name)
            assert (img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev  # why?

        img = cv2.imread(str(img_path), 1)  # 读彩色图
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)
        # ------------np.ndarray转为torch.Tensor------------------------------------
        # numpy image: H x W x C
        # torch image:(nSample) x C x H x W
        # np.transpose(xxx, (2, 0, 1))   # 将 H x W x C 转化为 C x H x W

        # torch image: C x H x W
        # clip是越界则取边界


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    dataset = FaceDataset(args.data_dir, "train")
    print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "valid")
    print("valid dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "test")
    print("test dataset len: {}".format(len(dataset)))


if __name__ == '__main__':
    main()
