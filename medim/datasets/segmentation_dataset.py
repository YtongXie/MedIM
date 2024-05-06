import pickle

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations import Compose, Normalize, Resize, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
from medim.constants import *
from medim.datasets.classification_dataset import BaseImageDataset
from medim.datasets.utils import resize_img
from PIL import Image
import random

np.random.seed(42)


class SIIMImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 data_pct=0.01, phase="segmentation", imsize=224):
        super().__init__(split, transform)

        self.phase = phase
        self.imsize = imsize
        if self.phase == "segmentation":
            self.seg_transform = self.get_transforms()
        else:
            raise NotImplementedError(f"{self.phase} not implemented")

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(PNEUMOTHORAX_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(PNEUMOTHORAX_VALID_CSV)
        else:
            self.df = pd.read_csv(PNEUMOTHORAX_TEST_CSV)

        self.df["Path"] = self.df["Path"].apply(
            lambda x: os.path.join(PNEUMOTHORAX_IMG_DIR, x))

        # only keep positive samples for segmentation
        self.df["class"] = self.df[" EncodedPixels"].apply(lambda x: x != "-1")
        if self.phase == "segmentation" and split == "train":
            self.df_neg = self.df[self.df["class"] == False]
            self.df_pos = self.df[self.df["class"] == True]
            n_pos = self.df_pos["ImageId"].nunique()
            neg_series = self.df_neg["ImageId"].unique()
            neg_series_selected = np.random.choice(
                neg_series, size=n_pos, replace=False
            )
            self.df_neg = self.df_neg[self.df_neg["ImageId"].isin(
                neg_series_selected)]
            self.df = pd.concat([self.df_pos, self.df_neg])

        # sample data
        if data_pct != 1 and split == "train":
            ids = self.df["ImageId"].unique()
            n_samples = int(len(ids) * data_pct)
            series_selected = np.random.choice(
                ids, size=n_samples, replace=False)
            self.df = self.df[self.df["ImageId"].isin(series_selected)]
        
        self.imgids = self.df.ImageId.unique().tolist()
        # print(self.imgids)

    def __getitem__(self, index):
        imgid = self.imgids[index]
        imgid_df = self.df.groupby("ImageId").get_group(imgid)

        # get image
        img_path = imgid_df.iloc[0]["Path"]
        x = self.read_from_dicom(img_path)

        # get labels
        if self.phase == "segmentation":
            rle_list = imgid_df[" EncodedPixels"].tolist()
            mask = np.zeros([1024, 1024])
            if rle_list[0] != "-1":
                for rle in rle_list:
                    mask += self.rle2mask(
                        rle, PNEUMOTHORAX_IMG_SIZE, PNEUMOTHORAX_IMG_SIZE
                    )
            mask = (mask >= 1).astype("float32")
            mask = resize_img(mask, self.imsize)

            augmented = self.seg_transform(image=x, mask=mask)
            x = augmented["image"]
            y = augmented["mask"].squeeze()
        else:
            y = imgid_df.iloc[0]["Label"]
            y = torch.tensor([y])

        return x, y

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def __len__(self):
        return len(self.imgids)

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def rle2mask(self, rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position: current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.imsize, self.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms

if __name__ == "__main__":
    dataset = SIIMImageDataset()
    for data in dataset:
        img, mask = data
        print(img.shape)
        break
