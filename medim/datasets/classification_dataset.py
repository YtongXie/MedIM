import os

import numpy as np
import pandas as pd
import torch
from medim.constants import *
from medim.datasets.utils import get_imgs, read_from_dicom
from torch.utils.data import Dataset

np.random.seed(42)


class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CheXpert14ImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 img_type="Frontal", data_pct=0.01, imsize=240):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError(f"{CHEXPERT_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(CHEXPERT_ORIGINAL_VALID_CSV)
        elif split == "test":
            self.df = pd.read_csv(CHEXPERT_ORIGINAL_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[CHEXPERT_VIEW_COL] == img_type]

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=123)

        # get path
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(
                CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0)


        # replace uncertains
        uncertain_mask = {k: -1 for k in CHEXPERT_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN14_MAPPINGS)

        self.path = self.df["Path"].values
        self.labels = self.df.loc[:, CHEXPERT_TASKS].values

    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y)

        return x, y#, img_path

    def __len__(self):
        return len(self.df)


class CheXpert5ImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 img_type="Frontal", data_pct=0.01, imsize=240):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError(f"{CHEXPERT_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(CHEXPERT_ORIGINAL_VALID_CSV)
        elif split == "test":
            self.df = pd.read_csv(CHEXPERT_ORIGINAL_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[CHEXPERT_VIEW_COL] == img_type]

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=123)

        # get path
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(
                CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.path = self.df["Path"].values
        self.labels = self.df.loc[:, CHEXPERT_COMPETITION_TASKS].values


    def __getitem__(self, index):
        # get image
        img_path = self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y)

        return x, y#, img_path

    def __len__(self):
        return len(self.df)


class VinDrImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=0.01, imsize=240):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(VinDr_DATA_DIR):
            raise RuntimeError(f"{VinDr_DATA_DIR} does not exist!")

        self.imsize = imsize

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(VinDr_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(VinDr_VALID_CSV)
        elif split == "test":
            self.df = pd.read_csv(VinDr_TEST_CSV)
        else:
            raise NotImplementedError(f"split {split} is not implemented!")

        # sample data
        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=123)

        # fill na with 0s
        self.df = self.df.fillna(0)
        self.path = self.df["image_id"].values
        self.labels = self.df.loc[:, self.df.columns.tolist()[2:]].values


    def __getitem__(self, index):
        # get image
        img_path = VinDr_DATA_DIR + self.path[index]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = self.labels[index]
        y = torch.tensor(y)

        return x, y#, img_path

    def __len__(self):
        return len(self.df)


class MIMICImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=1.0, img_type="Frontal", imsize=256):
        super().__init__(split, transform)
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(
                "MIMIC CXR data directory %s does not exist!" % MIMIC_CXR_DATA_DIR)

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)
        else:
            self.df = pd.read_csv(MIMIC_CXR_TEST_CSV)

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[MIMIC_CXR_VIEW_COL].isin(["PA", "AP"])]

        # get a fraction of dataset
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
            # print(self.df)

        # get path
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(
                MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.imsize = imsize

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # get image
        img_path = row["Path"]
        x = get_imgs(img_path, self.imsize, self.transform)

        # get labels
        y = list(row[CHEXPERT_COMPETITION_TASKS])
        y = torch.tensor(y)

        return x, y, img_path

    def __len__(self):
        return len(self.df)


class COVIDXImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(COVIDX_DATA_DIR):
            raise RuntimeError(f"{COVIDX_DATA_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(COVIDX_TRAIN_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: os.path.join(COVIDX_DATA_DIR, f"data/{x}"))
        elif self.split == "valid":
            self.df = pd.read_csv(COVIDX_VALID_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: os.path.join(COVIDX_DATA_DIR, f"data/{x}"))
        elif self.split == "test":
            self.df = pd.read_csv(COVIDX_TEST_CSV)
            self.df["filename"] = self.df["filename"].apply(
                lambda x: os.path.join(COVIDX_DATA_DIR, f"data/{x}"))
        else:
            raise ValueError(f"split {split} does not exist!")

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["filename"]
        x = get_imgs(img_path, self.imsize, self.transform)
        y = float(row["labels"])
        y = torch.tensor([y])

        return x, y
