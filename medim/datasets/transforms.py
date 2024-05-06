import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import torch


class DataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            data_transforms = [
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5))
            ]
        else:
            data_transforms = [
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        self.data_transforms = transforms.Compose(data_transforms)
    def __call__(self, image):
        return self.data_transforms(image)


class Moco2Transform(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224, test_augment=False) -> None:
        if is_train:
            # This setting follows SimCLR
            self.data_transforms = transforms.Compose(
                [
                    transforms.RandomCrop(crop_size),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
        else:
            transformations_list = []

            normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            if test_augment:
                transformations_list.append(transforms.TenCrop(crop_size))
                transformations_list.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
                if normalize is not None:
                    transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
            else:
                transformations_list.append(transforms.CenterCrop(crop_size))
                transformations_list.append(transforms.ToTensor())
                if normalize is not None:
                    transformations_list.append(normalize)

            self.data_transforms = transforms.Compose(transformations_list)


    def __call__(self, img):
        return self.data_transforms(img)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
