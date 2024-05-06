import os
import torch.nn as nn
from transformers import logging
import segmentation_models_pytorch as smp
import torch

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SETRDecoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64], masking_type = 'both'):
        super().__init__()
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode="nearest")
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.masking_type = masking_type
        if self.masking_type == 'word':
            self.final_out_word = nn.Conv2d(features[-1], out_channels, 3, padding=1)
        elif self.masking_type == 'sents':
            self.final_out_sent = nn.Conv2d(features[-1], out_channels, 3, padding=1)
        elif self.masking_type == 'both':
            self.final_out_word = nn.Conv2d(features[-1], out_channels, 3, padding=1)
            self.final_out_sent = nn.Conv2d(features[-1], out_channels, 3, padding=1)


    def forward(self, x):
        x = nn.functional.interpolate(self.decoder_1(x), scale_factor=2, mode="nearest")
        x = nn.functional.interpolate(self.decoder_2(x), scale_factor=2, mode="nearest")
        x = nn.functional.interpolate(self.decoder_3(x), scale_factor=2, mode="nearest")
        x = nn.functional.interpolate(self.decoder_4(x), scale_factor=2, mode="nearest")
        # x = self.final_out(x)

        if self.masking_type == 'word':
            x = self.final_out_word(x)
        elif self.masking_type == 'sents':
            x = self.final_out_sent(x)
        elif self.masking_type == 'both':
            x_word = self.final_out_word(x[0:int(len(x)/2)])
            x_sents = self.final_out_sent(x[int(len(x)/2)::])
            x = torch.cat([x_word, x_sents], 0)

        return x


class ImageDecoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 output_dim: int = 3,
                 masking_type: str = "both",
                 ):
        super(ImageDecoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim

        if "vit" in model_name:
            self.model = SETRDecoder2D(768, self.output_dim, features=[512, 256, 128, 64], masking_type = masking_type)
        else:
            self.model = smp.unet.decoder.UnetDecoder(
                        encoder_channels=(3, 64, 256, 512, 1024, 2048),
                        decoder_channels=(256, 128, 64, 32, 16),
                        n_blocks=5,
                        use_batchnorm=True,
                        center=True if model_name.startswith("vgg") else False,
                        attention_type=None)
            self.segmentation_head = smp.base.SegmentationHead(
                        in_channels=16,
                        out_channels=self.output_dim,
                        activation=None,
                        kernel_size=3,
                        )

    def resnet_de_forward(self, x):
        x = self.model(x)  
        x = self.segmentation_head(x)
        return x

    def vit_de_forward(self, x):
        return self.model(x)

    def forward(self, x):
        if "resnet" in self.model_name:
            return self.resnet_de_forward(x)
        elif "vit" in self.model_name:
            return self.vit_de_forward(x)

if __name__ == "__main__":
    from medim.datasets.pretrain_dataset import MultimodalPretrainingDataset
    from medim.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)

    for i, data in enumerate(dataset):
        imgs, caps, cap_len, key = data
        if caps["attention_mask"].sum() == 112:
            model = BertEncoder()
            report_feat, sent_feat, sent_mask, sents = model(
                caps["input_ids"],
                caps["attention_mask"],
                caps["token_type_ids"],
                get_local=True)
