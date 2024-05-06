# MedIM
This repo holds the Pytorch implementation of MedIM:<br />

**MedIM: Rethinking Masked Image Modeling for Medical Image Representation** 


### Requirements 
```
Python 3.10
Torch==1.13.1
Torchvision==0.14.1
CUDA 11.6
```
### Usage
* Create a new conda environment 
```
conda create --name medim python=3.10
source activate medim
```
* Clone this repo
```
git clone https://github.com/YtongXie/MedIM.git
cd MedIM
pip install -e .
```

* Install packages for MedIM
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

cd /medim/models/medim/pytorch-cosine-annealing-with-warmup
```
python setup.py install
```


### Dataset downloading

#### Datasets for pre-training
- **MIMIC-CXR**: downloaded images from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and paired medical reports from [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip). Both need to be a credentialed user for downloading.

#### Datasets for transfer learning
- **CheXpert**: downloaded the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset.

- **VinDr-CXR**: downloaded the [VinDr](https://vindr.ai/datasets/cxr) dataset. 

- **COVIDx**: downloaded the [COVIDx](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) dataset.

- **SIIM**: downloaded the [SIIM](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data) dataset.

After downloading datasets, please check if the path in `medim/constants.py` is correct.

#### Data Preprocessing
Preprocesseding these datasets and split the dataset into train/val/test set using the code in `medim/preprocess`.



### Pre-training

cd data/mesh
* Run
```
python MeSH_preprocess.py
python MeSH_pickle_train.py
python MeSH_pickle_valid.py

```
for generating `captions_train_MeSH.pickle` and `captions_valid_MeSH.pickle`.

cd /medim/models/medim
* Run
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python medim_train.py --gpus 8 --epochs 50 --outpath MedIM --strategy ddp
```
for MedIM pre-training.



### Finetune on downstream tasks
* We evlauate the performance of MedIM framework on five downstream tasks: classification and semantic segmentation.
```
CUDA_VISIBLE_DEVICES=0 python medim_finetuner.py --gpus 1 --dataset chexpert5 --batch_size 512 --accumulate 8 --path MedIM_weights.ckpt --outdir Cls_chexpert5 --outpath MedIM_1 --data_pct 1
CUDA_VISIBLE_DEVICES=0 python medim_finetuner.py --gpus 1 --dataset chexpert14 --batch_size 512 --accumulate 8 --path MedIM_weights.ckpt --outdir Cls_chexpert14 --outpath MedIM_1 --data_pct 1
CUDA_VISIBLE_DEVICES=0 python medim_finetuner.py --gpus 1 --dataset covidx --batch_size 96 --path MedIM_weights.ckpt --outdir Cls_covidx --outpath MedIM_1 --data_pct 1
CUDA_VISIBLE_DEVICES=0 python medim_finetuner.py --gpus 1 --dataset vindr --batch_size 512 --path MedIM_weights.ckpt --outdir Cls_vindr --outpath MedIM_1 --data_pct 1
CUDA_VISIBLE_DEVICES=0 python medim_segmenter.py --gpus 1 --dataset siim --batch_size 32 --ckpt_path MedIM_weights.ckpt --outdir Seg_siim --outpath MedIM_1 --data_pct 1 --seed 123
```


### Acknowledgements
Part of codes is reused from the [MGCA](https://github.com/HKU-MedAI/MGCA). Thanks to Fuying et al. for the codes of MGCA.

### Citation
If this code is helpful for your study, please cite:

```
@inproceedings{xie2023medim,
  title={Medim: Boost medical image representation via radiology report-guided masking},
  author={Xie, Yutong and Gu, Lin and Harada, Tatsuya and Zhang, Jianpeng and Xia, Yong and Wu, Qi},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={13--23},
  year={2023},
  organization={Springer}
}
```
The extended journal version is under review.

### Contact
Yutong Xie (yutong.xie678@gmail.com)
