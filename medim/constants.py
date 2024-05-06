import os
from pathlib import Path


DATA_BASE_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../../data")
DATA_BASE_DIR = Path(DATA_BASE_DIR)

# #############################################
# MIMIC-CXR-JPG constants
# #############################################
# MIMIC_CXR_DATA_DIR = "/media/userdisk1/ytxie/SSL/data/2019.MIMIC-CXR-JPG/2.0.0"
MIMIC_CXR_DATA_DIR = "/home/ytxie/userdisk2/ytxie/SSL_data/2019.MIMIC-CXR-JPG/2.0.0"
MIMIC_CXR_CHEXPERT_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "mimic-cxr-2.0.0-chexpert.csv")
MIMIC_CXR_META_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "mimic-cxr-2.0.0-metadata.csv")
MIMIC_CXR_TEXT_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "mimic_cxr_sectioned.csv")
MIMIC_CXR_SPLIT_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "mimic-cxr-2.0.0-split.csv")
# Created csv
MIMIC_CXR_TRAIN_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "train.csv")
MIMIC_CXR_VALID_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "test.csv")
MIMIC_CXR_TEST_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "test.csv")
MIMIC_CXR_MASTER_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "master_v0.csv")
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"


# #############################################
# CheXpert constants
# #############################################
CHEXPERT_DATA_DIR = "/media/userdisk2/ytxie/CheXpert-v1.0-small"
CHEXPERT_ORIGINAL_TRAIN_CSV = os.path.join(CHEXPERT_DATA_DIR, "train.csv")
CHEXPERT_ORIGINAL_VALID_CSV = os.path.join(CHEXPERT_DATA_DIR, "valid.csv")
CHEXPERT_ORIGINAL_TEST_CSV = os.path.join(CHEXPERT_DATA_DIR, "test_labels_update_500.csv")

CHEXPERT_TRAIN_CSV = os.path.join(CHEXPERT_DATA_DIR, "train_split.csv")  # train split from train.csv
CHEXPERT_VALID_CSV = os.path.join(CHEXPERT_DATA_DIR, "valid_split.csv")  # valid split from train.csv
CHEXPERT_TEST_CSV = (
    os.path.join(CHEXPERT_DATA_DIR, "valid.csv")
)  # using validation set as test set (test set label hidden)
CHEXPERT_MASTER_CSV = (
    os.path.join(CHEXPERT_DATA_DIR, "master_updated.csv")
)  # contains patient information, not PHI conplient
CHEXPERT_TRAIN_DIR = os.path.join(CHEXPERT_DATA_DIR, "train")
CHEXPERT_TEST_DIR = os.path.join(CHEXPERT_DATA_DIR, "valid")
CHEXPERT_5x200 = os.path.join(CHEXPERT_DATA_DIR, "chexpert_5x200.csv")
CHEXPERT_8x200_QUERY = os.path.join(CHEXPERT_DATA_DIR, "chexpert_8x200_query.csv")
CHEXPERT_8x200_CANDIDATES = os.path.join(CHEXPERT_DATA_DIR, "chexpert_8x200_candidates.csv")

CHEXPERT_VALID_NUM = 5000
CHEXPERT_VIEW_COL = "Frontal/Lateral"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_SPLIT_COL = "Split"
CHEXPERT_REPORT_COL = "Report Impression"

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# baseed on original chexpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}



# baseed on original chexpert paper
CHEXPERT_UNCERTAIN14_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
    "No Finding": 1,
    "Enlarged Cardiomediastinum": 1,
    "Lung Lesion": 1,
    "Lung Opacity": 1,
    "Pneumonia": 1,
    "Pneumothorax": 1,
    "Pleural Other": 1,
    "Fracture": 1,
    "Support Devices": 1,
}


CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}



# #############################################
# SIIM constants
# #############################################
PNEUMOTHORAX_DATA_DIR = "/media/userdisk2/ytxie/SIIM_TRAIN_TEST"
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = os.path.join(PNEUMOTHORAX_DATA_DIR, "train-rle.csv")
PNEUMOTHORAX_TRAIN_CSV = os.path.join(PNEUMOTHORAX_DATA_DIR, "train.csv")
PNEUMOTHORAX_VALID_CSV = os.path.join(PNEUMOTHORAX_DATA_DIR, "valid.csv")
PNEUMOTHORAX_TEST_CSV = os.path.join(PNEUMOTHORAX_DATA_DIR, "test.csv")
PNEUMOTHORAX_IMG_DIR = os.path.join(PNEUMOTHORAX_DATA_DIR, "dicom-images-train")
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7


# #############################################
# COVIDx constants
# #############################################
COVIDX_DATA_DIR = "/media/userdisk2/ytxie/COVIDx"
COVIDX_ORIGINAL_TRAIN_TXT = os.path.join(COVIDX_DATA_DIR, "train_COVIDx_CXR-3A.txt")
COVIDX_ORIGINAL_TEST_TXT = os.path.join(COVIDX_DATA_DIR, "test_COVIDx_CXR-3A.txt")
COVIDX_TRAIN_CSV = os.path.join(COVIDX_DATA_DIR, "train.csv")
COVIDX_VALID_CSV = os.path.join(COVIDX_DATA_DIR, "valid.csv")
COVIDX_TEST_CSV = os.path.join(COVIDX_DATA_DIR, "test.csv")



# #############################################
# VinDr constants
# #############################################
VinDr_DATA_DIR = "/media/userdisk2/ytxie/VinBigData/"
VinDr_ORIGINAL_TRAIN = os.path.join(VinDr_DATA_DIR, "annotations/image_labels_train.csv")
VinDr_ORIGINAL_TEST = os.path.join(VinDr_DATA_DIR, "annotations/image_labels_test.csv")
VinDr_AllTRAIN_CSV = os.path.join(VinDr_DATA_DIR, "trainall.csv")
VinDr_TRAIN_CSV = os.path.join(VinDr_DATA_DIR, "train.csv")
VinDr_VALID_CSV = os.path.join(VinDr_DATA_DIR, "valid.csv")
VinDr_TEST_CSV = os.path.join(VinDr_DATA_DIR, "test.csv")
# VinDr_FIND_TASKS = []
VinDr_FIND_DIAGO = ['Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'COPD', 'No finding']




