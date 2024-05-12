import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from medim.constants import *
from medim.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        with open(os.path.join(BASE_DIR, '../../data/mesh/mesh_ACE_pre.txt'), mode='rb') as file:
            mesh = file.readlines()
        
        self.MeSH = []
        for line in mesh:
            for sub_line in line.decode('utf-8')[0:-1].lower().split(','):
                if sub_line[0]==' ':
                    sub_line = sub_line[1::]
                if sub_line[-1]==' ':
                    sub_line = sub_line[0:-1]
                self.MeSH.append(sub_line)
                
        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").cuda()
        self.max_words = max_words

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2MaskMeSH_or, self.path2MaskMeSH_re = self.load_text_data(split)


    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(BASE_DIR, "../../data/captions_" + split + "_HardMatchv1.pickle")

        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, path2MaskMeSH_or, path2MaskMeSH_re = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                pickle.dump(path2MaskMeSH_or, f, protocol=2)
                pickle.dump(path2MaskMeSH_re, f, protocol=2)

                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)
                path2MaskMeSH_or = pickle.load(f)
                path2MaskMeSH_re = pickle.load(f)


        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        # for k in filenames[0:139918]:
        #     path2sent.pop(k)

        # return filenames[139918::], path2sent
        return filenames, path2sent, path2MaskMeSH_or, path2MaskMeSH_re

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        path2MaskMeSH_or = {}
        path2MaskMeSH_re = {}

        splitter = re.compile("[0-9]+\.")
        tokenizer = RegexpTokenizer(r"\w+")

        # iterrows is not faster than itertuples ...  but it is ok
        for row in tqdm(self.df.itertuples(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            # captions += row["impression"]
            captions += getattr(row, 'impression')
            captions += " "
            # captions += row["findings"]
            captions += getattr(row, "findings")

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # study_tags = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                # included_tags = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))
                    # study_tags.append(included_tags)

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[getattr(row, MIMIC_CXR_PATH_COL)] = study_sent

                # mask_MeSH
                series_sents = list(filter(lambda x: x != "", study_sent))
                sent = " ".join(series_sents)

                score_simi_or = []
                score_simi_re = []
                match_hard = []
                ids = []
                for sent_i_w in sent.split(" "):

                    tokens = self.tokenizer(sent_i_w, return_tensors='pt')                    
                    num_words = 0
                    if "_" in sent_i_w:
                        if '_' == sent_i_w:
                            num_words = num_words + 1   # '_'
                        elif '__' == sent_i_w:
                            num_words = num_words + 2   # '__'
                        elif '___' == sent_i_w:
                            num_words = num_words + 3   # '___'
                        elif '____' == sent_i_w:
                            num_words = num_words + 4   # '____'
                        elif '_____' == sent_i_w:
                            num_words = num_words + 5   # '_____'
                        elif '______________________________________________________________________________' == sent_i_w:
                            num_words = num_words + 78   # '_____'

                        elif '__' not in sent_i_w:
                            if ((sent_i_w[0] != '_') and (sent_i_w[-1] != '_')) or ((sent_i_w[0] == '_') and (sent_i_w[-1] == '_')):
                                num_words = num_words + 3   # 'x_x' or '_x_'                               
                            else:
                                num_words = num_words + 2   # '_x' or 'x_'

                        elif '___' not in sent_i_w:
                            if (sent_i_w[0] == '_') and (sent_i_w[-1] == '_'):
                                num_words = num_words + 5   # '__x__'  
                            elif (sent_i_w[0] != '_') and (sent_i_w[-1] != '_'):
                                num_words = num_words + 4   # 'x__x'
                            else:
                                num_words = num_words + 3   # '__x' or 'x__'

                        elif '____' not in sent_i_w:
                            if (sent_i_w[0] == '_') and (sent_i_w[-1] == '_'):
                                num_words = num_words + 7   # '___x___'  
                            elif (sent_i_w[0] != '_') and (sent_i_w[-1] != '_'):
                                num_words = num_words + 5   # 'x___x'
                            else:                            
                                num_words = num_words + 4   # '___x' or 'x___'

                        elif '_____' not in sent_i_w:
                            if (sent_i_w[0] == '_') and (sent_i_w[-1] == '_'):
                                num_words = num_words + 9   # '____x____'  
                            elif (sent_i_w[0] != '_') and (sent_i_w[-1] != '_'):
                                num_words = num_words + 6   # 'x____x'
                            else:
                                num_words = num_words + 5   # '____x' or 'x____'

                        elif '_____' not in sent_i_w:
                            if (sent_i_w[0] == '_') and (sent_i_w[-1] == '_'):
                                num_words = num_words + 11   # '_____x_____'  
                            elif (sent_i_w[0] != '_') and (sent_i_w[-1] != '_'):
                                num_words = num_words + 7   # 'x_____x'
                            else:
                                num_words = num_words + 6   # '_____x' or 'x_____'

                        else:   # 78
                            if (sent_i_w[0] == '_') and (sent_i_w[-1] == '_'):
                                num_words = num_words + 157   # '_____x_____'  
                            elif (sent_i_w[0] != '_') and (sent_i_w[-1] != '_'):
                                num_words = num_words + 80   # 'x_____x'
                            else:
                                num_words = num_words + 79   # '_____x' or 'x_____'
    
                        score_simi_i = 0
                        score_simi_or.extend(np.repeat(score_simi_i, num_words, axis=None))

                    else:
                        if sent_i_w in self.MeSH:
                            match_hard.append(sent_i_w)
                            score_simi_i = 1
                        else:
                            score_simi_i = 0
                        score_simi_or.append(score_simi_i)

                    score_simi_re.extend(np.repeat(score_simi_i, tokens['input_ids'][0][1:-1].shape[0], axis=None))
                    ids.append(tokens['input_ids'][0][1:-1])

                if len(score_simi_re) > (self.max_words-2):
                    score_simi_re = score_simi_re[0:self.max_words-2]

                score_simi_re = np.pad(score_simi_re, (1, self.max_words-len(score_simi_re)-1), 'constant', constant_values=(0, 0))

                path2MaskMeSH_or[getattr(row, MIMIC_CXR_PATH_COL)] = score_simi_or
                path2MaskMeSH_re[getattr(row, MIMIC_CXR_PATH_COL)] = score_simi_re

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, path2MaskMeSH_or, path2MaskMeSH_re

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]
        mask_MeSH = self.path2MaskMeSH_re[path]
        
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len, mask_MeSH

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len, mask_MeSH = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, mask_MeSH, key
        # key = self.filenames[index]
        # return key

def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, mask_MeSH, ids, tokens, attention = [], [], [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, m_MeSH, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        mask_MeSH.append(m_MeSH)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    mask_MeSH = torch.from_numpy(np.stack(mask_MeSH).squeeze())

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "mask_MeSH": mask_MeSH[sorted_cap_indices],
        "path": path[sorted_cap_indices],
        "cap_lens": sorted_cap_lens
    }
    return return_dict


if __name__ == "__main__":
    from medim.datasets.transforms import DataTransforms
    from medim.datasets.data_module import DataModule
    import cv2

    transform = DataTransforms(is_train=True)
    dataset_valid = MultimodalPretrainingDataset(split="valid", transform=transform)
    for batch in dataset_valid:
        # if not os.path.exists(batch):
        print(batch)
        print(cv2.imread(str(batch), 0).shape)
