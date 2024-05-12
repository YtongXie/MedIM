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
from transformers import BertTokenizer, BertModel, BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=240, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        print(self.transform)
        self.imsize = imsize
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.tokenizerfast = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        # load studies and study to text mapping
        self.filenames, self.path2sent, self.path2MaskMeSH_or, self.path2MaskMeSH_re = self.load_text_data(split)
        print(len(self.filenames))


    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(BASE_DIR, "../../data/captions_" + split + "_HardMatchv1.pickle")

        with open(filepath, "rb") as f:
            path2sent = pickle.load(f)
            path2MaskMeSH_or = pickle.load(f)
            path2MaskMeSH_re = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            path = path.replace('/home/ytxie','/media')
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent, path2MaskMeSH_or, path2MaskMeSH_re


    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]
        mask_MeSH = self.path2MaskMeSH_or[path]
        
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))

        series_sents_mask = []
        for sent_id in range(len(series_sents)):
            sent_i = series_sents[sent_id]
            if "_" in sent_i:
                num_words = 0
                for sent_i_w in sent_i.split(" "):
                    if '_' in sent_i_w:
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

                    else:
                        num_words = num_words + 1
                sent_i_mask = np.zeros(num_words) + sent_id + 1
            else:
                sent_i_mask = np.zeros(len(sent_i.split(" "))) + sent_id + 1
            series_sents_mask.append(sent_i_mask)

        all_sents_mask = np.concatenate(series_sents_mask, 0)
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        words = []
        word_bank = []
        for word_id in tokens['input_ids'][0][0:-1]:
            word = self.idxtoword[word_id.item()]
            if word == "[SEP]":
                words.append("".join(word_bank))
                words.append(word)
                break
            # This is because some words are divided into two words.
            if not word.startswith("##"):
                if len(word_bank) == 0:
                    word_bank.append(word)
                else:
                    words.append("".join(word_bank))
                    word_bank = [word]
            else:
                word_bank.append(word[2:])

        if "[SEP]" in words:
            mask_MeSH_trunc = mask_MeSH[0:len(words)-2]
            all_sents_mask_trunc = all_sents_mask[0:len(words)-2]
        else:
            mask_MeSH_trunc = mask_MeSH[0:len(words)-1]
            all_sents_mask_trunc = all_sents_mask[0:len(words)-1]

        # find the sents with the number of works > 3  
        select_sents = np.unique(all_sents_mask_trunc, return_counts=True)[0][np.where((np.unique(all_sents_mask_trunc, return_counts=True)[1] > 3))]
        if len(select_sents) == 0:
            select_sent_id = np.random.permutation(np.unique(all_sents_mask_trunc, return_counts=True)[0])[0]
        else:
            select_sent_id = np.random.permutation(np.unique(select_sents, return_counts=True)[0])[0]

        return tokens, x_len, mask_MeSH_trunc, all_sents_mask_trunc, select_sent_id

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len, mask_MeSH, all_sents_mask, select_sent_id = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        # imgs = get_imgs(self.images[key], self.transform, multiscale=False)

        return imgs, caps, cap_len, mask_MeSH, all_sents_mask, select_sent_id, key


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, mask_MeSH, all_sents_mask, select_sent_id, ids, tokens, attention = [], [], [], [], [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, m_MeSH, a_sents_mask, s_sent_id, p = b
        m_MeSH_p = np.pad(m_MeSH, (1, 112-len(m_MeSH)-1), 'constant', constant_values=(0, 0))
        a_sents_mask_p = np.pad(a_sents_mask, (1, 112-len(a_sents_mask)-1), 'constant', constant_values=(0, 0))
        s_sent_id = np.repeat(s_sent_id, len(a_sents_mask_p))
        imgs.append(img)
        cap_len.append(cap_l)
        mask_MeSH.append(m_MeSH_p)
        all_sents_mask.append(a_sents_mask_p)
        select_sent_id.append(s_sent_id)
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
    all_sents_mask = torch.from_numpy(np.stack(all_sents_mask).squeeze())
    select_sent_id = torch.from_numpy(np.stack(select_sent_id).squeeze())

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
        "all_sents_mask": all_sents_mask[sorted_cap_indices],
        "select_sent_id": select_sent_id[sorted_cap_indices],
        "path": path[sorted_cap_indices],
        "cap_lens": sorted_cap_lens
    }
    return return_dict


if __name__ == "__main__":
    from medim.datasets.transforms import DataTransforms

    transform = DataTransforms(is_train=True)
    dataset_train = MultimodalPretrainingDataset(split="train", transform=transform)
    # dataset_valid = MultimodalPretrainingDataset(split="valid", transform=transform)
    n = 0
    for batch in dataset_train:
        n = n + 1
        print(n)


