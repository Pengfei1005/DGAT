import json
import numpy as np
from imageio.v3 import imread
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from preprocess_data import encoding_token
import torchvision.transforms as transforms


class LEVIR_CC_Dataset(Dataset):
    def __init__(self, data_path, list_path, split, max_length=40, allow_unknown=0, vocab_file=None, token_path=None):

        self.mean = [100.6790, 99.5023, 84.9932]
        self.std = [50.9820, 48.4838, 44.7057]
        self.split = split
        self.max_length = max_length

        assert self.split in ['train', 'val', 'test']
        self.img_info = [i.strip() for i in open(os.path.join(list_path, self.split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unknown = allow_unknown

        self.files = []
        if split == 'train':
            for img_ids in self.img_info:
                img_A = os.path.join(data_path + '/' + split + '/A/' + img_ids.split('-')[0])
                img_B = os.path.join(data_path + '/' + split + '/B/' + img_ids.split('-')[0])
                token_id = img_ids.split('-')[-1]
                if token_path is not None:
                    token_file = os.path.join(token_path + img_ids.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "img_A": img_A,
                    "img_B": img_B,
                    "token": token_file,
                    "token_id": token_id,
                    "img_ids": img_ids.split('-')[0]
                })

        elif split == 'val':
            for img_ids in self.img_info:
                img_A = os.path.join(data_path + '/' + split + '/A/' + img_ids)
                img_B = os.path.join(data_path + '/' + split + '/B/' + img_ids)

                token_id = None
                if token_path is not None:
                    token_file = os.path.join(token_path + img_ids.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "img_A": img_A,
                    "img_B": img_B,
                    "token": token_file,
                    "token_id": token_id,
                    "img_ids": img_ids
                })

        elif split == 'test':
            for img_ids in self.img_info:
                img_A = os.path.join(data_path + '/' + split + '/A/' + img_ids)
                img_B = os.path.join(data_path + '/' + split + '/B/' + img_ids)

                token_id = None
                if token_path is not None:
                    token_file = os.path.join(token_path + img_ids.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "img_A": img_A,
                    "img_B": img_B,
                    "token": token_file,
                    "token_id": token_id,
                    "img_ids": img_ids
                })

        
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_ids = datafiles["img_ids"]

        img_A = imread(datafiles["img_A"])
        img_B = imread(datafiles["img_B"])

        img_A = np.asarray(img_A, np.float32)
        img_B = np.asarray(img_B, np.float32)
        
        img_A = np.moveaxis(img_A, -1, 0)
        img_B = np.moveaxis(img_B, -1, 0)

        for i in range(len(self.mean)):
            img_A[i, :, :] -= self.mean[i]
            img_A[i, :, :] /= self.std[i]
            img_B[i, :, :] -= self.mean[i]
            img_B[i, :, :] /= self.std[i]

        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for i, tokens in enumerate(caption_list):
                token_encode = encoding_token(tokens, self.word_vocab, allow_unknown=self.allow_unknown == 1)
                token_all[i, :len(token_encode)] = token_encode
                token_all_len[i] = len(token_encode)

            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                i = np.random.randint(len(caption_list) - 1)
                token = token_all[i]
                token_len = token_all_len[i].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)

        return (img_A.copy(), img_B.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len),
                img_ids)

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    train_dataset = LEVIR_CC_Dataset(data_path='/mnt/data0/qpf_4000/DGAT/data/LEVIR_CC/images', list_path='/mnt/data0/qpf_4000/DGAT/data/LEVIR_CC',
                                     split='train', token_path=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    print("ok")
