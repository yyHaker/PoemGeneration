import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm
import json


class PoemDataSet(Dataset):
    """DataSet for poem generation.
    data:
       'en_in_seq',
       'en_len',
       'de_in_seq',
       'de_out',
       'de_len',
    """
    def __init__(self, filename):
        super(PoemDataSet, self).__init__()

        self._load_data(filename)

    def _load_data(self, filename):
        """
        load data, get word2id and id2word.
        :param filename:
        :return:
        """
        # total data list
        self.total_samples_list = []
        self.word2id = {}
        self.id2word = {}
        with open(filename, 'r', encoding='utf-8') as f:
            # load data
            for item in f.readlines():
                data = json.loads(item)
                # get word2id
                samples = data['sample']
                self.total_samples_list.extend(samples)
                for s in samples:
                    seq = s[1]
                    for ch in seq:
                        if ch not in self.word2id:
                            self.word2id[ch] = len(self.word2id)
            # get id2word
            for k, v in self.word2id.items():
                self.id2word[v] = k
            # print(self.word2id)
            #  get train data
            # sample = self.data_list[0]
            # print(sample['author'])
            # print(sample['paragraphs'])
            # print(sample['title'])
            # print(sample['sents'])
            # print(sample['sample'][0][0], " ", sample['sample'][0][1])
        # print("total sample list: ", self.total_samples_list)

    def __len__(self):
        return len(self.total_samples_list)

    def __getitem__(self, index):
        # chars list, 2item
        sample = self.total_samples_list[index]
        # tensor list(for train)
        points = torch.LongTensor(self.strList2ids(sample[1]))
        solution = torch.LongTensor([sample[1].index(ch) for ch in sample[0]])
        sample_tensor = {'Points': points, 'Solution': solution}
        return sample_tensor

    def strList2ids(self, chs):
        return [self.word2id[ch] for ch in chs]

    def idList2str(self, idList):
        return [self.id2word[idx] for idx in idList]


if __name__ == "__main__":
    poemDataset = PoemDataSet(filename='./data/resource_total')
    for sample in poemDataset:
        print(sample)
    print("data size: ", len(poemDataset))
    # 11483
