# -*- coding: utf-8 -*-
import numpy as np
from Data_Generator import PoemDataSet


def load_wordvec(path, word2id, word_dim=100):
    """load word vectors
    :param path: the word vectors path.
    :param word2id: map word to id.
    :return:
    """
    weight = np.zeros([len(word2id), word_dim])
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            word_vector = line.strip().split()
            word = np.array(word_vector[0])
            vector = np.array([float(i) for i in word_vector[1:]])
            # print(word_vector)
            # print("word: ", word, " vector: ", vector)
            weight[word2id[str(word)]] = vector
    return weight


if __name__ == "__main__":
    poem_data = PoemDataSet(filename='./data/train_resource.json')
    load_wordvec("wordvect/data/embedding_model_t2s/vector_t2s", poem_data.word2id)
