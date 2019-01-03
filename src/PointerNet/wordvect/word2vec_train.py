# -*- coding: utf-8 -*-
import os
from gensim.models import word2vec
import multiprocessing
import codecs


class Config:
    data_path = 'data'
    poem_data = 'poem_data.txt'
    embedded_model_t2s = 'embedding_model_t2s/poem_embedding_t2s.model'
    embedded_vector_t2s = 'embedding_model_t2s/vector_t2s'


def word2vec_train(_config, saved=False):
    print('Start...')
    model = word2vec.Word2Vec(LineSentence(os.path.join(_config.data_path, _config.poem_data)),
                              size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
    if saved:
        model.save(os.path.join(_config.data_path, _config.embedded_model_t2s))
        model.wv.save_word2vec_format(os.path.join(_config.data_path, _config.embedded_vector_t2s), binary=False)
    print("Finished!")
    return model


def wordsimilarity(word, model):
    semi = ''
    try:
        semi = model.most_similar(word, topn=10)
    except KeyError:
        print('The word not in vocabulary!')
    for term in semi:
        print('%s,%s' % (term[0], term[1]))


def LineSentence(path):
    """将指定路经的文本转换成iterable of iterables"""
    sentences = []
    i = 0
    with codecs.open(path, 'r', encoding="UTF-8") as raw_texts:
        for line in raw_texts.readlines():
            line = line.strip()
            sent_list = [s for s in line]
            i += 1
            print("sent ", i, sent_list)
            sentences.append(sent_list)
    print("read sentences done!")
    return sentences


if __name__ == "__main__":
    config = Config()
    model = word2vec_train(config, saved=True)