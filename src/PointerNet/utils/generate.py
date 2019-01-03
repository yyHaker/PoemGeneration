# -*- coding: utf-8 -*-
"""
generate different level poems training set.
"""
import json
from random import sample, choice


def generate_poems_training_set(poems_file, level=1, generate_num=1000):
    """
    generate poems training set.
    :param poems_file: poems file. (to be confirmed.)
    :param level: difficulty level.
         '1':   resource num = 4(easy)
         '2':  resource num = 3(easy)
         '3': resource num = 2 (easy)

         '4': resource num = 4 (random)
         '5': resource num = 3 (random)
         '6': resource num = 2 (random)
    :param generate_num:
    :return:
     'poems_json_list:'
    """
    # read poems
    print("read poems....")
    poems_json_list = []
    poems_set = []  # poems set, poem is a list of words.
    poems_list = []
    with open(poems_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            poem_json = json.loads(line)
            poems_json_list.append(poem_json)
            poem_sents = poem_json['sents']
            title = poem_json['title']
            for sent in poem_sents:
                sent = sent.strip()
                poem = [w for w in sent]
                poem_s = set(poem)
                if len(poem_s) != 5:
                    continue
                poems_set.append(poem_s)
                poems_list.append((sent, title))

    print("generate poems words set.....")
    if level < 4:
        result = _generate_poems_words_set(poems_set, poems_list, resource_num=5-level, generate_num=generate_num)
    else:
        result = _generate_poems_words_set(poems_set, poems_list, resource_num=8-level, generate_num=generate_num)
    print("generate done!")
    for data in result:
        poem_title = data[0]
        for p_json in poems_json_list:
            if poem_title == p_json['title']:
                if 'sample' in p_json.keys():
                    p_json['sample'].append([data[1], ''.join(data[2])])
                else:
                    p_json['sample'] = [[data[1], ''.join(data[2])]]

    return poems_json_list


def _generate_poems_words_set(poem_set, poem_list, resource_num=2, generate_num=150000):
    """
    generate poems words set, according to resource num of the poems.
    :param poem_set: all poems set.
    :param poem_list: poems list of (sent, title)
    :param resource_num: resource num of the poems.
    :param generate_num: generate num.
    :return:
     'result': list,
    """
    result = []
    num = 0
    all_index = set([i for i in range(len(poem_set))])
    while num < generate_num:
        if num % 100 == 0:
            print("generate ", num)
        index = sample(all_index, 1)[0]
        poem_s = poem_set[index]
        poem = poem_list[index][0]
        other_index = all_index - set([index])
        extra_poem_index = sample(other_index, resource_num - 1)
        if resource_num == 2:   # 来源于两句诗
            extra_poem = poem_set[extra_poem_index[0]]
            extra_word = sample(extra_poem, 4)
            poem_sample = poem_s.union(set(extra_word))
            if extra_poem - poem_sample != set() and len(poem_sample) == 9:
                result.append([poem_list[index][1], poem_list[index][0], poem_sample])
                num += 1
        elif resource_num == 3:  # 来源于三句诗
            extra_poem1 = poem_set[extra_poem_index[0]]
            extra_poem2 = poem_set[extra_poem_index[1]]
            extra_word1 = sample(extra_poem1, 2)
            extra_word2 = sample(extra_poem2, 2)
            poem_sample = poem_s.union(set(extra_word1))
            poem_sample = poem_sample.union(set(extra_word2))
            if extra_poem1 - poem_sample != set() and extra_poem2 - poem_sample != set() and len(poem_sample) == 9:
                result.append([poem_list[index][1], poem_list[index][0], poem_sample])
                num += 1
        elif resource_num == 4:  #来源于四句诗
            extra_poem1 = poem_set[extra_poem_index[0]]
            extra_poem2 = poem_set[extra_poem_index[1]]
            extra_poem3 = poem_set[extra_poem_index[2]]
            extra_word1 = sample(extra_poem1, 1)
            extra_word2 = sample(extra_poem2, 1)
            extra_word3 = sample(extra_poem3, 2)
            poem_sample = poem_s.union(set(extra_word1))
            poem_sample = poem_sample.union(set(extra_word2))
            poem_sample = poem_sample.union(set(extra_word3))
            if extra_poem1 - poem_sample != set() and extra_poem2 - poem_sample != set() and extra_poem3 - poem_sample != set() and len(
                    poem_sample) == 9:
                result.append([poem_list[index][1], poem_list[index][0], poem_sample])
                num += 1
    return result


def output(data, path):
    """
    save the data to the file.
    :param data:
    :param path:
    :return:
    """
    print("write data to path...")
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def output_to_total(out_path):
    """
    :param out_path:
    :return:
    """
    print("write data to resource total.....")
    poems_json_list_total = []
    for i in range(1, 7):
        with open('../data/resource_level_%d' % i, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                json_data = json.loads(line)
                poems_json_list_total.append(json_data)

    output(poems_json_list_total, out_path)


if __name__ == "__main__":
    # easy poem....
    poems_json_list = generate_poems_training_set(poems_file='../data/poems_easy.txt', level=1, generate_num=1500)
    output(poems_json_list, '../data/resource_level_1')

    poems_json_list = generate_poems_training_set(poems_file='../data/poems_easy.txt', level=2, generate_num=1500)
    output(poems_json_list, '../data/resource_level_2')

    poems_json_list = generate_poems_training_set(poems_file='../data/poems_easy.txt', level=3, generate_num=1500)
    output(poems_json_list, '../data/resource_level_3')

    # hard poem
    poems_json_list = generate_poems_training_set(poems_file='../data/poems.txt', level=4, generate_num=1500)
    output(poems_json_list, '../data/resource_level_4')

    poems_json_list = generate_poems_training_set(poems_file='../data/poems.txt', level=5, generate_num=1500)
    output(poems_json_list, '../data/resource_level_5')

    poems_json_list = generate_poems_training_set(poems_file='../data/poems.txt', level=6, generate_num=1500)
    output(poems_json_list, '../data/resource_level_6')

    output_to_total('../data/resource_total')





