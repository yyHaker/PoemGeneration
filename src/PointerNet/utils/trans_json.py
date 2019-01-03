# -*- coding: utf-8 -*-
import json
import codecs


def trans2json(filename, dest_file):
    """
    transfer the file to json.
    :param filename:
    :param dest_file:
    :return:
    """
    jstr = []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            jstr.append(json.loads(line))
    with codecs.open(dest_file, 'w', encoding='utf-8') as df:
        json.dump(jstr, df, ensure_ascii=False)


if __name__ == "__main__":
    trans2json("../data/resource_total", "../data/resource_total.json")
    trans2json("../data/resource_level_1", "../data/resource_level_1.json")
    trans2json("../data/resource_level_2", "../data/resource_level_2.json")
    trans2json("../data/resource_level_3", "../data/resource_level_3.json")
    trans2json("../data/resource_level_4", "../data/resource_level_4.json")
    trans2json("../data/resource_level_5", "../data/resource_level_5.json")
    trans2json("../data/resource_level_6", "../data/resource_level_6.json")
