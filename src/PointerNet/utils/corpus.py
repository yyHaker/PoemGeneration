# -*- coding: utf-8 -*-
"""
make poems corpus.
"""
from __future__ import absolute_import
import json
from urllib import parse, request
from lxml import etree
import re


def getPoemsFreq(json_file, output_file='poems.txt'):
    """
    get poem freq from Baidu search result.
    --------
    json file: dict {'author': , 'paragraphs': , 'title': , 'sents': },
         --> {'author': , 'paragraphs': , 'title': , 'sents': , 'freq': }
    :param json_file: json file.
    :param output_file: output json file.
    :return:
    """
    print("try to get baidu search freq....")
    o_file = open(output_file, 'a', encoding='utf-8')
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            max_freq = 0
            for poem in json_data['sents']:
                freq = getPoemFreqFromBaidu(poem)
                if freq > max_freq:
                    max_freq = freq
            json_data['freq'] = max_freq
            print("processed one!")
            o_file.write(json.dumps(json_data, ensure_ascii=False)+"\n")
    print("process down!")


def getPoemFreqFromBaidu(poem='锄禾日当午'):
    """
    get poem freq from Baidu.
    :param poem: str.
    :return: 'freq', int.
    """
    url = 'http://www.baidu.com/s'
    data = {'wd': poem}
    data = parse.urlencode(data)
    get_url = url + "?" + data
    page = request.urlopen(get_url).read()
    page = page.decode('utf-8')

    # 从返回的网页中解析HTML找到freq
    selector = etree.HTML(page)
    p = selector.xpath(u"//span[@class='nums_text']")
    # print("p: ", p, "length: ", len(p))
    text = p[0].text
    freq = re.findall(r'\d+\.?\d*', text.replace(',', ''))[0]
    print(text+" ", "freq:", int(freq))
    return int(freq)


def sort_poems_by_freq(json_file, output_file):
    """
    sort poems by freq.
    :param json_file:
    :param output_file:
    :return:
    """
    poems_json_data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            json_data = json.loads(line.rstrip('\n'))
            poems_json_data.append(json_data)
    # sort (降序排序)
    print("sort the poem...")
    poems_json_data_sorted = sorted(poems_json_data,
                                    key=lambda e: e.__getitem__('freq'), reverse=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for poem in poems_json_data_sorted:
            f.write(json.dumps(poem, ensure_ascii=False) + "\n")
    print("write done!")


def transfer_json(json_file='b.json', out_file='poems.txt'):
    """
    read b.json , transfer to json file.
    :param json_file:
    :return:
    json file: '{'author': , 'paragraphs': , 'title': , 'sents': }'.
    """
    file = open(out_file, 'a', encoding='utf-8')
    with open(json_file, 'r', encoding='utf-8') as f:
        datas_list = json.load(f)
        for datas in datas_list:
            paragraphs = datas['paragraphs']
            sents = []
            for para in paragraphs:
                if len(para) == 12:
                    sents.append(para[0: 5])
                    sents.append(para[6: 11])
            datas['sents'] = sents
            file.write(str(datas)+"\n")
        print(datas_list)


if __name__ == "__main__":
    # transfer_json()
    # getPoemFreqFromBaidu()
    # getPoemsFreq('../data/poems.txt', '../data/poems.txt')
    sort_poems_by_freq('../data/poems.txt', '../data/poems.txt')
