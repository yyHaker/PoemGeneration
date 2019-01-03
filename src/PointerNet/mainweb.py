# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
部署到服务器上！
the module supply some web service.
1) 传入九个字的序列，得到输出的五个字序列
"""
import torch
import json
from torch.autograd import Variable
from PointerNet import PointerNet
from Data_Generator import PoemDataSet
from flask import Flask
from flask import jsonify, request
app = Flask(__name__)


@app.route("/getSeq", methods=['POST', 'GET'])
def getSeq(seq=['隣', '雨', '晦', '月', '照', '连', '苦', '甘', '日']):
    """
    传入seq, 得到输出的五个字的序列
    :param seq:  ['a', 'b', 'c', 'd', 'e', ....]
    :return:  index [8, 5, 6, 4, 3]
    """
    error = None
    if request.method == 'POST':
        seq = list(request.form['seq'])
    else:
        error = 'Invalid Sequence'
        return error
    # load model to predict
    print("loading word2id and id2word...")
    test_dataSet = PoemDataSet(filename='data/resources_total.json')
    word2id, id2word = test_dataSet.word2id, test_dataSet.id2word
    print("load the model......")
    # print(word2id)
    model = PointerNet(embedding_dim=100,
                       hidden_dim=512,
                       lstm_layers=1,
                       dropout=0.5,
                       vocab_size=len(test_dataSet.id2word),
                       init_embedding_weight=None,
                       bidir=True)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    state_dict = torch.load("models/model", map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    # prepare data
    # print("current seq: ", seq)
    seq_in = [int(word2id[s]) for s in seq]
    sample = Variable(torch.LongTensor(seq_in)).unsqueeze(0)
    if use_cuda:
        sample = sample.cuda()
    print("begin predicting.......")
    o, p = model(sample)
    prediction = p.data.numpy()[0]

    train_ch = seq
    pointers_ch = [train_ch[i] for i in prediction]
    # print("source seq: ", train_ch, "pointers: ", pointers_ch)
    # print("prediction:", str(prediction))
    return jsonify({"prediction": str(prediction)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)
