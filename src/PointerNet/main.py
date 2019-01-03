# -*- coding: utf-8 -*-
"""
Pytorch implementation of Pointer Network.
http://arxiv.org/pdf/1506.03134v1.pdf.
"""
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from PointerNet import PointerNet
from Data_Generator import PoemDataSet
from tensorboardX import SummaryWriter
from utils import load_wordvec


def evaluate(model, valid_dataloader, criterion, params, use_cuda=False):
    if use_cuda:
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    losses = []
    accs = []
    for i_batch, sample_batched in enumerate(valid_dataloader):
        train_batch = Variable(sample_batched['Points'])
        target_batch = Variable(sample_batched['Solution'])

        if use_cuda:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model(train_batch)

        # calc accuracy, acc = accuracy(p, target_batch)
        res = torch.abs(p - target_batch)
        res = torch.sum(res, 1).eq(torch.tensor([0]).type_as(res))
        acc = float(res.sum()) / float(res.size()[0])
        # print("acc.sum(): {}, float(res.size()[0]): {}, acc: {}".format(res.sum(), float(res.size()[0]), acc))
        accs.append(acc)

        o = o.contiguous().view(-1, o.size()[-1])  # [bz, seq_len]
        target_batch = target_batch.view(-1)  # [bz, seq_len]
        loss = criterion(o, target_batch)
        losses.append(loss.item())

    # calc accuracy and loss
    accuracy = np.average(accs)
    loss = np.average(losses)
    return loss, accuracy


def train(params, use_cuda=False):
    print("loading train data.....")
    train_dataset = PoemDataSet(filename=params.train_data)
    print("train data size is: ", len(train_dataset))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=True, num_workers=4)

    print("loading valid data....")
    valid_dataset = PoemDataSet(filename=params.valid_data)
    print("valid data size is: ", len(valid_dataset))
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=True, num_workers=4)

    # print("loading word vectors....")
    # weight = load_wordvec(params.wordvec_path, train_dataset.word2id,word_dim=params.embedding_size)

    print("create the model....")
    model = PointerNet(params.embedding_size,
                       params.hiddens,
                       params.nof_lstms,
                       params.dropout,
                       vocab_size=len(train_dataset.id2word),
                       init_embedding_weight=None,
                       bidir=params.bidir)
    # summary writer
    writer = SummaryWriter(log_dir=params.log_dir)

    if use_cuda:
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    CCE = torch.nn.CrossEntropyLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()), lr=params.lr)

    print("begin training....")
    losses = []
    best_valid_acc = 0.0
    for epoch in range(params.nof_epoch):
        batch_loss = []
        iterator = tqdm(train_dataloader, unit='Batch')

        for i_batch, sample_batched in enumerate(iterator):
            iterator.set_description('Batch %i/%i' % (epoch + 1, params.nof_epoch))

            train_batch = Variable(sample_batched['Points'])
            target_batch = Variable(sample_batched['Solution'])

            if use_cuda:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()

            o, p = model(train_batch)
            o = o.contiguous().view(-1, o.size()[-1])  # [bz, seq_len]

            target_batch = target_batch.view(-1)  # [bz, seq_len]

            loss = CCE(o, target_batch)

            losses.append(loss.item())
            batch_loss.append(loss.item())

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            iterator.set_postfix(loss='{}'.format(loss.item()))
        # every epoch write the loss
        writer.add_scalar('train/loss', scalar_value=np.average(batch_loss), global_step=epoch)
        iterator.set_postfix(loss=np.average(batch_loss))

        # evaluate the model
        valid_loss, valid_accuracy = evaluate(model, valid_dataloader, CCE, params, use_cuda=use_cuda)
        print("valid: epoch {}, loss {}, accuracy {}%".format(epoch, valid_loss, valid_accuracy * 100))
        writer.add_scalar("valid/loss", scalar_value=valid_loss, global_step=epoch)
        writer.add_scalar("valid/accuracy", scalar_value=valid_accuracy, global_step=epoch)
        # save the model
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            torch.save(model.state_dict(), params.save_model_path)


def predict(parmas, use_cuda=False):
    print("loading test data....")
    test_dataSet = PoemDataSet(filename=params.test_data)
    test_DataLoader = DataLoader(dataset=test_dataSet, batch_size=1,
                                 shuffle=False, num_workers=4)
    print("test data size: ", len(test_dataSet))
    word2id, id2word = test_dataSet.word2id, test_dataSet.id2word
    print("load the model......")
    model = PointerNet(params.embedding_size,
                       params.hiddens,
                       params.nof_lstms,
                       params.dropout,
                       len(test_dataSet.id2word),
                       None,
                       params.bidir)
    if use_cuda:
        model.cuda()
    state_dict = torch.load(params.save_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    print("begin predicting.......")
    for i_batch, sample_batched in enumerate(test_DataLoader):
        train_batch = Variable(sample_batched['Points'])
        target_batch = Variable(sample_batched['Solution'])

        if use_cuda:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model(train_batch)
        # train/target/prediction
        train = train_batch.data.numpy()[0]
        target = target_batch.data.numpy()[0]
        prediction = p.data.numpy()[0]

        train_ch = test_dataSet.idList2str(train)
        target_ch = [train_ch[i] for i in target]
        pointers_ch = [train_ch[i] for i in prediction]
        print("source seq: ", train_ch, "target:", target_ch, "pointers: ", pointers_ch)


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")
    # mode (train, evaluate, predict)
    parser.add_argument("--train", default=True, help="train mode")
    parser.add_argument("--evaluate", default=False, help="evaluate mode")
    parser.add_argument("--predict", default=False, help="predict mode")
    # Data
    parser.add_argument("--train_data", default='./data/resource_total', help="train data path")
    parser.add_argument("--valid_data", default='./data/resource_total', help="valid data path")
    # predict mode
    parser.add_argument("--test_data", default='./data/resources_total', help='test data path')

    # word vectors
    parser.add_argument("--wordvec_path", default='wordvect/data/embedding_model_t2s/vector_t2s', help='word vector path')

    # parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
    # parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
    # parser.add_argument('--test_size', default=10000, type=int, help='Test data size')

    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    # Train
    parser.add_argument('--nof_epoch', default=500, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # save model path
    parser.add_argument('--save_model_path', default='./models/model', help='save model path')
    # log dir
    parser.add_argument('--log_dir', type=str, default='./logs', help='log dir')
    # GPU
    parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
    # TSP
    # parser.add_argument('--nof_points', type=int, default=5, help='Number of points in poem recognition')
    # Network
    parser.add_argument('--embedding_size', type=int, default=100, help='Embedding size')
    parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--nof_lstms', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
    parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')
    params = parser.parse_args()

    print("params: ", params)

    if params.gpu and torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    if params.train:
        train(params, use_cuda=USE_CUDA)
    elif params.predict:
        predict(params)
