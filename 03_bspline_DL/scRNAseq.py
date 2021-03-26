"""
# Running example
    python RNAseq_test.py --file=ERR188453Aligned.sortedByCoord.out --readlength=163 --label=count_overlap --model=CNN

"""
# rnaseq
# get the splines
import argparse
import math
import os
import pickle  # 5 as pickle
import time

# import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.interpolate as si
import seaborn as sns
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnet as tnt
from PIL import Image
from rpy2.robjects import pandas2ri
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torchvision import transforms

from BSplineActivation import BSplineActivation
from utils import *

pandas2ri.activate()
readRDS = robjects.r["readRDS"]

# set seeds so the results are the same
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description="BiasNets")
parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="Provide one sample :D",
)
parser.add_argument(
    "--readlength", type=int, default=1, help="Provide a matched read length :D"
)
parser.add_argument("--label", type=str, default="count_5",
                    help="Specify a label :D")
parser.add_argument("--model", type=str, default="MLP",
                    help="Choose a model :D")
args = parser.parse_args()


class Config(object):

    """parameters"""

    def __init__(self, dataset_path, model_path_root, model_name):
        # information
        self.model_name = model_name  # model name
        self.dataset_path = dataset_path  # data path
        self.readlength = 123  # 123  # 162
        self.label_name = "count_overlap"
        self.saved_model_name = (
            self.model_name + "_best99"
        )  # for saving figures and pickles
        self.saved_model_time = "0221"

        # for saving labels
        self.save_mode = "prediction"  # or "metrics"
        self.label_iter = 1

        # saving model path
        self.model_path_root = model_path_root
        self.model_path = (
            self.model_path_root
            + "/"
            + self.saved_model_name
            + self.saved_model_time
            + "_model.pkl"
        )
        self.label_path_1 = (
            self.model_path_root
            + "/"
            + "label_1.pkl"
        )
        self.label_path_2 = (
            self.model_path_root
            + "/"
            + "label_2.pkl"
        )
        self.metric_path = (
            self.model_path_root
            + "/"
            + self.saved_model_name
            + self.saved_model_time
            + "_metrics.pkl"
        )
        self.metric_path_1 = (
            self.model_path_root
            + "/"
            + "_metrics_1.pkl"
        )
        self.metric_path_2 = (
            self.model_path_root
            + "/"
            + "_metrics_2.pkl"
        )
        self.log_path = (
            self.model_path_root
            + "/"
            + self.saved_model_name
            + self.saved_model_time
            + "_params.pkl"
        )
        # saving figures
        self.im_CE_path = (
            self.model_path_root
            + "/"
            + self.saved_model_name
            + self.saved_model_time
            + "_ce.png"
        )

        # utils
        self.visdom = False
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")  # default device is cpu
        self.cuda_num = 3

        # hyperparameters
        self.allow_early_stop = True
        self.dropout = [0, 0.1, 0.2, 0.3, 0.4,
                        0.5, 0.6, 0.7, 0.8, 0.9]  # dropout
        self.n_epochs_stop = 5  # for early stopping
        self.num_epochs = 20
        self.target = 1
        self.batch_size = 128
        self.lr = 1e-4  # 1e-3 is the best for fullyConnected Model; __ is the best for CNN
        self.momentum = 0.8
        self.filter_sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.filter_num = [16, 32, 64, 128]
        self.stride = [0, 1, 2, 3, 4]
        self.in_channels = 4  # 4 kind of base
        # spline
        self.n_bases = 51
        # self.mode = 'linear'  # 'conv'


class CNN_Model(nn.Module):  # without spline
    def __init__(self, config):
        super(CNN_Model, self).__init__()
        self.in_channels = config.in_channels
        self.readlength = config.readlength
        self.filter_num_conv1 = config.filter_num[0]  # 16
        self.filter_size_conv1 = config.filter_sizes[5]  # 4
        self.stride_conv1 = config.stride[1]  # 1
        self.filter_size_maxpool1 = config.filter_sizes[4]  # 4
        self.stride_maxpool1 = config.stride[4]  # 4
        self.output = config.target
        self.dropout = config.dropout[5]  # 0.3
        self.first_hidden = self.readlength - self.filter_size_conv1 + 1  # 119  # 158
        # self.batchNorm1 = torch.nn.BatchNorm1d(
        #     self.first_hidden * self.filter_num_conv1)
        # self.batchNorm2 = torch.nn.BatchNorm1d(1024)
        # self.batchNorm3 = torch.nn.BatchNorm1d(512)
        # self.batchNorm = torch.nn.BatchNorm1d(self.filter_num_conv1)

        # filter size = 4, out = 159; filter size = 2, out = 161; filter size = 6, out =

        self.Conv1 = nn.Conv1d(
            in_channels=self.in_channels,  # 4
            out_channels=self.filter_num_conv1,  # 16
            kernel_size=self.filter_size_conv1,  # 4
            stride=self.stride_conv1,  # 1
        )
        # self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        # self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)

        # self.Maxpool = nn.MaxPool1d(
        #     kernel_size=self.filter_size_maxpool1, stride=self.stride_maxpool1  # 4, 4
        # )
        self.Dropout = nn.Dropout(p=self.dropout)

        self.fc1 = nn.Linear(self.first_hidden *
                             self.filter_num_conv1, 1024)  # rna
        self.fc11 = nn.Linear(self.first_hidden *
                              self.filter_num_conv1, 512)  # scrna
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, self.output)

        self.fc = nn.Linear(self.first_hidden *
                            self.filter_num_conv1, self.output)  # rna
        self.fcc = nn.Linear(1024, self.output)  # scrna

        # self.Linear2 = nn.Linear(925, 919)
        self.relu = torch.nn.ReLU()
        self.bs_af1 = BSplineActivation(
            num_activations=self.filter_num_conv1, size=config.n_bases, mode='conv', device=config.device)

    def forward(self, input):
        x = input.permute(0, 2, 1)  # (N, 4, 162)
        # channel should be at the index = 1: [batch, seq_len, 4] -> [batch, 4, seq_len]
        x = self.Conv1(x)  # (N, 32, 159)
        # output: [batch, filter_num, feature1 = (seq_len - filter_size + 1)/stride ]
        # x = x.unsqueeze(2)  # (N, 32, 1, 159)
        # print("x.shape: ", x.shape)
        x = self.bs_af1(x)  # (N, 32, 1, 159)
        # x = self.relu(x)
        x = x.squeeze(2)  # (N, 32, 159)
        # print("x.shape: ", x.shape)
        # x = self.Maxpool(x)
        # output: [batch, filter_num, feature2 = (feature1 - filter_size + 1)/stride]
        # x = self.Dropout(x)
        # print("x.shape: ", x.shape)
        # (N, 159 * 32 = 2544)
        # x = self.Dropout(x)
        # x = x.view(-1, self.first_hidden * self.filter_num_conv1)
        x = x.reshape(x.shape[0], self.first_hidden * self.filter_num_conv1)
        # print("x.shape: ", x.shape)
        # x = self.fc(x)
        x = self.Dropout(x)
        # x = self.relu(self.fc1(x))
        # x = self.fcc(x)
        # x = torch.exp(self.fc(x))
        x = self.relu(self.fc(x))
        x = x.squeeze(1)  # should be [batch,] not [batch, 1]

        return x


class FullyConnected_Model(nn.Module):  # with spline
    def __init__(self, config):
        super(FullyConnected_Model, self).__init__()
        self.in_channels = config.in_channels  # 4
        self.readlength = config.readlength  # 162
        self.hidden_size1 = 256
        self.hidden_size2 = 128
        self.hidden_size3 = 64
        self.output = config.target  # 1
        self.dropout = torch.nn.Dropout(config.dropout[1])
        # self.batchNorm1 = torch.nn.BatchNorm1d(self.hidden_size1)
        # self.batchNorm2 = torch.nn.BatchNorm1d(self.hidden_size2)
        # self.batchNorm3 = torch.nn.BatchNorm1d(self.hidden_size3)

        self.fc1 = nn.Linear(
            self.readlength * self.in_channels, self.hidden_size1)  # 648 -> 256
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)  # -> 128
        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)  # -> 64
        self.fc4 = nn.Linear(self.hidden_size3, self.output)  # ->1

        self.fc = nn.Linear(self.hidden_size1, self.output)

        self.relu = torch.nn.ReLU()
        self.bs_af1 = BSplineActivation(
            num_activations=self.hidden_size1, mode='linear', device=config.device)
        self.bs_af2 = BSplineActivation(
            num_activations=self.readlength * self.in_channels, mode='linear', device=config.device)

    def forward(self, input):
        x = input.view(input.shape[0], -1)
        x = self.bs_af1(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = torch.exp(self.fc4(x))
        x = self.relu(self.fc(x))
        # x = torch.exp(self.fc(x))
        x = x.squeeze(1)  # should be [batch,] not [batch, 1]
        return x


def training(config, model, train_loader, valid_loader, test_loader):

    # Returns lists of metrics
    train_losses = []  # saving training CE loss
    valid_losses = []  # saving testing CE loss

    # For early stopping
    min_val_loss = np.Inf
    epochs_no_improve = 0
    # n_epochs_stop = 6
    early_stop = False

    print("Training...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))

        model.train()
        avg_tr_loss = tnt.meter.AverageValueMeter()
        sum_tr_loss = 0
        total_tr_preds = np.array([])
        for i, batch in enumerate(train_loader):

            batch = [r.to(config.device) for r in batch]
            x_batch, y_batch = batch
            y_pred = model(x_batch)
            # y_pred[y_pred < 0] = 0  # add
            loss = loss_fn(y_pred, y_batch)
            # print("training loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_tr_preds = np.append(
                total_tr_preds, y_pred.detach().cpu().numpy())
            avg_tr_loss.add(loss.item())
            sum_tr_loss += loss.item()

        avg_ts_loss, sum_ts_loss = evaluate(config, model, valid_loader)
        scheduler.step(avg_ts_loss)
        ### print out ###
        # print(f"Epoch: {epoch + 1}")
        print(
            f"\tAvg Loss: {avg_tr_loss.value()[0]:.4f}(train)\t|\tAvg Loss: {avg_ts_loss:.4f}(valid)"
        )
        print(
            f"\tTotal Loss: {sum_tr_loss:.4f}(train)\t|\tTotal Loss: {sum_ts_loss:.4f}(valid)"
        )

        ### vis ###
        if config.visdom == True:
            import visdom
            from visualize import Visualizer

            vis.text(
                f"Epoch: {epoch + 1} \n \tLoss: {avg_tr_loss:.4f}(train)\t|\tLoss: {avg_ts_loss:.4f}(valid)",
                win=config.model_name,
            )

            vis.plot("Training Loss", avg_tr_loss)
            vis.plot("Testing Loss", avg_ts_loss)

        ### save ###
        # CE loss, validation accuracy, log loss, f1 score
        train_losses.append(avg_tr_loss.value()[0])
        valid_losses.append(avg_ts_loss)

        if avg_ts_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = avg_ts_loss
            if config.saved_model_name != None:
                torch.save(model.state_dict(), config.model_path)
        else:
            epochs_no_improve += 1
        # print('epochs_no_improve: ', epochs_no_improve)
        # early stopping
        if config.allow_early_stop == True:
            if epoch > 8 and epochs_no_improve >= config.n_epochs_stop:
                print("Stopping >///<")
                early_stop = True

            if early_stop:
                print("Stopped >///<")
                break

    end = time.time()
    timer(start_time, end)

    # plots CE, acc, logloss, and f1score curves
    plot_CE_graph(config, train_losses, valid_losses)

    # using current model weights on all validation set
    metric_frame = final_test(config, model, test_loader)
    return train_losses, valid_losses, metric_frame


def final_test(config, model, test_loader):
    # test
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    # test on validation sets and report metrics
    (
        total_ts_labels,
        total_ts_preds,
        avg_ts_loss,
    ) = evaluate(config, model, test_loader, test=True)

    # total_ts_preds[total_ts_preds < 0] = 0
    total_ts_preds = pd.DataFrame(total_ts_preds)
    total_ts_labels = pd.DataFrame(total_ts_labels)

    # metrics
    print("Final Test...")
    print("total_ts_labels[:5]: ", total_ts_labels[:5])
    print("total_ts_preds[:5]: ", total_ts_preds[:5])
    metric_frame = regression_report(config, total_ts_labels, total_ts_preds)

    # test on training data and save predictions
    total_ts_labels["preds"] = total_ts_preds
    if config.label_iter == 1:  # train
        total_ts_labels.to_pickle(config.label_path_1)
        metric_frame.to_pickle(config.metric_path_1)
        print("Label saved: ", config.label_iter)
    else:
        total_ts_labels.to_pickle(config.label_path_2)
        print("Label saved: ", config.label_iter)
        metric_frame.to_pickle(config.metric_path_2)
    return metric_frame


def evaluate(config, model, valid_loader, test=False):
    model.eval()

    total_ts_labels = np.array([])
    total_ts_preds = np.array([])

    avg_ts_loss = tnt.meter.AverageValueMeter()
    sum_ts_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            batch = [r.to(config.device) for r in batch]
            x_batch, y_batch = batch
            y_pred = model(x_batch)
            # y_pred[y_pred < 0] = 0  # add
            loss = loss_fn(y_pred, y_batch)
            avg_ts_loss.add(loss.item())
            sum_ts_loss += loss.item()
            total_ts_labels = np.append(
                total_ts_labels, y_batch.detach().cpu().numpy()
            )  # (batch_size, 1)
            total_ts_preds = np.append(
                total_ts_preds, y_pred.detach().cpu().numpy()
            )  # (batch_size, 9)

    if test == True:
        return (
            total_ts_labels,
            total_ts_preds,
            avg_ts_loss.value()[0],
        )
    else:
        return avg_ts_loss.value()[0], sum_ts_loss


def get_data(config):
    """
    label_name: "count_5" or "count_overlap"
    """
    print("Getting data...")

    # load data
    df = readRDS(config.dataset_path)
    # down-sampling
    # df = df.sample(frac=0.8, replace=False, random_state=42)
    print("df.shape: ", df.shape)
    df.rename(columns={
        "fitpar..i...count": "count_5",
        "fitpar..i...count_overlap": "count_overlap",
    }, inplace=True)
    df = df.reset_index()
    df = df.astype('int32')

    # save demo
    # df.to_pickle("/home/zhendi/wei/scripts/baseline/rnaseq021_0.1.pkl")
    # df.to_pickle("/home/zhendi/wei/scripts/baseline/scrna8867_1.pkl")
    # df.to_pickle("/home/zhendi/wei/scripts/baseline/scrna8867_0.01.pkl")

    # a demo
    # df = pd.read_pickle("/home/zhendi/wei/scripts/baseline/rnaseq021_0.1.pkl")
    # df = pd.read_pickle("/home/zhendi/wei/scripts/baseline/scrna8867_1.pkl")
    # df = pd.read_pickle("/home/zhendi/wei/scripts/baseline/scrna8867_0.01.pkl")

    # filter
    df = df.loc[df[config.label_name] >= 30]
    # df = df.sample(frac=0.1, replace=False, random_state=42)
    print("Sample Size after filtering: ", df.shape[0])

    # one-hot encoding
    cube = np.eye(4)[np.asarray(df.iloc[:, 1: 1 + config.readlength])]
    print("Shape of one-hot features: ", cube.shape)

    # append label of interest
    labels = np.array(df[config.label_name]).astype("float32")

    # log count_overlap
    labels = np.log(labels)
    print("type(labels): ", type(labels))

    # split to training set and testing set
    # when
    del df

    if config.save_mode == "prediction":
        train_X, test_X, train_y, test_y = train_test_split(
            cube, labels, test_size=0.5, random_state=42
        )
    else:
        train_X, test_X, train_y, test_y = train_test_split(
            cube, labels, test_size=0.2, random_state=42
        )

    # config.batch_size = int(train_X.shape[0])
    # print(config.train_sample_size)
    # print("batch_size: ", config.batch_size)

    print(
        "Shape of split data: ",
        train_X.shape,
        test_X.shape,
        train_y.shape,
        test_y.shape,
    )
    # Shape of one-hot features:  (181019, 162, 4)
    # Shape of split data:  (144815, 162, 4) (36204, 162, 4) (144815,) (36204,)

    # # load to torch
    # print("Generating data loaders...")
    # train_loader, valid_loader, test_loader = data2torch(
    #     config, train_X, val_X, test_X, train_y, val_y, test_y, config.batch_size)
    # return train_loader, valid_loader  # train_X, test_X, train_y, test_y
    return train_X, test_X, train_y, test_y


def data2torch(config, train_X, val_X, test_X, train_y, val_y, test_y, batch_size):
    # load data to torch
    # load train and test
    x_train = torch.tensor(train_X, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    x_cv = torch.tensor(val_X, dtype=torch.float32)
    y_cv = torch.tensor(val_y, dtype=torch.float32)
    x_test = torch.tensor(test_X, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)
    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)
    test = torch.utils.data.TensorDataset(x_test, y_test)

    if config.device == 'cpu':
        pin_memory = False
    else:
        pin_memory = True
    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=pin_memory
    )
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=pin_memory
    )
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":

    # import pdb; pdb.set_trace()
    # data info
    # "/data/zhendi/wei/star_RNAseq/seq/ERR188021Aligned.sortedByCoord.out.rds"
    device = "cuda"  # 'cuda' or 'cpu'
    # file_name = "ERR188021Aligned.sortedByCoord.out"  # should be assigend here
    # seq = "star_RNAseq"  # should be assiged here
    # root = "/data/zhendi/wei"  # should be fixed here

    # data infor
    # "/data/zhendi/wei/bwa_scRNA/seq/ERR8867.rds"
    file_name = args.file  # "SRR5968867_s"  # should be assigend here
    readlength = args.readlength  # 123  # 162  # should be assigned in config
    # should be assigned in config |"count_overlap"  "count_5"
    label_name = args.label  # "count_overlap"
    if args.model == "model1":
        model_name = "MLP"
    else:
        model_name = "CNN"
    print("The model name is actually: ", model_name)
    # model_name = args.model  # "MLP"

    # RUN
    # /home/zhendi/anaconda3/bin/python RNAseq_test.py --file=SRR5968867_s --readlength=123 --label=count_overlap --model=MLP
    # python RNAseq_test.py --file=SRR5968867_s --readlength=123 --label=count_overlap --model=MLP

    # make paths
    # each RNAseq
    # '/data/zhendi/wei/spline-NN/star_RNAseq/'
    if file_name[:3] == "SRR":
        seq = "bwa_scRNA"  # should be assiged here
    else:
        seq = "star_RNAseq"
    root = "/data/zhendi/wei"  # should be fixed here
    RNASEQ = os.path.join(root, seq, "seq", file_name + ".rds")
    seq_path = os.path.join(root, "spline-NN-re", seq)
    print("seq_path: ", seq_path)
    if not os.path.exists(seq_path):
        os.mkdir(seq_path)
    # each bam file
    file_path = os.path.join(
        seq_path, file_name
    )  # '/data/zhendi/wei/spline-NN/star_RNAseq/ERR188021Aligned.sortedByCoord.out
    print("file_path: ", file_path)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    # each kind of label
    label_path = os.path.join(
        file_path, label_name
    )  # '/data/zhendi/wei/baseline/star_RNAseq/ERR188021Aligned.sortedByCoord.out/count_5
    print("label_path: ", label_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    # each model
    model_path = os.path.join(label_path, model_name)
    print("model_path: ", model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # configuring parameters
    config = Config(RNASEQ, model_path, model_name)
    config.readlength = readlength
    config.save_mode = "prediction"  # "metrics" "prediction"
    config.label_name = label_name

    # set device
    if device == "cuda":
        config.device = torch.device(device)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_num)
        torch.cuda.set_device(config.cuda_num)
        print(torch.cuda.current_device())
    else:
        config.device = torch.device(device)
        print(torch.cuda.current_device())

    if config.visdom == True:
        import visdom
        from visualize import Visualizer

        tfmt = "%m%d_%H%M%S"
        vis = Visualizer(time.strftime(tfmt))

    # load data
    train_X, test_X, train_y, test_y = get_data(
        config)  # get data one time to save time

    # model related
    loss_fn = nn.MSELoss()

    # module
    if config.model_name == "MLP":
        model = FullyConnected_Model(config)
    elif config.model_name == "CNN":
        model = CNN_Model(config)
    else:
        model = CNN_Model(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, verbose=1
    )
    model.to(config.device)
    print(model)

    # training
    if config.save_mode == "prediction":

        # 1. training data
        config.label_iter = 1
        print("Generating data loaders...")

        X_train, X_val, y_train, y_val = train_test_split(
            train_X, train_y, test_size=0.2, random_state=42
        )
        print(X_train.shape, X_val.shape, test_X.shape,
              y_train.shape, y_val.shape, test_y.shape)
        train_loader, valid_loader, test_loader = data2torch(
            config, X_train, X_val, test_X, y_train, y_val, test_y, config.batch_size)

        print("Training ", config.label_iter, " part of data:")
        train_loss, valid_loss, metric_frame = training(
            config, model, train_loader, valid_loader, test_loader
        )
        print("Metric frame of the ", config.label_iter, " part of data:")
        print(metric_frame)

        # module
        if config.model_name == "MLP":
            model = FullyConnected_Model(config)
        elif config.model_name == "CNN":
            model = CNN_Model(config)
        else:
            model = CNN_Model(config)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=8, verbose=1
        )
        model.to(config.device)
        print(model)

        # 2. validation data
        config.label_iter = 2
        X_train, X_val, y_train, y_val = train_test_split(
            test_X, test_y, test_size=0.2, random_state=42
        )

        print(X_train.shape, X_val.shape, train_X.shape,
              y_train.shape, y_val.shape, train_y.shape)

        train_loader, valid_loader, test_loader = data2torch(
            config, X_train, X_val, train_X, y_train, y_val, train_y, config.batch_size)

        print("Training ", config.label_iter, " part of data:")

        train_loss, valid_loss, metric_frame = training(
            config, model, train_loader, valid_loader, test_loader
        )
        print("Metric frame of the ", config.label_iter, " part of data:")
        print(metric_frame)
    else:
        print("Train for reporting metrics.")
        train_loss, valid_loss, metric_frame = training(
            config, model, train_loader, valid_loader, test_loader
        )
        print(metric_frame)

    print("Ended ", file_name, " ,", readlength, " ,", label_name)
