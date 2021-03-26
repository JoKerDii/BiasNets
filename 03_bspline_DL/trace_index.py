"""
# Running example
    
    # python RNAseq_index_copy.py --file=ERR188453Aligned.sortedByCoord.out --readlength=163 --label=count_overlap --model=model 

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
parser.add_argument("--label", type=str, default="count_5", help="Specify a label :D")
parser.add_argument("--model", type=str, default="MLP", help="Choose a model :D")
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
        self.label_path_1 = self.model_path_root + "/" + "label_1.pkl"
        self.label_path_2 = self.model_path_root + "/" + "label_2.pkl"
        self.index_path_1 = self.model_path_root + "/" + "index_train.pkl"
        self.index_path_2 = self.model_path_root + "/" + "index_test.pkl"
        self.metric_path = (
            self.model_path_root
            + "/"
            + self.saved_model_name
            + self.saved_model_time
            + "_metrics.pkl"
        )
        self.metric_path_1 = self.model_path_root + "/" + "_metrics_1.pkl"
        self.metric_path_2 = self.model_path_root + "/" + "_metrics_2.pkl"
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")  # default device is cpu
        self.cuda_num = 3

        # hyperparameters
        self.allow_early_stop = True
        self.dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # dropout
        self.n_epochs_stop = 3  # for early stopping
        self.num_epochs = 2000
        self.target = 1
        self.batch_size = 128
        self.lr = (
            1e-4  # 1e-3 is the best for fullyConnected Model; __ is the best for CNN
        )
        self.momentum = 0.8
        self.filter_sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.filter_num = [16, 32, 64, 128]
        self.stride = [0, 1, 2, 3, 4]
        self.in_channels = 4  # 4 kind of base
        # spline
        self.n_bases = 51
        # self.mode = 'linear'  # 'conv'


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
    df.rename(
        columns={
            "fitpar..i...count": "count_5",
            "fitpar..i...count_overlap": "count_overlap",
        },
        inplace=True,
    )
    df = df.reset_index()
    df = df.astype("int32")

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
    # cube = np.eye(4)[np.asarray(df.iloc[:, 0: 1 + config.readlength])]
    cube = df.iloc[:, 0 : 1 + config.readlength]  # for saving index
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

    # save index
    pd.to_pickle(train_X[["index"]], config.index_path_1)
    pd.to_pickle(test_X[["index"]], config.index_path_2)
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
    if args.model == "MLP":
        model_name = "MLP"
    elif args.model == "CNN":
        model_name = "CNN"
    else:
        model_name = "CNN"
    print("The model name is actually: ", model_name)
    # model_name = args.model  # "MLP"

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
    train_X, test_X, train_y, test_y = get_data(config)

    print("Ended ", file_name, " ,", readlength, " ,", label_name)
