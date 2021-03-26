import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def combine_ind_label(index_path, label_path, save_path):
    """match and combine indexes and labels"""
    index1 = pd.read_pickle(os.path.join(index_path, "index_train.pkl"))
    index2 = pd.read_pickle(os.path.join(index_path, "index_test.pkl"))
    label1 = pd.read_pickle(os.path.join(label_path, "label_1.pkl"))
    label2 = pd.read_pickle(os.path.join(label_path, "label_2.pkl"))

    print("index1.shape: ", index1.shape)
    print("index2.shape: ", index2.shape)
    print("label1.shape: ", label1.shape)
    print("label2.shape: ", label2.shape)

    # save_path = os.path.join(
    #     "/home/zhendi/wei/plot_data", seqtype, colName+"_plot.pkl"
    # )

    ratio1 = label1.iloc[:, 0]-label1.iloc[:, 1]
    ratio2 = label2.iloc[:, 0]-label2.iloc[:, 1]

    if index1.shape[0] == label1.shape[0]:
        index1 = index1.reset_index(drop=True)
        index1[colName] = np.array(ratio1)
        index1 = index1.set_index("index")

        index2 = index2.reset_index(drop=True)
        index2[colName] = np.array(ratio2)
        index2 = index2.set_index("index")
    else:
        index1 = index1.reset_index(drop=True)
        index1[colName] = np.array(ratio2)
        index1 = index1.set_index("index")

        index2 = index2.reset_index(drop=True)
        index2[colName] = np.array(ratio1)
        index2 = index2.set_index("index")

    frames = [index1, index2]
    result = pd.concat(frames)
    pd.to_pickle(result, save_path)
    print("Saved in: ", save_path)


if __name__ == "__main__":

    # seqtype = "star_RNAseq"
    seqtype = "bwa_scRNA"
    label = "count_overlap"

    if seqtype == "star_RNAseq":
        root = "/data/zhendi/wei/spline-NN"
        datalist = [
            # index in MLP folder
            "ERR188021", "ERR188204", "ERR188052", "ERR188276", "ERR188334",
            "ERR188153", "ERR188088", "ERR188145", "ERR188288", "ERR188295", "ERR188297",
            "ERR188329", "ERR188356", "ERR188382", "ERR188343", "ERR188402", "ERR188479",
            "ERR188436", "ERR188258", "ERR188132", "ERR188114", "ERR188192", "ERR188317",
            # index in CNN folder
            "ERR188408", "ERR188155", "ERR188345", "ERR188347",
            "ERR188265", "ERR188453", "ERR188353"
        ]
    elif seqtype == "bwa_scRNA":
        root = "/data/zhendi/wei/spline-NN-re"
        datalist = ["SRR5968867", "SRR5968879", "SRR5968905", "SRR5968940", "SRR5968873", "SRR5968875",
                    "SRR5968887", "SRR5968939", "SRR5968906", "SRR5959996", "SRR5968871", "SRR5968872",
                    "SRR5968889", "SRR5968890", "SRR5968891", "SRR5968881", "SRR5968894", "SRR5968884",
                    "SRR5968878", "SRR5968893", "SRR5968885", "SRR5968895", "SRR5968896",
                    "SRR5968901", "SRR5968897", "SRR5968898", "SRR5968900", "SRR5968892", "SRR5968869",
                    "SRR5968909"]

    for i in datalist:
        print("Now start: ", i)
        data = i
        # data = "SRR5968879_s"
        if seqtype == "star_RNAseq":
            colName = re.search(r"(^.*?)Aligned.", data).group(1)
        elif seqtype == "bwa_scRNA":
            colName = re.search(r"(^.*?)_", data).group(1)
        print("colName")

        index_folder = "MLP"
        modelname = "CNN"

        index_path = os.path.join(
            root, seqtype, data, label, index_folder
        )

        label_path = os.path.join(
            root, seqtype, data, label, modelname
        )

        save_path = os.path.join(
            "/home/zhendi/wei/plot_data", seqtype, modelname, colName+"_plot.pkl"
        )

        # run function
        combine_ind_label(index_path, label_path, save_path)
