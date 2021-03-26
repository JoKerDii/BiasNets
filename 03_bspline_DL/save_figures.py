import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_table(datalist, root, seqtype, modelname):
    """generate one table for all samples and save it"""
    for i, f in enumerate(datalist):
        filename = f + "_plot.pkl"
        path = os.path.join(
            root, seqtype, modelname, filename
        )
        print("Path: ", path)
        if i == 0:
            myfile = pd.read_pickle(path)
        else:
            myfile = pd.concat([myfile, pd.read_pickle(path)], axis=1)
    pd.to_pickle(myfile, os.path.join(
        root, seqtype, modelname, "allsamples.pkl"
    ))


def plotting(myfile, path):
    """plot one lineplot for all samples"""
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 6)
    sns.lineplot(data=myfile, dashes=False, alpha=0.2)
    ax.set(ylim=(-4, 10), xlim=(0, max(myfile.index) + 100000),
           xlabel='transcript', ylabel='log(count/count_pred)')
    plt.legend(bbox_to_anchor=(1.001, 1), loc=2, borderaxespad=0.)
    # plt.show()
    fig.savefig(path)


if __name__ == "__main__":

    root = "/home/zhendi/wei/plot_data"

    for seqtype in ["star_RNAseq", "bwa_scRNA"]:
        print(seqtype)
       if seqtype == "star_RNAseq":
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
            datalist = ["SRR5968867", "SRR5968879", "SRR5968905", "SRR5968940", "SRR5968873", "SRR5968875",
                        "SRR5968887", "SRR5968939", "SRR5968906", "SRR5959996", "SRR5968871", "SRR5968872",
                        "SRR5968889", "SRR5968890", "SRR5968891", "SRR5968881", "SRR5968894", "SRR5968884",
                        "SRR5968878", "SRR5968893", "SRR5968885", "SRR5968895", "SRR5968896",
                        "SRR5968901", "SRR5968897", "SRR5968898", "SRR5968900", "SRR5968892", "SRR5968869",
                        "SRR5968909"]

        for modelname in ["CNN", "MLP"]:
            print(modelname)
            # generate table
            generate_table(datalist, root, seqtype, modelname)

            # get table
            myfile = pd.read_pickle(os.path.join(
                root, seqtype, modelname, "allsamples.pkl"
            ))
            # plot
            path = os.path.join(root, seqtype, modelname, "bigfigure.png")

            print("Start to plot...")
            plotting(myfile, path)
            print("Ended and saved in:", path)
