import os
import pickle
import time

import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import seaborn as sns

# from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from rpy2.robjects import pandas2ri


def get_model_data(all_path, df, label_name):
    """
    label_name: "count_5" or "count_overlap"
    """
    # check frequency of labels
    print("check frequency of labels:")
    freq_tab_count = (
        df[label_name].squeeze().value_counts(sort=True).to_frame().reset_index()
    )  # col2 fragments have col1 overlaped count
    print(freq_tab_count)

    # generate 3d one hot features
    cube = np.eye(4)[np.asarray(df.iloc[:, 1 : 1 + readlength])]
    print("Shape of cube features: ", cube.shape)
    # flatten the 3d feature cube to 2d
    flattened_cube = cube.reshape(-1, cube.shape[0]).T
    print("Shape of flattened features: ")
    flattened_cube_df = pd.DataFrame(flattened_cube)

    # append label of interest
    flattened_cube_df[label_name] = df[label_name]
    flattened_cube_df[label_name] = flattened_cube_df[label_name].astype(
        {label_name: "float32"}
    )  # added 1.12
    print("The number cols without variance.")  # added 1.13
    print((flattened_cube_df.var() == 0).sum())  # added 1.13

    # transform pandas frame to h2o frame
    train = h2o.H2OFrame(flattened_cube_df)
    y = label_name
    X = list(train.columns)
    X.remove(y)
    train[y] = train[y].asnumeric()  # added 1.13
    print("Shape of h2o frame: ", train.shape)

    # save tuple data
    datainfo_path = all_path["data_path"] + "/data_info.pkl"
    freq_tab_count.to_pickle(datainfo_path)

    return X, y, train


def training(all_path, model, X, y, train, label_name):
    nFolds = 2

    start = time.time()
    if model == "GLM":
        model_id = "GLM_defaults" + "_" + label_name
        estimator = H2OGeneralizedLinearEstimator(
            seed=1,
            family="poisson",
            model_id=model_id,
            keep_cross_validation_predictions=True,
            keep_cross_validation_fold_assignment=True,
            nfolds=nFolds,
        )
        estimator.train(X, y, train)

    if model == "RF":
        model_id = "RF_defaults" + "_" + label_name
        estimator = H2ORandomForestEstimator(
            seed=1,
            model_id=model_id,
            keep_cross_validation_predictions=True,
            keep_cross_validation_fold_assignment=True,
            distribution="poisson",
            nfolds=nFolds,
        )
        estimator.train(X, y, train)

    if model == "XGB":
        model_id = "XGB_defaults" + "_" + label_name
        estimator = H2OXGBoostEstimator(
            seed=1,
            # backend="gpu",
            # gpu_id=3,
            distribution="poisson",
            model_id=model_id,
            keep_cross_validation_predictions=True,
            keep_cross_validation_fold_assignment=True,
            nfolds=nFolds,
        )
        estimator.train(X, y, train)

    if model == "DL":
        model_id = "MLP_defaults" + "_" + label_name
        estimator = H2ODeepLearningEstimator(
            distribution="poisson",
            seed=1,
            model_id=model_id,
            keep_cross_validation_predictions=True,
            keep_cross_validation_fold_assignment=True,
            nfolds=nFolds,
        )
        estimator.train(X, y, train)

    end = time.time()
    print("Time: ")
    timer(start, end)

    # save estimator
    h2o.download_model(estimator, path=all_path["model_path"])

    # predictions
    y_pred = estimator.cross_validation_holdout_predictions()
    y_true = train[y]
    y_true_pd = h2o.as_list(y_true)
    y_pred_pd = h2o.as_list(y_pred)

    # calculating extra metrics
    poi_logloss = poisson_logloss(y_pred_pd, y_true_pd)
    mean_poi_deviance = mean_poisson_deviance(y_pred_pd, y_true_pd)
    sum_poi_deviance = sum_poisson_deviance(y_pred_pd, y_true_pd)

    # performance
    pd_frame = estimator.cross_validation_metrics_summary().as_data_frame()
    pd_frame = pd_frame.rename({"": "metrics"}, axis="columns")
    new_poi_logloss = {
        "metrics": "poisson cross entropy",
        "mean": poi_logloss,
        "sd": float("NaN"),
        "cv_1_valid": float("NaN"),
        "cv_2_valid": float("NaN"),
    }
    new_mean_poi_deviance = {
        "metrics": "mean poisson deviance",
        "mean": mean_poi_deviance,
        "sd": float("NaN"),
        "cv_1_valid": float("NaN"),
        "cv_2_valid": float("NaN"),
    }
    new_sum_poi_deviance = {
        "metrics": "sum poisson deviance",
        "mean": sum_poi_deviance,
        "sd": float("NaN"),
        "cv_1_valid": float("NaN"),
        "cv_2_valid": float("NaN"),
    }

    pd_frame = pd_frame.append(new_poi_logloss, ignore_index=True)
    pd_frame = pd_frame.append(new_mean_poi_deviance, ignore_index=True)
    pd_frame = pd_frame.append(new_sum_poi_deviance, ignore_index=True)

    # save performance frame
    metri_path = all_path["model_path"] + "/" + model + "_metri.pkl"
    pd_frame.to_pickle(metri_path)

    y_true_pd["y_pred"] = y_pred_pd
    # save predictions and true labels
    y_path = all_path["model_path"] + "/" + model + "_ys.pkl"
    y_true_pd.to_pickle(y_path)


def sum_poisson_deviance(y_pred, y_true):
    """
    y_pred & y_true: pandas data frame or series [convert h2o to pandas: h2o.to_list(y_pred)]
    """
    y_pred = y_pred.astype("float")
    y_true = y_true.astype("float")

    eps = 1e-99
    sum_poisson_deviance = (
        2
        * np.add(
            np.multiply(
                np.add(np.log(y_true.clip(eps)), -np.log(y_pred.clip(eps))), y_true
            ),
            np.add(-y_true, y_pred),
        ).sum()
    )
    return sum_poisson_deviance[0]


def mean_poisson_deviance(y_pred, y_true):
    """
    y_pred & y_true: pandas data frame or series [convert h2o to pandas: h2o.to_list(y_pred)]
    """
    y_pred = y_pred.astype("float")
    y_true = y_true.astype("float")

    eps = 1e-99
    # mean_poisson_deviance = (
    #     2
    #     * np.add(
    #         np.multiply(np.log(np.divide(y_true, y_pred).clip(eps)), y_true),
    #         np.add(-y_true, y_pred),
    #     ).mean()
    # )
    mean_poisson_deviance = (
        2
        * np.add(
            np.multiply(
                np.add(np.log(y_true.clip(eps)), -np.log(y_pred.clip(eps))), y_true
            ),
            np.add(-y_true, y_pred),
        ).mean()
    )
    return mean_poisson_deviance[0]


def poisson_logloss(y_pred, y_true):
    """
    y_pred & y_true: pandas data frame or series [convert h2o to pandas: h2o.to_list(y_pred)]
    """
    import numpy as np
    from scipy.special import gamma

    eps = 1e-15  # all preds are non-negative
    y_pred = y_pred.clip(eps)
    p_logloss = np.mean(
        np.add(
            np.add(np.log(gamma((y_true).clip(upper=170) + 1)), y_pred),
            -np.multiply(np.log(y_pred), y_true),
        )
    )
    return p_logloss[0]


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == "__main__":

    # candidates
    run_three = {}
    run_three["SRR5968867_s"] = 123
    run_three["SRR5959996_s"] = 214
    run_three["SRR5968940_s"] = 175

    # run_three["ERR188356Aligned.sortedByCoord.out"] = 198

    # loop all candidates
    for file_name, readlength in run_three.items():
        print(file_name, str(readlength))

        total_time_start = time.time()

        # parameters

        ### scRNA seq
        # file_name = "SRR5968905_s"
        # dir = "bwa_scRNA"
        # RNASEQ = "/data/zhendi/wei/" + dir + "/seq/" + file_name + ".rds"

        ### RNA seq
        # file_name = "ERR188288Aligned.sortedByCoord.out"
        # readlength = 205
        seq = "bwa_scRNA"  # "star_RNAseq"
        root = "/data/zhendi/wei"
        RNASEQ = os.path.join(root, seq, "seq", file_name + ".rds")
        models = ["GLM", "RF", "XGB", "DL"]
        labels = ["count_5", "count_overlap"]

        # load data
        start = time.time()
        pandas2ri.activate()
        readRDS = robjects.r["readRDS"]
        mydf = readRDS(RNASEQ)
        mydf = mydf.sample(frac=0.01, replace=True, random_state=1)
        mydf.rename(
            columns={
                "fitpar..i...count": "count_5",
                "fitpar..i...count_overlap": "count_overlap",
            },
            inplace=True,
        )
        mydf = mydf.reset_index()
        mydf = mydf.astype("int32")
        end = time.time()

        # paths
        # seq_path: RNAseq or scRNAseq
        seq_path = os.path.join(
            root, "baseline", seq
        )  # '/data/zhendi/wei/baseline/star_RNAseq/'
        print("seq_path: ", seq_path)
        if not os.path.exists(seq_path):
            os.mkdir(seq_path)

        # save_path: which sample
        save_path = os.path.join(
            seq_path, file_name
        )  # '/data/zhendi/wei/baseline/star_RNAseq/ERR188021Aligned.sortedByCoord.out
        print("save_path: ", save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        all_path = {}
        all_path["seq_path"] = seq_path
        all_path["save_path"] = save_path

        # init h2o
        h2o.init(ip="localhost", port=54321, nthreads=15)

        # training
        for label in labels:
            # paths
            # label_path: modeling count_5 or count_overlap
            label_path = os.path.join(
                save_path, label
            )  # '/data/zhendi/wei/baseline/star_RNAseq/ERR188021Aligned.sortedByCoord.out/count_5
            print("label_path: ", label_path)
            if not os.path.exists(label_path):
                os.mkdir(label_path)

            # data_path: tuple of (X, y, train)
            data_path = os.path.join(
                label_path, "data"
            )  # '/data/zhendi/wei/baseline/star_RNAseq/ERR188021Aligned.sortedByCoord.out/count_5/data
            print("data_path: ", data_path)
            if not os.path.exists(data_path):
                os.mkdir(data_path)

            all_path["label_path"] = label_path
            all_path["data_path"] = data_path

            # get data
            print("Getting training data:")
            X, y, train = get_model_data(all_path, mydf, label)

            for model in models:

                # path
                # model_path: GLM, RF, XGB, DL, saving estimator, metrics, and labels
                model_path = os.path.join(label_path, model)
                print("model_path: ", model_path)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)

                all_path["model_path"] = model_path

                # training
                print("Training: Label: " + label + " Model: " + model)
                training(all_path, model, X, y, train, label)

        # total time
        total_time_end = time.time()
        timer(total_time_start, total_time_end)

        # shutdown h2o
        h2o.shutdown()
