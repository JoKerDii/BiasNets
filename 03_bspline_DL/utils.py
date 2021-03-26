
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics


def sum_poisson_deviance(y_true, y_pred):
    """
    y_pred & y_true: pandas data frame or series
    """
    y_pred = y_pred.astype("float")
    y_true = y_true.astype("float")

    eps = 1e-99
    sum_poisson_deviance = (
        2
        * np.add(
            np.multiply(
                np.add(np.log(y_true.clip(eps)), -
                       np.log(y_pred.clip(eps))), y_true
            ),
            np.add(-y_true, y_pred),
        ).sum()
    )
    return sum_poisson_deviance[0]


def mean_poisson_deviance(y_true, y_pred):
    """
    y_pred & y_true: pandas data frame or series
    """
    y_pred = y_pred.astype("float")
    y_true = y_true.astype("float")

    eps = 1e-99

    mean_poisson_deviance = (
        2
        * np.add(
            np.multiply(
                np.add(np.log(y_true.clip(eps)), -
                       np.log(y_pred.clip(eps))), y_true
            ),
            np.add(-y_true, y_pred),
        ).mean()
    )
    return mean_poisson_deviance[0]


def poisson_logloss(y_true, y_pred):
    """
    y_pred & y_true: pandas data frame or series
    """
    import numpy as np
    from scipy.special import gamma

    eps = 1e-99  # all preds are non-negative
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


def regression_report(config, y_true, y_pred):
    """
    For final evaluation on validation set.
    """
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    poi_logloss = poisson_logloss(y_true, y_pred)
    mean_poi_deviance = mean_poisson_deviance(y_true, y_pred)
    sum_poi_deviance = sum_poisson_deviance(y_true, y_pred)

    metric_frame = {}
    metric_frame["explained_variance"] = round(explained_variance, 4)
    metric_frame["mean_squared_log_error"] = round(mean_squared_log_error, 4)
    metric_frame["r2"] = round(r2, 4)
    metric_frame["Mean_AE"] = round(mean_absolute_error, 4)
    metric_frame["Median_AE"] = round(median_absolute_error, 4)
    metric_frame["MSE"] = round(mse, 4)
    metric_frame["RMSE"] = round(np.sqrt(mse), 4)
    metric_frame["poisson_logloss"] = round(poi_logloss, 4)
    metric_frame["mean_poisson_deviance"] = round(mean_poi_deviance, 4)
    metric_frame["sum_poisson_deviance"] = round(sum_poi_deviance, 4)

    metric_frame = pd.DataFrame(
        metric_frame,
        columns=[
            "explained_variance",
            "mean_squared_log_error",
            "r2",
            "Mean_AE",
            "Median_AE",
            "MSE",
            "RMSE",
            "poisson_logloss",
            "mean_poisson_deviance",
            "sum_poisson_deviance",
        ],
        index=["metrics"],
    )
    metric_frame = metric_frame.T
    metric_frame.to_pickle(config.metric_path)

    print("explained_variance: ", round(explained_variance, 4))
    print("mean_squared_log_error: ", round(mean_squared_log_error, 4))
    print("r2: ", round(r2, 4))
    print("Mean AE: ", round(mean_absolute_error, 4))
    print("Median AE: ", round(median_absolute_error, 4))
    print("MSE: ", round(mse, 4))
    print("RMSE: ", round(np.sqrt(mse), 4))
    print("poisson_logloss: ", round(poi_logloss, 4))
    print("mean_poisson_deviance: ", round(mean_poi_deviance, 4))
    print("sum_poisson_deviance: ", round(sum_poi_deviance, 4))
    return metric_frame


def plot_CE_graph(config, train_loss, valid_loss):

    fig = plt.figure(figsize=(12, 12))
    plt.title("Train/Validation Cross Entropy Loss", fontsize=20)
    plt.plot(list(np.arange(len(train_loss)) + 1), train_loss, label="train")
    plt.plot(list(np.arange(len(train_loss)) + 1),
             valid_loss, label="validation")
    plt.xlabel("num_epochs", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="best", fontsize=15)
    fig.savefig(config.im_CE_path)
    plt.show()
