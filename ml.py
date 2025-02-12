#!/usr/bin/env python3

"""Library for Machine Learning """
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, preprocessing
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve


try:
    from sklearn.model_selection import learning_curve
except ImportError as error:
    print(f"Error: {error}. Possibly sklearn too old. Learning Curve not working")

TRAIN_SIZES = np.linspace(0.1, 1.0, 5)


## Feature transformation
def bin_by_quantile(df, col, quantiles, missing_as=-9999):
    """Bin a column with certain quantiles given
    Note:
     - set missing value as the first bin
     - remove the original col from df
    """
    bins = df[col].quantile(quantiles).round(2).tolist()

    # Create the bin for missing values
    bins.insert(0, missing_as)
    df.fillna({col: missing_as}, inplace=True)
    labels = [f"{i}_{j}" for i, j in zip(bins[:-1], bins[1:])]
    labels[0] = "missing"

    # right=True to include the right end
    df["bin_" + col] = pd.cut(df[col], bins, right=True, labels=labels)
    return df.drop(col, axis=1)


def cap_cat(df, col, max_label, max_cats=6):
    """Cap the number of categories of a column in place.
    - Rank categories by value counts, and keep the top (max_cats - 1)
    - Collapse the rest to the category max_label
    """
    top_levels = df[col].value_counts(dropna=True).index[:max_cats]
    s = df[col].copy()
    s[~s.isin(top_levels)] = max_label
    df[col] = s


def get_xy(dff, y_name="y"):
    """Get X and y, and feature names from data frame"""

    df_x = dff.drop(y_name, axis=1, errors="ignore")
    y = None if y_name not in dff else dff[y_name].values
    return df_x.values, y, df_x.columns.values


def normalizer(X, scaler=None):
    """Normalize data to have mean 0 and std 1

    Use provided scaler; if None, scale with the given data

    """
    if not scaler:
        scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, scaler


def scale_data(X_train, X_test=None, is_scale=True):
    """Unify process to scale or not scale data"""

    if is_scale:
        print("Scaling data")
        X_train_scaled, scaler = normalizer(X_train, scaler=None)
        if X_test is not None:
            X_test_scaled, _ = normalizer(X_test, scaler=scaler)
        else:
            X_test_scaled = X_test
    else:
        print("Not scaling data")
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
    return X_train_scaled, X_test_scaled, scaler


# Data and class sampling
def sample_df(df, ratio=0.8, seed=11):
    """Sample a df by index"""

    random.seed(seed)
    rows = random.sample(df.index.tolist(), int(len(df) * ratio))
    df_train = df.loc[rows]
    df_cv = df.drop(rows)
    return df_train, df_cv


def sampling_imbalanced(df_major, df_minor, seed, ratio_sample=None):
    """Sample to balanced imbalanced data df_major and df_minor, return the combined

    Note:
        If no sampling ratio is given, will use the ratio of # majro data over # minor data,
        so that the resulted new minor data will have comparable size with the major
    Returns:
        Combination of df_major and sampled df_minor

    """

    np.random.seed(seed)
    if ratio_sample is None:
        ratio_sample = len(df_major) / len(df_minor)
    n_sample = np.random.choice(
        len(df_minor), int(len(df_minor) * ratio_sample), replace=True
    )
    return df_major.append(df_minor.iloc[n_sample])


# Plot
def plot_cor(corrmat):
    """Plot correlation matrix"""
    sns.set_theme(context="paper", font="monospace")
    f, _ = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.99, linewidths=0, square=True)
    f.tight_layout()


def plot_importance(importances, feature_names, ax=None):
    """Plot feature importance, ordered"""

    sorted_idx = np.argsort(importances)
    feature_names = np.array(feature_names)[sorted_idx]
    # fig = plt.figure(figsize=(8, len(feature_names) / 4))
    ax = ax or plt.subplots(figsize=(8, len(feature_names) / 4))[1]

    ax.barh(range(len(importances)), importances[sorted_idx], alpha=0.3, lw=0)
    plt.yticks(np.arange(len(importances)) + 0.5, feature_names)
    plt.title("Feature Importance")
    plt.show()
    return importances[sorted_idx], feature_names


def plot_roc(score, y_test, ax=None):
    """Compute ROC curve"""

    fpr, tpr, _ = metrics.roc_curve(y_test, score)
    roc_auc = metrics.roc_auc_score(y_test, score)

    # Plot of a ROC curve for a specific class
    ax = ax or plt.subplots()[1]
    ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


def proba_calibration(y_test, pred_proba):
    """Calibration plots"""

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, pred_proba, n_bins=10
    )

    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax2.hist(pred_proba, range=(0, 1), bins=10, label="Model", histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.show()


def precision_recall_curve_plus(
    y_test,
    score,
    title="",
    pos_label=None,
    upper=None,
    lower=None,
    is_max_f1=True,
    is_plot=True,
):

    upper = upper or round(score.max() * 1.1, 2)
    upper = min(upper, 1)
    lower = lower or round(score.min() * 0.9, 2)

    precision, recall, threshold = precision_recall_curve(
        y_test, score, pos_label=pos_label
    )
    precision = precision[:-1]  # the last one is an extra and is 1
    recall = recall[:-1]  # the last one is an extra and is 0
    f1 = precision * recall * 2 / (precision + recall)
    frac_pred_pos = [(score > ii).mean() for ii in threshold]
    df_metric = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "frac_pred_pos": frac_pred_pos,
        },
        index=threshold,
    )

    # Return only threshold within certain range
    df_metric = df_metric.loc[(df_metric.index >= lower) & (df_metric.index <= upper)]

    # Get argmaxf1 for threshold
    metric_max = df_metric[df_metric["f1"] == df_metric["f1"].max()].iloc[:1, :]
    metric_max.index.name = "threshold"

    if is_plot:
        f, ax = plt.subplots(figsize=(8, 6))
        df_metric.plot(ax=ax)
        xticks_int = 0.1 if upper - lower > 0.6 else 0.05
        xticks = list(np.arange(lower, upper, xticks_int))
        ax.set_xticks(xticks)
        ax.set_title(title)
        if is_max_f1:
            ax.axvline(x=metric_max.index[0], color="black", linestyle="--", lw=0.5)
            ax.text(
                x=metric_max.index[0] + 0.01,
                y=metric_max["f1"] * 1.2,
                s=metric_max.to_string(index=False, float_format="    %.2f"),
            )

    return metric_max, df_metric


choose_threshold = precision_recall_curve_plus  # for backward compatibility


def plot_learning_curve(
    estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=TRAIN_SIZES
):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    Example
    --------
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=random_state)
        estimator = RandomForestClassifier()
        plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
        plt.show()

    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    return plt


def plot_gain(df, n_split=10, y_test_col="y_test", score_col="score"):
    """Plot gain charts
    Input:
        df: dataframe containing true value and predicted score
        n_split: how many buckets to split predictions
        y_test_col: name of true value column
        score: name of predicted score column

    """

    df_score = (
        df[[y_test_col, score_col]]
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )
    df_score["prediction_score_bucket"] = (
        df_score.index / ((len(df_score.index) + 1) / n_split) + 1
    )
    df_score["prediction_score_bucket"] = df_score["prediction_score_bucket"].apply(
        lambda x: min(x, n_split)
    )

    ap = df_score[y_test_col].sum()
    df_quantile = df_score.groupby("prediction_score_bucket")[y_test_col].agg(
        {
            "random chance": lambda x: 1.0 / n_split,
            "model": lambda x: x.sum() * 1.0 / ap,
            "precision": lambda x: x.mean(),
        }
    )
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    df_quantile["precision"].plot(
        kind="bar", title="Precision in prediction score buckets", ax=axs[0]
    )
    axs[0].set_xticklabels(range(1, n_split + 1), rotation=0)
    axs[0].set_ylabel("precision")
    df_quantile.loc[0] = [0, 0, 0]
    df_quantile[["random chance", "model"]].sort_index().cumsum().plot(
        title="Cumulative Gain Chart", ax=axs[1]
    )
    axs[1].set_xticks(range(n_split + 1))
    axs[1].set_ylabel("target population%")
    axs[1].set_xticklabels(range(n_split + 1), rotation=0)
    plt.show()


def tree_to_code(tree, feature_names):
    # Return rules from tree
    from sklearn.tree import _tree

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            rslt = tree_.value[node][0]
            print(
                "{}return {}, prob: {:.2f}".format(
                    indent, rslt, rslt[1] * 1.0 / sum(rslt)
                )
            )

    recurse(0, 1)
