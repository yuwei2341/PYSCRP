"""Library for Machine Learning """
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
try:
    from sklearn.model_selection import learning_curve
except ImportError as error:
    print "Error: {}. Possibly sklearn too old. Learning Curve not working".format(error)


def sample_df(df, ratio=0.8, seed=11):
    """Sample a df by index """

    random.seed(seed)
    rows = random.sample(df.index, int(len(df) * ratio))
    df_train = df.ix[rows]
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
    n_sample = np.random.choice(len(df_minor), int(len(df_minor) * ratio_sample), replace=True)
    return df_major.append(df_minor.iloc[n_sample])

def get_xy(dff, y_name='y'):
    """Get X and y, and feature names from data frame """

    dfX = dff.drop(y_name, axis=1, errors='ignore')
    y = None if y_name not in dff else dff[y_name].values
    return dfX.values, y, dfX.columns

def normalizer(X, scaler=None):
    """Normalize data to have mean 0 and std 1 
    
    Use provided scaler; if None, scale with the given data
    
    """
    if not scaler:
        scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, scaler

def plot_cor(corrmat):
    """Plot correlation matrix """
    sns.set(context="paper", font="monospace")
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.99, linewidths=0, square=True)
    f.tight_layout()

def plot_importance(importances, feature_names, ax=None):
    """Plot feature importance, ordered"""

    sorted_idx = np.argsort(importances)
    feature_names = np.array(feature_names)[sorted_idx]
    #fig = plt.figure(figsize=(8, len(feature_names) / 4))
    ax = ax or plt.subplots(figsize=(8, len(feature_names) / 4))[1]
    
    ax.barh(range(len(importances)), importances[sorted_idx], alpha=0.3, lw=0)
    plt.yticks(np.arange(len(importances)) + 0.5, feature_names)
    plt.title('Feature Importance')
    plt.show()


def plot_roc(score, y_test, ax=None):
    """Compute ROC curve """

    fpr, tpr, _ = metrics.roc_curve(y_test, score)
    roc_auc = metrics.roc_auc_score(y_test, score)

    # Plot of a ROC curve for a specific class
    ax = ax or plt.subplots()[1]
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
