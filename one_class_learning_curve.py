# -*- coding: utf-8 -*-
import time

import numpy as np

from sklearn.base import is_classifier, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import indexable, check_random_state
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _score, _translate_train_sizes,\
    _fit_and_score

def one_class_learning_curve(
    estimator, X, y, groups=None,
    train_sizes=np.linspace(0.1, 1.0, 5), cv=None, scoring=None,
    n_jobs=1,
    pre_dispatch="all", verbose=0, shuffle=False,
    random_state=None):
    """One-class learning curve.
    Determines cross-validated training and test scores for different one-class
    training set sizes.  This should help choosing the best downsampling ratio.
    A cross-validation generator splits the whole dataset k times in training
    and test data. Subsets of the training set with varying sizes will be used
    to train the estimator and a score for each training subset size and the
    test set will be computed. Afterwards, the scores will be averaged over
    all k runs for each training subset size.
    Read more in the :ref:`User Guide <learning_curve>`.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.
    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.
    shuffle : boolean, optional
        Whether to shuffle training data before taking prefixes of it
        based on``train_sizes``.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == 'True'.
    -------
    train_sizes_abs : array, shape = (n_unique_ticks,), dtype int
        Numbers of training examples that has been used to generate the
        learning curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.
    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.
    Notes
    -----
    See :ref:`examples/model_selection/plot_learning_curve.py
    <sphx_glr_auto_examples_model_selection_plot_learning_curve.py>`
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    # Store it as list as we will be iterating over the list multiple times
    cv_iter = list(cv.split(X, y, groups))

    scorer = check_scoring(estimator, scoring=scoring)

    n_max_training_samples = len(cv_iter[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs = _translate_train_sizes(train_sizes,
                                             n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)

    if shuffle:
        rng = check_random_state(random_state)
        cv_iter = ((rng.permutation(train), test) for train, test in cv_iter)

    one_class_sizes = list()

    train_test_proportions = []
    for train, test in cv_iter:
        pos = train[y.iloc[train] == 1]
        for n_train_samples in train_sizes_abs:
            train_split = train[:n_train_samples]
            neg = train_split[y.iloc[train_split] == 0]
            selected = np.concatenate((pos, neg), axis=0)
            train_test_proportions.append((selected, test))
            if len(one_class_sizes) < train_sizes_abs.shape[0]:
                one_class_sizes.append(neg.shape[0])

    out = parallel(delayed(_fit_and_score)(
        clone(estimator), X, y, scorer, train, test,
        verbose, parameters=None, fit_params=None, return_train_score=True)
        for train, test in train_test_proportions)
    out = np.array(out)
    n_cv_folds = out.shape[0] // n_unique_ticks
    out = out.reshape(n_cv_folds, n_unique_ticks, 2)

    out = np.asarray(out).transpose((2, 1, 0))

    return np.array(one_class_sizes), out[0], out[1]
