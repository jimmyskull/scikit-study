# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Eric Martin <eric@ericmart.in>
#          Giorgio Patrini <giorgio.patrini@anu.edu.au>
#          Paulo Urio <paulourio gmail>
# License: BSD 3 clause
import warnings

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.six import string_types
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.preprocessing.data import (check_array,
    _handle_zeros_in_scale, _incremental_mean_and_var)


def nanscale(X, axis=0, with_mean=True, with_std=True, copy=True):
    """Standardize a dataset along any axis
    Center to the mean and component wise scale to unit variance.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    Parameters
    ----------
    X : {array-like, sparse matrix}
        The data to center and scale.
    axis : int (0 by default)
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.
    with_mean : boolean, True by default
        If True, center the data before scaling.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSC matrix and if axis is 1).
    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.
    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.
    To avoid memory copy the caller should pass a CSC matrix.
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    See also
    --------
    StandardScaler: Performs scaling to unit variance using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
    """  # noqa
    X = check_array(X, accept_sparse='csc', copy=copy, ensure_2d=False,
                    warn_on_dtype=True, estimator='the scale function',
                    force_all_finite=False, dtype=FLOAT_DTYPES)
    if sparse.issparse(X):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives.")
        if axis != 0:
            raise ValueError("Can only scale sparse matrix on axis=0, "
                             " got axis=%d" % axis)
        if with_std:
            _, var = mean_variance_axis(X, axis=0)
            var = _handle_zeros_in_scale(var, copy=False)
            inplace_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        if with_mean:
            mean_ = np.nanmean(X, axis)
        if with_std:
            scale_ = np.nanstd(X, axis)
        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = Xr.mean(axis=0)
            # Verify that mean_1 is 'close to zero'. If X contains very
            # large values, mean_1 can also be very large, due to a lack of
            # precision of mean_. In this case, a pre-scaling of the
            # concerned feature is efficient, for instance by its mean or
            # maximum.
            if not np.allclose(mean_1[np.isfinite(mean_1)], 0):
                warnings.warn("Numerical issues were encountered "
                              "when centering the data "
                              "and might not be solved. Dataset may "
                              "contain too large values. You may need "
                              "to prescale your features.")
                Xr -= mean_1
        if with_std:
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            Xr /= scale_
            if with_mean:
                mean_2 = Xr.mean(axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of mean_. A solution is then to
                # subtract the mean again:
                if not np.allclose(mean_2[np.isfinite(mean_1)], 0):
                    warnings.warn("Numerical issues were encountered "
                                  "when scaling the data "
                                  "and might not be solved. The standard "
                                  "deviation of the data is probably "
                                  "very close to 0. ")
                    Xr -= mean_2
    return X


class StandardNANScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.
    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).
    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.
    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    Attributes
    ----------
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
        .. versionadded:: 0.17
           *scale_*
    mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.
    var_ : array of floats with shape [n_features]
        The variance for each feature in the training set. Used to compute
        `scale_`
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    >>> print(scaler.mean_)
    [ 0.5  0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[ 3.  3.]]
    See also
    --------
    scale: Equivalent function without the estimator API.
    :class:`sklearn.decomposition.PCA`
        Further removes the linear correlation across features with 'whiten=True'.
    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """  # noqa

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : Passthrough for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        force_all_finite=False, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        self.mean_ = .0
        if self.with_std:
            self.var_ = .0
        else:
            self.var_ = None

        if sparse.issparse(X):
            raise NotImplementedError

        if self.with_mean:
            self.mean_ = np.nanmean(X, 0)
        if self.with_std:
            scale_ = np.nanstd(X, 0)
            self.scale_ = _handle_zeros_in_scale(scale_, copy=False)

        return self

    def transform(self, X, y='deprecated', copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        y : (ignored)
            .. deprecated:: 0.19
               This parameter will be removed in 0.21.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, warn_on_dtype=True,
                        force_all_finite=False, estimator=self,
                        dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            raise NotImplementedError

        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, 0)
        if self.with_mean:
            Xr -= self.mean_
            mean_1 = Xr.mean(axis=0)
            # Verify that mean_1 is 'close to zero'. If X contains very
            # large values, mean_1 can also be very large, due to a lack of
            # precision of mean_. In this case, a pre-scaling of the
            # concerned feature is efficient, for instance by its mean or
            # maximum.
            if not np.allclose(mean_1[np.isfinite(mean_1)], 0):
                warnings.warn("Numerical issues were encountered "
                              "when centering the data "
                              "and might not be solved. Dataset may "
                              "contain too large values. You may need "
                              "to prescale your features.")
                Xr -= mean_1
        if self.with_std:
            Xr /= self.scale_
            if self.with_mean:
                mean_2 = Xr.mean(axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of mean_. A solution is then to
                # subtract the mean again:
                if not np.allclose(mean_2[np.isfinite(mean_1)], 0):
                    warnings.warn("Numerical issues were encountered "
                                  "when scaling the data "
                                  "and might not be solved. The standard "
                                  "deviation of the data is probably "
                                  "very close to 0. ")
                    Xr -= mean_2

        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            raise NotImplementedError
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X