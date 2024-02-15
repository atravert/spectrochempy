# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the base abstract classes to define models and estimators
(analysis, processing, ...).
"""

import logging
from copy import copy

import numpy as np
import traitlets as tr

from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.constants import MASKED, NOMASK
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.exceptions import NotTransformedError
from spectrochempy.utils.metaconfigurable import MetaConfigurable
from spectrochempy.utils.traits import NDDatasetType


# ======================================================================================
# Base class BaseConfigurable
# ======================================================================================
class BaseConfigurable(MetaConfigurable):
    __doc__ = _docstring.dedent(
        r"""
    Abstract class to write configurable models (analysis, preprocessing, ...).

    Model class must subclass this to get a minimal structure

    Parameters
    ----------
    log_level : any of [``"INFO"``\ , ``"DEBUG"``\ , ``"WARNING"``\ , ``"ERROR"``\ ], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    """
    )

    # Get doc sections for reuse in subclass
    _docstring.get_sections(__doc__, base="BaseConfigurable")
    _docstring.keep_params("BaseConfigurable.parameters", "log_level")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _applied = tr.Bool(False, help="False if the model was not yet applied")

    _is_dataset = tr.Bool(help="True if the input X data is a NDDataset")

    _multiblock = tr.Bool(False, help="True if the data consists in several NDDatasets")

    _concatenation_axis = tr.Enum(
        (0, 1, 2),
        allow_none=True,
        default_value=None,
        help="Concatenation axis for multiblock data. "
        "None: no concatenation, "
        "O: column-wise, 1: row-wise, 2: column- & row-wise."
        "Default is None when multiblock is False, 0 when X is "
        "a list of datasets is given and 2 when a list of lists"
        "od NDDataset is given",
    )

    _masked_rc = tr.Tuple(allow_none=True, help="List of masked rows and columns")

    _X = tr.Union(
        [
            tr.List(help="A list of NDDatasets to fit a model"),
            NDDatasetType(allow_none=True, help="A single NDDataset to fit a " "model"),
        ]
    )

    _X_mask = tr.Union(
        [
            tr.List(
                Array(allow_none=True, help="Masks information of the input X data")
            ),
            Array(allow_none=True, help="Mask information of the input X data"),
        ]
    )

    _X_preprocessed = Array(
        help="Preprocessed initial input X data, including the "
        "concatenation of all datasets when mltiblock data "
        "are used"
    )

    _X_shape = tr.Union(
        [
            tr.List(
                help="Original shapes of the input X data before any transformation"
            ),
            tr.Tuple(
                help="Original shape of the input X data before any transformation"
            ),
        ]
    )

    _X_coordset = tr.Union(
        [
            tr.Instance(CoordSet, allow_none=True),
            tr.List(tr.Instance(CoordSet, allow_none=True)),
        ]
    )

    _warm_start = tr.Bool(False)
    _output_type = tr.Enum(
        ["NDDataset", "ndarray"],
        default_value="NDDataset",
        help="Whether the output is a NDDataset or a ndarray",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model)
    # ----------------------------------------------------------------------------------

    # Write here traits like e.g.,
    #     A = Unicode("A", help='description").tag(config=True)

    concatenation_axis = tr.Enum(
        (0, 1, 2),
        allow_none=True,
        default_value=None,
        help="Concatenation axis for multiblock data. "
        "None: no concatenation, "
        "O: column-wise, 1: row-wise, 2: column- & row-wise."
        "Default is None when multiblock is False, 0 when X is "
        "a list of datasets is given and 2 when a list of lists"
        "od NDDataset is given",
    )
    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        log_level=logging.WARNING,
        **kwargs,
    ):

        """ """
        # An empty __doc__ is placed here, else Configurable.__doc__
        # will appear when there is no __init___.doc in subclass
        from spectrochempy.application import app
        from spectrochempy.core import set_loglevel

        # Reset default configuration if not warm_start
        reset = not self._warm_start

        # Call the super class (MetaConfigurable) for initialisation
        super().__init__(parent=app, reset=reset)

        # Set log_level of the console report (accessible using the log property)
        set_loglevel(log_level)

        # Initial configuration
        # ---------------------
        # Reset all config parameters to default, if not warm_start
        defaults = self.params(default=True)
        configkw = {} if self._warm_start else defaults

        # Eventually take parameters from kwargs
        configkw.update(kwargs)

        # Now update all configuration parameters
        # if an item k is not in the config parameters, an error is raised.
        for k, v in configkw.items():
            if hasattr(self, k) and k in defaults:
                if getattr(self, k) != v:
                    setattr(self, k, v)
            else:
                raise KeyError(
                    f"'{k}' is not a valid configuration parameters. "
                    f"Use the method `parameters()` to check the current "
                    f"allowed parameters and their current value."
                )

        # If warm start we can use the previous fit as starting profiles.
        # so the flag _applied is not set.
        self._applied = False

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _make_dataset(self, d):
        # Transform an array-like object to NDDataset
        # or a list of array-like to a list of NDQataset
        if d is None:
            return
        if isinstance(d, (tuple, list)):
            d = [self._make_dataset(item) for item in d]
        elif not isinstance(d, NDDataset):
            d = NDDataset(d, copy=True)
        else:
            d = d.copy()
        return d

    def _make_unsqueezed_dataset(self, d):
        # add a dimension to 1D Dataset
        if d.ndim == 1:
            coordset = d.coordset
            d._data = d._data[np.newaxis]
            if np.any(d.mask):
                d._mask = d._mask[np.newaxis]
            d.dims = ["y", "x"]  # "y" is the new dimension
            coordx = coordset[0] if coordset is not None else None
            d.set_coordset(x=coordx, y=None)
        return d

    def _get_masked_rc(self, mask):
        # Get the mask by row and columns.
        # -------------------------------
        # When a single element in the array is
        # masked, the whole row and columns for this element is masked as well as the
        # corresponding columns.
        if np.any(mask):
            masked_columns = np.all(mask, axis=-2)  # if mask.ndim == 2 else None
            masked_rows = np.all(mask, axis=-1)
        else:
            masked_columns = np.zeros(self._X_shape[-1], dtype=bool)
            masked_rows = np.zeros(self._X_shape[-2], dtype=bool)
        return masked_rows, masked_columns

    def _remove_masked_data(self, X):
        # Retains only valid rows and columns
        # -----------------------------------
        # unfortunately, the implementation of linalg library
        # doesn't support numpy masked arrays as input. So we will have to
        # remove the masked values ourselves

        # the following however assumes that entire rows or columns are masked,
        # not only some individual data (if this is what you wanted, this
        # will fail)
        if not hasattr(X, "mask") or not np.any(X._mask):
            return X

        # remove masked rows and columns
        masked_rows, masked_columns = self._get_masked_rc(X._mask)

        Xc = X[:, ~masked_columns]
        Xrc = Xc[~masked_rows]

        # destroy the mask
        Xrc._mask = NOMASK

        # return the modified X dataset
        return Xrc

    def _restore_masked_data(self, D, axis=-1):
        # by default, we restore columns, put axis=0 to restore rows instead
        # Note that it is very important to use here the ma version of zeros
        # array constructor or both if both axis should be restored
        if not np.any(self._X_mask):
            # return it inchanged as wa had no mask originally
            return D

        rowsize, colsize = self._X_shape
        masked_rows, masked_columns = self._get_masked_rc(self._X_mask)

        if D.ndim == 2:
            # Put back masked columns in D
            # ----------------------------
            M, N = D.shape
            if axis == "both":  # and D.shape[0] == rowsize:
                if np.any(masked_columns) or np.any(masked_rows):
                    Dtemp = np.ma.zeros((rowsize, colsize))  # note np.ma, not np.
                    Dtemp[~self._X_mask] = D.data.flatten()
                    Dtemp[self._X_mask] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-1]] = self._X_coordset[D.dims[-1]]
                        D.coordset[D.dims[-2]] = self._X_coordset[D.dims[-2]]
                    except TypeError:
                        # probably no coordset
                        pass
            elif axis == -1 or axis == 1:
                if np.any(masked_columns):
                    Dtemp = np.ma.zeros((M, colsize))  # note np.ma, not np.
                    Dtemp[:, ~masked_columns] = D
                    Dtemp[:, masked_columns] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-1]] = self._X_coordset[D.dims[-1]]
                    except TypeError:
                        # probably no coordset
                        pass

            # Put back masked rows in D
            # -------------------------
            elif axis == -2 or axis == 0:
                if np.any(masked_rows):
                    Dtemp = np.ma.zeros((rowsize, N))
                    Dtemp[~masked_rows] = D
                    Dtemp[masked_rows] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-2]] = self._X_coordset[D.dims[-2]]
                    except TypeError:
                        # probably no coordset
                        pass
        elif D.ndim == 1:
            # we assume here that the only case it happens is for array as explained
            # variance so that we deal with masked rows
            if np.any(masked_rows):
                Dtemp = np.ma.zeros((rowsize,))  # note np.ma, not np.
                Dtemp[~masked_rows] = D
                Dtemp[masked_rows] = MASKED
                D.data = Dtemp

        elif D.ndim == 3:
            # CASE of IRIS for instance

            # Put back masked columns in D
            # ----------------------------
            J, M, N = D.shape
            if axis == -1 or axis == 2:
                if np.any(masked_columns):
                    Dtemp = np.ma.zeros((J, M, colsize))  # note np.ma, not np.
                    Dtemp[..., ~masked_columns] = D
                    Dtemp[..., masked_columns] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-1]] = self._X_coordset[D.dims[-1]]
                    except TypeError:
                        # probably no coordset
                        pass

        # return the D array with restored masked data
        return D

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.default("_X")
    def _X_default(self):
        raise NotTransformedError

    @tr.validate("_X")
    def _X_validate(self, proposal):
        # validation fired when self._X is assigned
        X = proposal.value

        # X can be a NDDataset, or a list or tuple of NDDatasets. In the latter case, a
        # concatenation_axis must have been given.

        if isinstance(X, NDDataset):
            self._is_dataset = True
            self._multiblock = False
            X = [X]
        elif isinstance(X, (list, tuple)):
            self._multiblock = True
            if all([isinstance(x, (list, tuple)) for x in X]):
                # case of a list/tuple of lists/tuples of NDDatasets
                raise NotImplementedError(
                    "row- and column-wise multiblock " "not yet implemented"
                )
                # todo: check that all elements of lists/tuples are NDDataset and have
                # the correct shape
                self._multiblock = True
                self._concatenation_axis = 2

            elif not all([isinstance(x, NDDataset) for x in X]):
                raise ValueError("X must be a NDDataset or a list of NDDatasets")
            elif self._concatenation_axis is None:
                # we have a list of NDDatasets, try to guess the concatenation axis
                samey, samex = False, False
                if all([x.data.shape[0] == X[0].data.shape[0] for x in X]):
                    samey = True
                if all([x.data.shape[1] == X[0].data.shape[1] for x in X]):
                    samex = True
                if samey and samex:
                    raise ValueError(
                        "all NDDatasets in X have both dimensions with the "
                        "length: the concatenation axis can't be guessed"
                    )
                elif not samey and not samex:
                    raise ValueError(
                        "all NDDatasets in X must have at least one dimension "
                        "with the same length"
                    )
                elif samey:
                    self._concatenation_axis = 1
                else:
                    self._concatenation_axis = 0
            else:
                # we have a list of NDDatasets and the concatenation axis was given
                # todo: check that all NDDatasets have the correct shape
                pass

            self._X_shape = []
            self._X_mask = []
            self._X_coordset = []

        for x in X:
            # for the following we need x with two dimensions
            # So let's generate the un-squeezed x
            x = x.atleast_2d()  # self._make_unsqueezed_dataset(x)

            # if x is complex or quaternion, we will work on the real part only
            if x.is_complex or x.is_quaternion:
                x = x.real

            if not self._multiblock:
                # as in fit methods we often use np.linalg library, we cannot handle directly
                # masked data (so we remove them here and they will be restored at the end of
                # the process during transform or inverse transform methods
                # store the original shape as it will be eventually modified as well as the
                # original coordset
                self._X_shape = x.shape

                # store the mask because it may be destroyed
                self._X_mask = x._mask.copy()

                # and the original coordset
                self._X_coordset = copy(x._coordset)

            else:
                # the same, but as lists
                self._X_shape.append(x.shape)
                self._X_mask.append(x._mask.copy())
                self._X_coordset.append(x._coordset)

            # remove masked data and return modified dataset
            x = self._remove_masked_data(x)

        if not self._multiblock:
            return X[0]
        else:
            return X

    @property
    def _X_is_missing(self):
        # check whether X has been already defined
        try:
            if self._X is None:
                return True
        except NotTransformedError:
            return True
        return False

    # ----------------------------------------------------------------------------------
    # Private methods that should be, most of the time, overloaded in subclass
    # ----------------------------------------------------------------------------------
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        X = change.new
        # .... preprocessing as scaling, centering, ... must return a ndarray with
        #  same shape a X.data, or a list of ndarrays with same shape as the NDDatasets

        # Set a X.data by default
        if not self._multiblock:
            self._X_preprocessed = X.data
        else:
            # multiblock: must be concatenated
            if self._concatenation_axis in (0, 1):
                try:
                    self._X_preprocessed = np.concatenate(
                        [x.data for x in X], axis=self._concatenation_axis
                    )
                except ValueError as e:
                    raise ValueError(
                        f"The concatenation of input data along the axis "
                        f"{self._concatenation_axis} went wrong with "
                        f"the followig error: {e}. Please check your input "
                        f"data."
                    )

            else:
                try:
                    # concatenate list of lists of NDDataset
                    col = []
                    for row in X:
                        col.append(np.concatenate([x.data for x in row], axis=1))
                    self._X_preprocessed = np.concatenate([x.data for x in col], axis=0)
                except ValueError as e:
                    raise ValueError(
                        f"The concatenation of input data went wrong with "
                        f"the followig error: {e}. Please check your input "
                        f"data."
                    )

    # ----------------------------------------------------------------------------------
    # Public methods and property
    # ----------------------------------------------------------------------------------

    @property
    def log(self):
        """
        Return ``log`` output.
        """
        # A string handler (#1) is defined for the Spectrochempy logger,
        # thus we will return it's content
        from spectrochempy.application import app

        return app.log.handlers[1].stream.getvalue().rstrip()
