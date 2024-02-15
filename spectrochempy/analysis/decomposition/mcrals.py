# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the MCRALS class.
"""

# DEVNOTE:
# API methods accessible as  scp.method or scp.class must be defined in __all__
# Configurable class (which requires a configuration file)
# must be declared in __configurable__

__all__ = ["MCRALS"]
__configurables__ = ["MCRALS"]

import base64
import logging
import warnings

import dill
import numpy as np
import scipy
import traitlets as tr
from sklearn import decomposition

from spectrochempy.analysis._base._analysisbase import (
    DecompositionAnalysis,
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.application import info_
from spectrochempy.extern.traittypes import Array
from spectrochempy.processing.transformation.concatenate import concatenate
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring


# DEVNOTE:
# the following decorator allow to correct signature and docs of traitlets.HasTraits
# derived class
@signature_has_configurable_traits
class MCRALS(DecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "MCRALS")

    __doc__ = _docstring.dedent(
        """
    Multivariate Curve Resolution Alternating Least Squares (MCRALS).

    :term:`MCR-ALS` ( ``Multivariate Curve Resolution Alternating Least Squares`` )
    resolve's a set (or several sets) of spectra :math:`X` of an evolving mixture
    (or a set of mixtures) into the spectra :math:`S^t` of "pure" species and their
    concentration profiles :math:`C`\ .

    In terms of matrix equation:

    .. math:: X = C.S^t + E

    where :math:`E` is the matrix of residuals.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_MCRALS)s
    """
    )

    # Developer notes
    # ----------------
    # Private traits with internal validation (see Traitlets library for more
    # information)
    # Notice that variable not defined this way lack this type validation, so they are
    # more prone to errors.

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # They will be written in a file from which the default can be modified)
    # Obviously, the parameters can also be modified at runtime as usual by assignment.
    # ----------------------------------------------------------------------------------

    tol = tr.Float(
        0.1,
        help=(
            "Convergence criterion on the change of residuals (percent change of "
            "standard deviation of residuals), default is 0.1."
        ),
    ).tag(config=True)

    max_iter = tr.Integer(
        50, help="Maximum number of :term:`ALS` iteration, " "default is 50"
    ).tag(config=True)

    maxdiv = tr.Integer(
        5, help="Maximum number of successive non-converging iterations, default is 5."
    ).tag(config=True)

    solverConc = tr.Enum(
        ["lstsq", "nnls", "pnnls"],
        default_value="lstsq",
        help=(
            r"""Solver used to get `C` from `X` and `St`\ .

- ``'lstsq'``\ (default): uses ordinary least squares with `~numpy.linalg.lstsq`
- ``'nnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied
  sequentially on all profiles
- ``'pnnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied on
  profiles indicated in `nonnegConc` and ordinary least squares on other profiles.
"""
        ),
    ).tag(config=True)

    nonnegConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Non-negativity constraint on concentrations.

- ``'all'``\ (default): all concentrations profiles are considered
  non-negative.
- `list` of indexes: the corresponding profiles are considered non-negative,
  not the others. For instance ``[0, 2]`` indicates that profile \#0 and \#2
  are non-negative while profile \#1 *can* be negative.
- ``[]``\ : all profiles can be negative.
- `list`of lists or tuples of indexes: in case of multiblock data, the corresponding profiles are
  considered non-negative, not the others. For instance ``[(0, 2), (1)]`` indicates that
  2 profiles are considered non-negative in the first block and only one in the second.
          """
        ),
    ).tag(config=True)

    unimodConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Unimodality constraint on concentrations.

- ``'all'``\ (default): all concentrations profiles are considered unimodal.
- `list` of indexes: the corresponding profiles are considered unimodal, not the others.
  For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are unimodal while
  profile ``#1`` *can* be multimodal.
- ``[]``\ : all profiles can be multimodal.
- `list`of lists or tuples of indexes: in case of multiblock data, the corresponding
  profiles are considered unimodal, not the others. For instance ``[[0, 2), (1)]``
  indicates that 2 profiles are considered unimodal in the first block and only one in
  the second.
          """
        ),
    ).tag(config=True)

    unimodConcMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help=(
            r"""Method to apply unimodality.

- ``'strict'``\ (default): values deviating from :term:`unimodality` are reset to the value of the
  previous point.
- ``'smooth'``\ : both values (deviating point and previous point) are modified to avoid
  steps in the concentration profile."""
        ),
    ).tag(config=True)

    unimodConcTol = tr.Float(
        default_value=1.1,
        help=(
            r"""Tolerance parameter for :term:`unimodality`\ (default is 1.1).

Correction is applied only if:

- ``C[i,j] > C[i-1,j] * unimodTol`` on the decreasing branch of profile ``#j``\ ,
- ``C[i,j] < C[i-1,j] * unimodTol`` on the increasing branch of profile ``#j``\ ."""
        ),
    ).tag(config=True)

    monoDecConc = tr.List(
        default_value=[],
        help=(
            r"""Monotonic decrease constraint on concentrations.

- ``[]``\ (default): no constraint is applied.
- `list` of indexes: the corresponding profiles are considered to decrease
  monotonically, not the others. For instance ``[0, 2]`` indicates that profile ``#0``
  and ``#2`` are decreasing while profile ``#1`` *can* increase.
- `list`of lists or tuples of indexes: in case of multiblock data, the corresponding
  profiles are considered monotonic, not the others. For instance ``[(0, 2), (1)]``
  indicates that 2 profiles are considered monotonic in the first block and only one in
  the second."""
        ),
    ).tag(config=True)

    monoDecTol = tr.Float(
        default_value=1.1,
        help=r"""Tolerance parameter for monotonic decrease (default is 1.1).

Correction is applied only if: ``C[i,j] > C[i-1,j] * unimodTol`` .""",
    ).tag(config=True)

    monoIncConc = tr.List(
        default_value=[],
        help=(
            r"""Monotonic increase constraint on concentrations.

- ``[]``\ (default): no constraint is applied.
- `list` of indexes: the corresponding profiles are considered to increase
  monotonically, not the others. For instance ``[0, 2]`` indicates that profile ``#0``
  and ``#2`` are increasing while profile ``#1`` *can* decrease.
- `list`of lists or tuples of indexes: in case of multiblock data, the corresponding
  profiles are considered monotonic, not the others. For instance ``[(0, 2), (1)]``
  indicates that 2 profiles are considered monotonic in the first block and only one in
  the second"""
        ),
    ).tag(config=True)

    monoIncTol = tr.Float(
        default_value=1.1,
        help="""Tolerance parameter for monotonic decrease (default is 1.1).

Correction is applied only if ``C[i,j] < C[i-1,j] * unimodTol`` along profile ``#j``\ .""",
    ).tag(config=True)

    closureConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help=(
            r"""Defines the concentration profiles subjected to closure constraint.

- ``[]``\ (default): no constraint is applied.
- ``'all'``\ : all profile are constrained so that their weighted sum equals the
  `closureTarget`
- `list` of indexes: the corresponding profiles are constrained so that their weighted sum
  equals `closureTarget`\ .
- `list`of lists or tuples of indexes: in case of multiblock data, the corresponding
  profiles are considered for closure constraint, not the others. For instance ``[(0, 2), (1)]``
  indicates that 2 profiles are considered monotonic in the first block and only one in
  the second"""
        ),
    ).tag(config=True)

    closureTarget = tr.Union(
        (tr.Enum(["default"]), Array(), tr.List()),
        default_value="default",
        help=(
            r"""The value of the sum of concentrations profiles subjected to closure.

- ``'default'``\ : the total concentration is set to ``1.0`` for all observations.
- :term:`array-like` of size :term:`n_observations`: the values of concentration for
  each observation. Hence, ``np.ones(X.shape[0])`` would be equivalent to
  ``'default'``\ .
- `list` of :term:`array-like`: in case of multiblock data, the corresponding
  concentration targets.
"""
        ),
    ).tag(config=True)

    closureMethod = tr.Enum(
        ["scaling", "constantSum"],
        default_value="scaling",
        help=(
            r"""The method used to enforce :term:`closure` (:cite:t:`omidikia:2018`).

- ``'scaling'`` (default) recompute the concentration profiles using least squares:

  .. math::

     C \leftarrow C \cdot \textrm{diag} \left( C_L^{-1} c_t \right)

  where :math:`c_t` is the vector given by `closureTarget` and :math:`C_L^{-1}`
  is the left inverse of :math:`C`\ .
- ``'constantSum'`` normalize the sum of concentration profiles to `closureTarget`\ .
"""
        ),
    ).tag(config=True)

    hardConc = tr.List(
        default_value=[],
        help=(
            r"""Defines hard constraints on the concentration profiles.

- ``[]``\ (default): no constraint is applied.
- `list` of indexes: the corresponding profiles will set by `getConc`\ .
- `list`of lists or tuples of indexes: in case of multiblock data, the corresponding
  profiles will be set."""
        ),
    ).tag(config=True)

    getConc = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help=(
            r"""An external function that provide ``len(hardConc)`` concentration
profiles.

It should be using one of the following syntax:

- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC``
- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc``
- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc,
  extraOutputGetConc``

with:

- ``Ccurr`` is the current `C` dataset,
- ``argsGetConc`` are the parameters needed to completely specify the function.
- ``hardC`` is a `~numpy.ndarray` or `NDDataset` of shape
  (:term:`n_observations` , len(``hardConc``\ ), or a list of such arrays or NDDatasets.
- ``newArgsGetConc`` are the updated parameters for the next iteration (can be `None`),
- ``extraOutputGetConc`` can be any other relevant output to be kept in
  ``extraOutputGetConc`` attribute, a list of ``extraOutputGetConc`` at each MCR ALS
  iteration.

.. note::
    ``getConc`` can be also a serialized function created using dill and base64
    python libraries. Normally not used directly, it is here for internal
    process."""
        ),
    ).tag(config=True)

    argsGetConc = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetConc = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    getC_to_C_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help=(
            r"""Correspondence of the profiles returned by `getConc`
and `C[:,hardConc]`\ .

- ``'default'``: the profiles correspond to those of `C[:,hardConc]`. This is equivalent
  to ``range(len(hardConc))``
- `list` of indexes or of `None`. For instance ``[2, 1, 0]`` indicates that the
  third profile returned by `getC` (index ``2``\ ) corresponds to the 1st profile of
  `C[:, hardConc]`\ , the 2nd returned profile (index ``1``\ ) corresponds to
  second profile of `C[:, hardConc]`, etc..."""
        ),
    ).tag(config=True)

    solverSpec = tr.Enum(
        ["lstsq", "nnls", "pnnls"],
        default_value="lstsq",
        help=(
            r"""Solver used to get `St` from `X` and `C`\ .

- ``'lstsq'``\ (default): uses ordinary least squares with `~numpy.linalg.lstsq` (default)
- ``'nnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied
  sequentially on all profiles
- ``'pnnls'``\ : non-negative least squares (`~scipy.optimize.nnls`\ ) are applied on
  profiles indicated in `nonnegConc` and ordinary least squares on other profiles."""
        ),
    ).tag(config=True)

    nonnegSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Non-negativity constraint on spectra.

- ``'all'``\ (default): all profiles are considered non-negative (default).
- `list` of indexes : the corresponding profiles are considered non-negative, not the
  others. For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are
  non-negative while profile ``#1`` *can* be negative.
- ``[]``\ : all profiles can be negative."""
        ),
    ).tag(config=True)

    normSpec = tr.Enum(
        [None, "euclid", "max"],
        default_value=None,
        help=(
            r"""Defines whether the spectral profiles should be normalized.

- `None`\ (default): no normalization is applied.
- ``'euclid'``\ : spectra are normalized with respect to their total area,
- ``'max'``\ : spectra are normalized with respect to their maximum value."""
        ),
    ).tag(config=True)

    unimodSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help=(
            r"""Unimodality constraint on Spectra.

- ``[]``\ (default): all profiles can be multimodal.
- ``'all'``\ : all profiles are unimodal (equivalent to ``range(n_components)``\ ).
- array of indexes : the corresponding profiles are considered unimodal, not the others.
  For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are unimodal while
  profile ``#1`` *can* be multimodal."""
        ),
    ).tag(config=True)

    unimodSpecMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help=(
            r"""Method used to apply unimodality.

- ``'strict'``\ : values deviating from unimodality are reset to the value of the previous
  point.
- ``'smooth'``\ : both values (deviating point and previous point) are modified to avoid
  steps in the concentration profile."""
        ),
    ).tag(config=True)

    unimodSpecTol = tr.Float(
        default_value=1.1,
        help=(
            r"""Tolerance parameter for unimodality (default is 1.1)

Correction is applied only if the deviating point ``St[j, i]`` is larger than
``St[j, i-1] * unimodSpecTol`` on the decreasing branch of profile
``#j``\ , or lower than ``St[j, i-1] * unimodTol`` on the increasing branch of
profile  ``#j``\ ."""
        ),
    ).tag(config=True)

    hardSpec = tr.List(
        default_value=[],
        help=(
            r"""Defines hard constraints on the spectral profiles.

- ``[]``\ (default): no constraint is applied.
- `list` of indexes : the corresponding profiles will set by `getSpec`\ ."""
        ),
    ).tag(config=True)

    getSpec = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help=(
            r"""An external function that will provide ``len(hardSpec)`` spectral
profiles.

It should be using one of the following syntax:

- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt``
- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec``
- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec,
  extraOutputGetSpec``

with:

- ``Stcurr``\ : the current value of `St` in the :term:`ALS` loop,
- ``*argsGetSpec`` and ``**kwargsGetSpec``\ : the parameters needed to completely
  specify the function.
- ``hardSt``\ : `~numpy.ndarray` or `NDDataset` of shape
  ``(n_observations, len(hardSpec)``\ ,
- ``newArgsGetSpec``\ : updated parameters for the next ALS iteration (can be None),
- ``extraOutputGetSpec``\ : any other relevant output to be kept in
  `extraOutputGetSpec` attribute, a list of ``extraOutputGetSpec`` at each iterations.

.. note::
    ``getSpec`` can be also a serialized function created using dill and base64
    python libraries. Normally not used directly, it is here for internal process.
"""
        ),
    ).tag(config=True)

    argsGetSpec = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetSpec = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    getSt_to_St_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help=(
            r"""Correspondence between the indexes of the spectra returned by `getSpec`
and `St`.

- ``'default'``\ : the indexes correspond to those of `St`. This is equivalent
  to ``range(len(hardSpec))``\ .
- `list` of indexes : corresponding indexes in `St`, i.e. ``[2, None, 0]`` indicates that the
  first returned profile corresponds to the third `St` profile (index ``2``\ ), the 2nd
  returned profile does not correspond to any profile in `St`, the 3rd returned profile
  corresponds to the first `St` profile (index ``0`` )."""
        ),
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *args,
        log_level=logging.WARNING,
        warm_start=False,
        **kwargs,
    ):
        if len(args) > 0:
            raise ValueError(
                "Passing arguments such as MCRALS(X, profile) "
                "is now deprecated. "
                "Instead, use MCRAL() followed by MCRALS.fit(X, profile). "
                "See the documentation and examples"
            )

        # warn about deprecation
        # ----------------------
        # We use pop to remove the deprecated argument before processing the rest
        # TODO: These arguments should be removed in version 0.7 probably

        # verbose
        if "verbose" in kwargs:
            deprecated("verbose", replace="log_level='INFO'", removed="0.7")
            verbose = kwargs.pop("verbose")
            if verbose:
                log_level = "INFO"

        # unimodTol deprecation
        if "unimodTol" in kwargs:
            deprecated("unimodTol", replace="unimodConcTol", removed="0.7")
            kwargs["unimodConcTol"] = kwargs.pop("unimodTol")

        # unimodMod deprecation
        if "unimodMod" in kwargs:
            deprecated("unimodMod", replace="unimodConcMod", removed="0.7")
            kwargs["unimodConcMod"] = kwargs.pop("unimodMod")

        # hardC_to_C_idx deprecation
        if "hardC_to_C_idx" in kwargs:
            deprecated("hardC_to_C_idx", replace="getC_to_C_idx", removed="0.7")
            kwargs["getC_to_C_idx"] = kwargs.pop("hardC_to_C_idx")

        # hardSt_to_St_idx deprecation
        if "hardSt_to_St_idx" in kwargs:
            deprecated("hardSt_to_St_idx", replace="getSt_to_St_idx", removed="0.7")
            kwargs["getSt_to_St_idx"] = kwargs.pop("hardSt_to_St_idx")

        # call the super class for initialisation
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

        # deal with the callable that may have been serialized
        if self.getConc is not None and isinstance(self.getConc, str):
            self.getConc = dill.loads(base64.b64decode(self.getConc))
        if self.getSpec is not None and isinstance(self.getSpec, str):
            self.getSpec = dill.loads(base64.b64decode(self.getSpec))

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------

    def _solve_C(self, St):
        if self.solverConc == "lstsq":
            return _lstsq(St.T, self._X_preprocessed.T).T
        elif self.solverConc == "nnls":
            return _nnls(St.T, self._X_preprocessed.T).T
        elif self.solverConc == "pnnls":
            return _pnnls(St.T, self._X_preprocessed.T, nonneg=self.nonnegConc).T

    def _solve_St(self, C):
        if self.solverSpec == "lstsq":
            return _lstsq(C, self._X_preprocessed)
        elif self.solverSpec == "nnls":
            return _nnls(C, self._X_preprocessed)
        elif self.solverSpec == "pnnls":
            return _pnnls(C, self._X_preprocessed, nonneg=self.nonnegSpec)

    def _guess_profile(self, profile):
        # Set or guess an initial profile.

        if self._X_is_missing:
            return

        # check the dimensions compatibility
        #
        # when a list of profiles is given (augmented data), we check they have a
        # compatible shape with the data
        if self._multiblock and self._concatenation_axis in (0, 1):
            other_axis = abs(self._concatenation_axis - 1)
            if isinstance(profile, (list, tuple)):
                # a list of concentration or spectral profiles is given
                if len(profile) != len(self._X):
                    raise ValueError(
                        f"The number of given profiles ({len(profile)}) does not "
                        f"match the number of datasets ({len(self._X)}) "
                    )

                # check the shape of the profiles
                if not all(
                    [
                        p.shape[other_axis] == profile[0].shape[other_axis]
                        for p in profile
                    ]
                ):
                    raise ValueError(
                        f"The dimension of the given profiles along the axis {other_axis} "
                        f"should be the same as that of the input datasets: {self._X_preprocessed.shape[other_axis]}"
                    )
                elif (
                    sum([p.shape[self._concatenation_axis] for p in profile])
                    != self._X_preprocessed.shape[self._concatenation_axis]
                ):
                    raise ValueError(
                        f"The sum of the dimensions of the given profiles along axis {self._concatenation_axis} is "
                        f"{sum([p.shape[self._concatenation_axis] for p in profile])}, but the corresponding "
                        f"dimension of X is {self._X_shape[self._concatenation_axis ]}"
                    )

                # concatenate the profiles
                if self._concatenation_axis == 0:
                    C = concatenate(*profile, axis=0)
                    self._n_components = C.shape[1]
                    St = self._solve_St(C)
                else:
                    St = concatenate(*profile, axis=1)
                    self._n_components = St.shape[0]
                    C = self._solve_C(St)

            else:
                # a single profile is given
                if profile.shape[other_axis] != self._X_preprocessed.shape[other_axis]:
                    raise ValueError(
                        f"The dimension of the given profile along the axis {other_axis} "
                        f"should be the same as that of the input datasets: {self._X_preprocessed.shape[other_axis]}"
                    )

                if self._concatenation_axis == 1:
                    C = profile.copy()
                    self._n_components = C.shape[1]
                    St = self._solve_St(C)
                    info_("Initial spectra profile computed")
                else:
                    St = profile.copy()
                    self._n_components = St.shape[0]
                    C = self._solve_C(St)
                    info_("Initial concentration profile computed")

        elif self._multiblock and self.concatenation_axis == 2:
            # todo: implement this case
            pass

        else:  # a single dataset X is given

            # As the dimension of profile should match the initial shape
            # of X we use self._X_shape not self._X.shape (because for this masked columns
            # or rows have already been removed.
            if (self._X_shape[1] != profile.shape[1]) and (
                self._X_shape[0] != profile.shape[0]
            ):
                raise ValueError(
                    f"None of the dimensions of the given profile(s) "
                    f"[{profile.shape}] correspond to any of those "
                    f"of X [{self._X_shape}]."
                )

            # mask info
            if np.any(self._X_mask):
                masked_rows, masked_columns = self._get_masked_rc(self._X_mask)

            # make the profile
            if profile.shape[0] == self._X_shape[0]:
                # this should be a concentration profile.
                C = profile.copy()
                self._n_components = C.shape[1]
                info_(
                    f"Concentration profile initialized with {self._n_components} components"
                )

                # compute initial spectra (using X eventually masked)
                St = self._solve_St(C)
                info_("Initial spectra profile computed")
                # if everything went well here, C and St are set, we return
                # after having removed the eventual C mask!
                if np.any(self._X_mask):
                    C = C[~masked_rows]
                return C, St

            else:  # necessarily: profile.shape[1] == profile.shape[0]
                St = profile.copy()
                self._n_components = St.shape[0]
                info_(
                    f"Spectra profile initialized with {self._n_components} components"
                )

                # compute initial spectra
                C = self._solve_C(St)
                info_("Initial concentration profile computed")
                # if everything went well here, C and St are set, we return
                # after having removed the eventual St mask!
                if np.any(self._X_mask):
                    St = St[:, ~masked_columns]

        return C, St

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def _C_2_NDDataset(self, C):
        # getconc takes the C NDDataset as first argument (to take advantage
        # of useful metadata). But the current C in fit method is a ndarray (without
        # the masked rows and colums, nor the coord information: this
        # function will create the corresponding dataset
        return C

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def _St_2_NDDataset(self, St):
        # getconc takes the C NDDataset as first argument (to take advantage
        # of useful metadata). The current St in fit method is a ndarray (without
        # the masked rows and columns, nor the coord information: this
        # function will create the corresponding dataset
        return St

    # ----------------------------------------------------------------------------------
    # Private validation methods and default getter
    # ----------------------------------------------------------------------------------
    @tr.validate("nonnegConc")
    def _validate_nonnegConc(self, proposal):

        if self._X_is_missing:
            return proposal.value
        nonnegConc = proposal.value

        if not self._n_components:  # not initialized or 0
            return nonnegConc

        if nonnegConc == "all":
            nonnegConc = np.arange(
                self._n_components
            ).tolist()  # IMPORTANT! .tolist, not list()
            # to get integer type not int64 which are not compatible with the setting
        elif np.any(nonnegConc) and (
            len(nonnegConc) > self._n_components
            or max(nonnegConc) + 1 > self._n_components
        ):  # note that we use np.any(nnonnegConc) instead of nnonnegConc != []
            # due to a deprecation warning from traitlets.
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `nonnegConc` configuration (value:{nonnegConc})"
            )
        return nonnegConc

    @tr.validate("unimodConc")
    def _validate_unimodConc(self, proposal):

        if self._X_is_missing:
            return proposal.value
        unimodConc = proposal.value

        if not self._n_components:  # not initialized or 0
            return unimodConc

        if unimodConc == "all":
            unimodConc = np.arange(self._n_components).tolist()

        # At this point we have either one list of indexes or a list of lists of indexes
        # if single list of indexes
        if all(isinstance(unimodConc[i], int) for i in range(len(unimodConc))):
            unimodConc = [unimodConc]
            if self._multiblock and self._concatenation_axis == 0:
                # we have to repeat the list as many times as there are
                # datasets in X
                unimodConc = np.repeat(unimodConc, len(self._X)).tolist()
        else:  # a list of lists, check that the number of lists is correct
            if len(unimodConc) != len(self._X):
                raise ValueError(
                    f"Multiblock dataset: the number of given unimodality constraints "
                    f"({len(unimodConc)}) does not match the number of datasets "
                    f"({len(self._X)}) "
                )
        # finally check that the number of indexes in each list is correct
        for _unimodConc in unimodConc:
            if len(_unimodConc) > self._n_components:
                raise ValueError(
                    f"The profile has only {self._n_components} species, please check "
                    f"the `unimodConc` configuration (value:{unimodConc})"
                )
            elif np.any(_unimodConc) and (
                len(_unimodConc) > self._n_components
                or max(_unimodConc) + 1 > self._n_components
            ):
                raise ValueError(
                    f"The profile has only {self._n_components} species, please check the "
                    f"`unimodConc` configuration (value:{_unimodConc})"
                )
        return unimodConc

    @tr.validate("closureConc")
    def _validate_closureConc(self, proposal):

        if self._X_is_missing:
            return proposal.value

        closureConc = proposal.value

        if closureConc == "all":
            closureConc = np.arange(self._n_components)

        # At this point we have either one list of indexes or a list of lists of indexes
        # if single list of indexes
        if all(isinstance(closureConc[i], int) for i in range(len(closureConc))):
            closureConc = [closureConc]
            if self._multiblock and self._concatenation_axis == 0:
                # we have to repeat the list as many times as there are
                # datasets in X
                closureConc = np.repeat(closureConc, len(self._X)).tolist()
        else:  # a list of lists, check that the number of lists is correct
            if len(closureConc) != len(self._X):
                raise ValueError(
                    f"Multiblock dataset: the number of given closure constraints "
                    f"({len(closureConc)}) does not match the number of datasets "
                    f"({len(self._X)}) "
                )
        # check that the number of indexes in each list is correct
        for _closureConc in closureConc:
            if len(_closureConc) > self._n_components:
                raise ValueError(
                    f"The profile has only {self._n_components} species, please check "
                    f"the `closureConc` configuration (value:{closureConc})"
                )
            elif np.any(_closureConc) and (
                len(_closureConc) > self._n_components
                or max(_closureConc) + 1 > self._n_components
            ):
                raise ValueError(
                    f"The profile has only {self._n_components} species, please check the "
                    f"`closureConc` configuration (value:{closureConc})"
                )
        return closureConc

    @tr.validate("closureTarget")
    def _validate_closureTarget(self, proposal):

        if self._X_is_missing:
            return proposal.value

        closureTarget = proposal.value

        if isinstance(closureTarget, str):
            if closureTarget == "default":
                closureTarget = np.ones(self._X_preprocessed.shape[0]).tolist()

        # At this point we have either one array of indexes or a list of arrays
        # if single list of indexes
        if all(
            isinstance(closureTarget[i], (int, float))
            for i in range(len(closureTarget))
        ):
            closureTarget = [closureTarget]
            if self._multiblock and self._concatenation_axis == 0:
                # we have to repeat the array as many times as there are
                # datasets in X
                closureTarget = np.repeat(closureTarget, len(self._X)).tolist()
        else:  # a list of arrays, check that the number of arrays is correct
            if len(closureTarget) != len(self._X):
                raise ValueError(
                    f"Multiblock dataset: the number of given closure targets "
                    f"({len(closureTarget)}) does not match the number of datasets "
                    f"({len(self._X)}) "
                )
        # check that the number of indexes in each list is correct
        for _closureTarget in closureTarget:
            if len(_closureTarget) != self._X_preprocessed.shape[0]:
                raise ValueError(
                    f"The data contain {self._X_preprocessed.shape[0]} observations, "
                    f"please check the 'closureTarget' configuration "
                    f"(value:{closureTarget})"
                )
        return closureTarget

    @tr.validate("getC_to_C_idx")
    def _validate_getC_to_C_idx(self, proposal):

        if self._X_is_missing:
            return proposal.value
        getC_to_C_idx = proposal.value

        if not self._n_components:  # not initialized or 0
            return getC_to_C_idx

        if getC_to_C_idx == "default":
            getC_to_C_idx = np.arange(self._n_components).tolist()

        # At this point we have either one list of indexes or a list of lists of indexes
        # if single list of indexes
        if all(isinstance(getC_to_C_idx[i], int) for i in range(len(getC_to_C_idx))):
            getC_to_C_idx = [getC_to_C_idx]
            if self._multiblock and self._concatenation_axis == 0:
                # we have to repeat the list as many times as there are
                # datasets in X
                # closureConc = np.repeat(getC_to_C_idx, len(self._X)).tolist()
                pass
        else:  # a list of lists, check that the number of lists is correct
            if len(getC_to_C_idx) != len(self._X):
                raise ValueError(
                    f"Multiblock dataset: the number of lists of indexes in "
                    f"`getC_to_C_idx` is ({len(getC_to_C_idx)}) and does not match the "
                    f"number of datasets ({len(self._X)}) "
                )
        # check that the number of indexes in each list is correct
        for _getC_to_C_idx in getC_to_C_idx:
            if len(_getC_to_C_idx) > self._n_components:
                raise ValueError(
                    f"The profile has only {self._n_components} species, please check "
                    f"the `getC_to_C_idx` configuration (value:{getC_to_C_idx})"
                )
            elif np.any(_getC_to_C_idx) and (
                len(_getC_to_C_idx) > self._n_components
                or max(_getC_to_C_idx) + 1 > self._n_components
            ):
                raise ValueError(
                    f"The profile has only {self._n_components} species, please check the "
                    f"`getC_to_C_idx` configuration (value:{getC_to_C_idx})"
                )

        return getC_to_C_idx

    @tr.validate("nonnegSpec")
    def _validate_nonnegSpec(self, proposal):
        if self._X_is_missing:
            return proposal.value
        nonnegSpec = proposal.value
        if not self._n_components:  # not initialized or 0
            return nonnegSpec
        if nonnegSpec == "all":
            nonnegSpec = np.arange(self._n_components).tolist()
        elif np.any(nonnegSpec) and (
            len(nonnegSpec) > self._n_components
            or max(nonnegSpec) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `nonnegSpec`configuration (value:{nonnegSpec})"
            )
        return nonnegSpec

    @tr.validate("unimodSpec")
    def _validate_unimodSpec(self, proposal):
        if self._X_is_missing:
            return proposal.value
        unimodSpec = proposal.value
        if not self._n_components:  # not initialized or 0
            return unimodSpec
        if unimodSpec == "all":
            unimodSpec = np.arange(self._n_components).tolist()
        elif np.any(unimodSpec) and (
            len(unimodSpec) > self._n_components
            or max(unimodSpec) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check the "
                f"`unimodSpec`configuration"
            )
        return unimodSpec

    @tr.validate("getSt_to_St_idx")
    def _validate_getSt_to_St_idx(self, proposal):
        if self._X_is_missing:
            return proposal.value
        getSt_to_St_idx = proposal.value
        if not self._n_components:  # not initialized or 0
            return getSt_to_St_idx
        if getSt_to_St_idx == "default":
            getSt_to_St_idx = np.arange(self._n_components).tolist()
        elif (
            len(getSt_to_St_idx) > self._n_components
            or max(getSt_to_St_idx) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `getSt_to_St_idx`  configuration (value:{getSt_to_St_idx})"
            )
        return getSt_to_St_idx

    @tr.observe("_n_components")
    def _n_components_change(self, change):
        # tiggered in _guess_profile
        if self._n_components > 0:
            # perform a validation of default configuration parameters
            # Indeed, if not forced here these parameters are validated only when they
            # are set explicitely.
            # Here is an ugly trick to force this validation. # TODO: better way?
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)

                self.getC_to_C_idx = self.getC_to_C_idx
                self.getSt_to_St_idx = self.getSt_to_St_idx
                self.nonnegConc = self.nonnegConc
                self.nonnegSpec = self.nonnegSpec
                self.unimodConc = self.unimodConc
                self.unimodSpec = self.unimodSpec
                self.closureConc = self.closureConc
                self.closureTarget = self.closureTarget

    @tr.default("_components")
    def _components_default(self):
        if self._fitted:
            # note: _outfit = (C, St, C_constrained, St_unconstrained, extraOutputGetConc, extraOutputGetSpec)
            return self._outfit[1]
        else:
            raise NotFittedError("The model was not yet fitted. Execute `fit` first!")

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    # To see all accessible members it is interesting to use the structure tab of
    # PyCharm
    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        # should be a tuple of profiles or only concentrations/spectra profiles
        profiles = change.new

        if isinstance(profiles, (list, tuple)) and self._Y_preprocessed != []:
            # the starting C and St are already computed
            # (for ex. from a previous run of fit)
            C, St = [item.data for item in profiles]
            self._n_components = C.shape[1]
            # eventually remove mask
            if np.any(self._X_mask):
                masked_rows, masked_columns = self._get_masked_rc(self._X_mask)
                St = St[:, ~masked_columns]
                C = C[~masked_rows]
        else:
            # not passed explicitly, try to guess.
            C, St = self._guess_profile(profiles)

        # we do a last validation
        shape = self._X_preprocessed.shape

        if shape[0] != C.shape[0]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of C do not match those of X.")
        if shape[1] != St.shape[1]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of St do not match those of X.")
        # return the list of C and St data
        # (with mask removed to fit the size of the _X data)
        self._Y_preprocessed = (C, St)

    def _fit(self, X, Y, axis=0):
        # this method is called by the abstract class fit.
        # Input X is a np.ndarray
        # Y is a tuple of guessed profiles (each of them being np.ndarray)
        # So every computation below implies only numpy arrays.

        C, St = Y
        ny, _ = X.shape
        n_components = self._n_components
        change = self.tol + 1
        stdev = X.std()
        niter = 0
        ndiv = 0

        info_("***           ALS optimisation log            ***")
        info_("#iter     RSE / PCA        RSE / Exp      %change")
        info_("-------------------------------------------------")

        # get sklearn PCA with same number of components for further comparison
        pca = decomposition.PCA(n_components=n_components)
        Xtransf = pca.fit_transform(X)
        Xpca = pca.inverse_transform(Xtransf)

        # in case of multiblock, we will need to split C and/or St and possibly use
        # block-specific constraints.
        if self._multiblock and self._concatenation_axis == 0:
            # determine where C should be split
            C_split_indices = [x.shape[0] for x in X][:-1]
            for i, idx in enumerate(C_split_indices):
                C_split_indices[i] += C_split_indices[i - 1] if i > 0 else 0

            # non-negativity constraints are not assumed to be -block-specific

            # unimodality constraints are block-specific if a list of list of indexes is
            # provided
            if np.any(self.unimodConc):
                ...

        # now start the ALS loop
        while change >= self.tol and niter < self.max_iter and ndiv < self.maxdiv:
            niter += 1

            # CONCENTRATION MATRIX
            # --------------------
            C = self._solve_C(St) if niter > 1 else C

            if self._multiblock and self._concatenation_axis == 0:
                C_list = np.split(C, C_split_indices, axis=0)
            else:
                C_list = [C]

            for i, C in enumerate(C_list):

                # Force non-negative concentration
                # ------------------------------------------
                if np.any(self.nonnegConc):
                    C[:, self.nonnegConc] = C[:, self.nonnegConc].clip(min=0)

                # Force unimodal concentration
                # ------------------------------------------
                if np.any(self.unimodConc):
                    C = _unimodal_2D(
                        C,
                        idxes=self.unimodConc,
                        axis=0,
                        tol=self.unimodConcTol,
                        mod=self.unimodConcMod,
                    )

                # Force monotonic increase
                # ------------------------------------------
                if np.any(self.monoIncConc):
                    for s in self.monoIncConc:
                        for curid in np.arange(ny - 1):
                            if C[curid + 1, s] < C[curid, s] / self.monoIncTol:
                                C[curid + 1, s] = C[curid, s]

                # Force monotonic decrease
                # ------------------------------------------
                if np.any(self.monoDecConc):
                    for s in self.monoDecConc:
                        for curid in np.arange(ny - 1):
                            if C[curid + 1, s] > C[curid, s] * self.monoDecTol:
                                C[curid + 1, s] = C[curid, s]

                # Closure
                # ------------------------------------------
                if self.closureConc:
                    if self.closureMethod == "scaling":
                        Q = _lstsq(C[:, self.closureConc], self.closureTarget.T)
                        C[:, self.closureConc] = np.dot(
                            C[:, self.closureConc], np.diag(Q)
                        )
                    elif self.closureMethod == "constantSum":
                        totalConc = np.sum(C[:, self.closureConc], axis=1)
                        C[:, self.closureConc] = (
                            C[:, self.closureConc]
                            * self.closureTarget[:, None]
                            / totalConc[:, None]
                        )

                # external concentration profiles
                # ------------------------------------------
                extraOutputGetConc = []
                if np.any(self.hardConc):
                    _C = self._C_2_NDDataset(C)
                    if self.kwargsGetConc != {} and self.argsGetConc != ():
                        output = self.getConc(
                            _C, *self.argsGetConc, **self.kwargsGetConc
                        )
                    elif self.kwargsGetConc == {} and self.argsGetConc != ():
                        output = self.getConc(_C, *self.argsGetConc)
                    elif self.kwargsGetConc != {} and self.argsGetConc == ():
                        output = self.getConc(_C, **self.kwargsGetConc)
                    else:
                        output = self.getConc(_C)

                    if isinstance(output, tuple):
                        fixedC = output[0]
                        self.argsGetConc = output[1]
                        if len(output) == 3:
                            extraOutputGetConc.append(output[2])
                        else:
                            fixedC = output
                    else:
                        fixedC = output

                    C[:, self.hardConc] = fixedC[:, self.getC_to_C_idx]

            # stores C in C_constrained
            # ------------------------------------------
            C_constrained = C.copy()

            # Compute St
            # -----------
            St = self._solve_St(C)

            # stores St in St_unconstrained
            # ------------------------------------------
            St_unconstrained = St.copy()

            # Force non-negative spectra
            # ------------------------------------------
            if np.any(self.nonnegSpec):
                St[self.nonnegSpec, :] = St[self.nonnegSpec, :].clip(min=0)

            # Force unimodal spectra
            # ------------------------------------------
            if np.any(self.unimodSpec):
                St = _unimodal_2D(
                    St,
                    idxes=self.unimodSpec,
                    axis=1,
                    tol=self.unimodSpecTol,
                    mod=self.unimodSpecMod,
                )

            # External spectral profile
            # ------------------------------------------
            extraOutputGetSpec = []
            if np.any(self.hardSpec):
                _St = self._St_2_NDDataset(St)
                if self.kwargsGetSpec != {} and self.argsGetSpec != ():
                    output = self.getSpec(_St, *self.argsGetSpec, **self.kwargsGetSpec)
                elif self.kwargsGetSpec == {} and self.argsGetSpec != ():
                    output = self.getSpec(_St, *self.argsGetSpecc)
                elif self.kwargsGetSpec != {} and self.argsGetSpec == ():
                    output = self.getSpec(_St, **self.kwargsGetSpec)
                else:
                    output = self.getSpec(_St)

                if isinstance(output, tuple):
                    fixedSt = output[0].data
                    self.argsGetSpec = output[1]
                    if len(output) == 3:
                        extraOutputGetSpec.append(output[2])
                    else:
                        fixedSt = output.data
                else:
                    fixedSt = output.data

                St[self.hardSpec, :] = fixedSt[self.getSt_to_St_idx, :]

            # recompute C for consistency
            # ------------------------------------------
            C = self._solve_C(St)

            # rescale spectra and concentrations
            # ------------------------------------------
            if self.normSpec == "max":
                alpha = np.max(St, axis=1).reshape(self._n_components, 1)
                St = St / alpha
                C = C * alpha.T
            elif self.normSpec == "euclid":
                alpha = np.linalg.norm(St, axis=1).reshape(self._n_components, 1)
                St = St / alpha
                C = C * alpha.T

            # compute residuals
            # ------------------------------------------
            Xhat = C @ St
            stdev2 = np.std(Xhat - X)
            change = 100 * (stdev2 - stdev) / stdev
            stdev = stdev2

            stdev_PCA = np.std(Xhat - Xpca)
            info_(
                f"{niter:3d}{' '*6}{stdev_PCA:10f}{' '*6}"
                f"{stdev2:10f}{' '*6}{change:10f}"
            )

            # check convergence
            # ------------------------------------------

            if change > 0:
                ndiv += 1
            else:
                ndiv = 0
                change = -change

            if change < self.tol:
                info_("converged !")

            if ndiv == self.maxdiv:
                info_(
                    f"Optimization not improved after {self.maxdiv} iterations"
                    f"... unconverged or 'tol' set too small ?"
                )
                info_("Stop ALS optimization.")

            if niter == self.max_iter:
                info_(
                    f"Convergence criterion ('tol') not reached after "
                    f"{ self.max_iter:d} iterations."
                )
                info_("Stop ALS optimization.")

        # return _fit results
        self._components = St
        _outfit = (
            C,
            St,
            C_constrained,
            St_unconstrained,
            extraOutputGetConc,
            extraOutputGetSpec,
        )
        return _outfit

    def _transform(self, X=None):
        # X is ignored for MCRALS
        return self._outfit[0]

    def _inverse_transform(self, X_transform=None):
        # X_transform is ignored for MCRALS
        return np.dot(self._transform(), self._components)

    def _get_components(self):
        return self._components

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def fit(self, X, Y, concatenation_axis=0):
        """
        Fit the MCRALS model on an X dataset using initial concentration or spectra.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        Y : :term:`array-like` or list of :term:`array-like`
            Initial concentration or spectra.
        concatenation_axis : int, optional
            Axis along which the concatenation is carried out. Default is 0.
        Returns
        -------
        %(analysis_fit.returns)s

        See Also
        --------
        %(analysis_fit.see_also)s
        """
        self._concatenation_axis = concatenation_axis
        return super().fit(X, Y)

    @_docstring.dedent
    def fit_transform(self, X, Y, **kwargs):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        Y : :term:`array-like` or list of :term:`array-like`
            Initial concentration or spectra.
        %(kwargs)s

        Returns
        -------
        %(analysis_transform.returns)s

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s
        """
        return super().fit_transform(X, Y, **kwargs)

    @_docstring.dedent
    def inverse_transform(self, X_transform=None, **kwargs):
        """
        Transform data back to its original space.

        In other words, return an input `X_original` whose reduce/transform would be X.

        Parameters
        ----------
        %(analysis_inverse_transform.parameters)s

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`\ , :term:`n_features`\ ).

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s
        """
        return super().inverse_transform(X_transform, **kwargs)

    @property
    def C(self):
        """
        The final concentration profiles.
        """
        C = self.transform()
        if self._multiblock:
            X_name = ", ".join([x.name for x in self.X])
            if isinstance(C, tuple):
                for c, x in zip(C, self.X):
                    c.name = f"MCR-ALS concentration profile of {x.name}"
            else:
                C.name = f"MCA-ALS concentration profile of {X_name}"
        else:
            C.name = f"Pure concentration profile, mcr-als of {self.X.name}"

        return C

    @property
    def St(self):
        """
        The final spectra profiles.
        """
        St = self.components
        if self._multiblock:
            X_name = ", ".join([x.name for x in self.X])
            if isinstance(St, tuple):
                for st, x in zip(St, self.X):
                    st.name = f"MCR-ALS spectral profiles of {x.name}"
            else:
                St.name = f"MCR-ALS spectral profiles of {X_name}"
        else:
            St.name = f"MCR-ALS spectral profiles of {self.X.name}"
        return St

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def C_constrained(self):
        """
        The constrained concentration profiles, i.e. after applying the hard and soft constraints.
        """
        return self._outfit[2]

    @property
    @deprecated(replace="C_constrained")
    def C_hard(self):
        """
        Deprecated. Equivalent to `C_constrained`.
        """
        return self.C_constrained

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def St_unconstrained(self):
        r"""
        The soft spectra profiles.

        Spectra obtained after solving :math:`C_{\textrm{constrained}} \cdot St = X`
        for :math:`St`\ .
        """
        return self._outfit[3]

    @property
    @deprecated(replace="St_unconstrained")
    def S_soft(self):
        """
        Deprecated. Equivalent to `C_constrained`.
        """
        return self.St_unconstrained

    @property
    def extraOutputGetConc(self):
        """
        The extra outputs of the external function used to get concentrations.
        """
        return self._outfit[4]

    @property
    def extraOutputGetSpec(self):
        """
        The extra outputs of the external function used to get spectra.
        """
        return self._outfit[5]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


# LS solvers for W in the linear matrix equation X @ W = Y
def _lstsq(X, Y, rcond=None):
    # Least-squares solution to a linear matrix equation X @ W = Y
    # Return W
    W = np.linalg.lstsq(X, Y, rcond)[0]
    return W


def _nnls(X, Y, withres=False):
    # Non negative least-squares solution to a linear matrix equation X @ W = Y
    # Return W >= 0
    # TODO: look at may be faster algorithm: see: https://gist.github.com/vene/7224672
    nsamp, nfeat = X.shape
    nsamp, ntarg = Y.shape
    W = np.empty((nfeat, ntarg))
    residuals = 0
    for i in range(ntarg):
        Y_ = Y[:, i]
        W[:, i], res = scipy.optimize.nnls(X, Y_)
        residuals += res**2
    return (W, np.sqrt(residuals)) if withres else W


def _pnnls(X, Y, nonneg=[], withres=False):
    # Least-squares  solution to a linear matrix equation X @ W = Y
    # with partial nonnegativity (indicated by the nonneg list of targets)
    # Return W with eventually some column non negative.
    nsamp, nfeat = X.shape
    nsamp, ntarg = Y.shape
    W = np.empty((nfeat, ntarg))
    residuals = 0
    for i in range(ntarg):
        Y_ = Y[:, i]
        if i in nonneg:
            W[:, i], res = scipy.optimize.nnls(X, Y_)
        else:
            W[:, i], res = np.linalg.lstsq(X, Y_)[:2]
        residuals += res**2
    return (W, np.sqrt(residuals)) if withres else W


def _unimodal_2D(a, axis, idxes, tol, mod):
    # Force unimodality on given lines or columnns od a 2D ndarray
    #
    # a: ndarray
    #
    # axis: int
    #     axis along which the correction is applied
    #
    # idxes: list of int
    #     indexes at which the correction is applied
    #
    # mod : str
    #     When set to `"strict"`\ , values deviating from unimodality are reset to the
    #     value of the previous point. When set to `"smooth"`\ , both values (deviating
    #     point and previous point) are modified to avoid "steps" in the profile.
    #
    # tol: float
    #     Tolerance parameter for unimodality. Correction is applied only if:
    #     `a[i] > a[i-1] * unimodTol`  on a decreasing branch of profile,
    #     `a[i] < a[i-1] * unimodTol`  on an increasing branch of profile.

    if axis == 0:
        a_ = a
    elif axis == 1:
        a_ = a.T

    for col, idx in zip(a_[:, idxes].T, idxes):
        a_[:, idx] = _unimodal_1D(col, tol, mod)

    return a


def _unimodal_1D(a: np.ndarray, tol: str, mod: str) -> np.ndarray:
    # force unimodal concentration
    #
    # makes a vector unimodal
    #
    # Parameters
    # ----------
    # a : 1D ndarray
    #
    # mod : str
    #     When set to `"strict"`\ , values deviating from unimodality are reset to the value
    #     of the previous point. When set to `"smooth"`\ , both values (deviating point and
    #     previous point) are modified to avoid "steps"
    #     in the profile.
    #
    # tol: float
    #     Tolerance parameter for unimodality. Correction is applied only if:
    #     `a[i] > a[i-1] * unimodTol`  on a decreasing branch of profile,
    #     `a[i] < a[i-1] * unimodTol`  on an increasing branch of profile.

    maxid = np.argmax(a)
    curmax = max(a)
    curid = maxid

    while curid > 0:
        # run backward
        curid -= 1
        if a[curid] > curmax * tol:
            if mod == "strict":
                a[curid] = a[curid + 1]
            if mod == "smooth":
                a[curid] = (a[curid] + a[curid + 1]) / 2
                a[curid + 1] = a[curid]
                curid = curid + 2
        curmax = a[curid]

    curid = maxid
    curmax = a[maxid]
    while curid < len(a) - 1:
        curid += 1
        if a[curid] > curmax * tol:
            if mod == "strict":
                a[curid] = a[curid - 1]
            if mod == "smooth":
                a[curid] = (a[curid] + a[curid - 1]) / 2
                a[curid - 1] = a[curid]
                curid = curid - 2
        curmax = a[curid]
    return a
