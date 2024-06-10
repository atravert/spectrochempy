# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of Multivariate Curve Resolution - Hard Kinetic Modeling
"""

import numpy as np
import traitlets as tr
from sklearn.decomposition import truncatedSVD

from spectrochempy import ActionMassKinetics
from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring

__all__ = ["MCRHKM"]
__configurables__ = ["MCRHKM"]


# ======================================================================================
# class MCRHKM
# ======================================================================================
@signature_has_configurable_traits
class MCRHKM(DecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "MCRHKM")

    __doc__ = _docstring.dedent(
        """
    Multivariate Curve Resolution with Hard Kinetic Modeling (HKM).

    Multivariate Curve Resolution based on PCA and kinetic models.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_MCRHKM)s
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

    kineticModel = tr.Instance(ActionMassKinetics)

    _n_components = tr.Int(
        kineticModel.nspecies,
        help=(
            "Number of componnents/species in spectra. Must be less or equal to the "
            "number of species of the kinetic model."
        ),
    ).tag(config=True)

    epsC = tr.Float(
        0.0,
        help=("Allows small negative elements in the concentraion matrix C)."),
    ).tag(config=True)

    epsSt = tr.Float(
        0.0,
        help=("Allows small negative elements in the spectra matrix St."),
    ).tag(config=True)

    nonNegSpecWeight = tr.Float(
        1.0,
        help=(
            "Weight of non-negativity constaint on spectra. If set < -0.5, no constraint is applied"
        ),
    ).tag(config=True)

    nonNegConcWeight = tr.Float(
        1.0,
        help=(
            "Weight of non-negativity constaint on concentraion. If set < -0.5, no constraint is applied"
        ),
    ).tag(config=True)

    reconstructionWeight = tr.Float(
        1.0,
        help=(
            "Weight of reconstruction constaint. If set < -0.5, no constraint is applied"
        ),
    ).tag(config=True)

    kineticFitWeight = tr.Float(
        0.1,
        help=("Weight of kinetic fit. If set < -0.5, no constraint is applied"),
    ).tag(config=True)

    algorithm_SVD = tr.Enum(["arpack", "randomized"], default_value="randomized")
    # todo: add other parameters for SVD

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):
        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------

    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        # we need the X.y axis (reaction times)
        # get it from the self._X nddataset (where masked data have been removed)
        X = change.new
        self._times = X.coordset[X.dims[0]]
        # use data only
        self._X_preprocessed = X.data

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    def _fit(self, X):
        # X is the data array to fit

        # Perform standard SVD
        SVD = truncatedSVD(self._n_components, algorithm=self.algorithm_SVD)
        X_svd_transformed = SVD.fit_transform(X)
        # Truncate matrices
        U = X_svd_transformed / SVD.singular_values_
        # Sigma = np.diag(SVD.singular_values_)
        Vt = SVD.components_

        # unify orientation of svd factors....from FACPACK m.file
        if np.min(Vt[0, :]) < 0:
            Vt[0, :] = -Vt[0, :]
            U[:, 0] = -U[:, 0]

        for i in range(1, self._n_components):
            idx = np.argmax(np.abs(Vt[i, :]))
            if np.sign(Vt[i, idx]) == -1:
                Vt[i, :] = -Vt[i, :]
                U[:, i] = -U[:, i]

        def objective(
            kin_param,
            dict_param_to_optimize,
        ):
            for param, item in zip(kin_param, dict_param_to_optimize):
                dict_param_to_optimize[item] = param

            # # step 1 (solve initial value problem)
            # self.kineticModel._modify_kinetics(dict_param_to_optimize, None)
            # Ckin = self.kineticModel.integrate(self._times)
            #
            # # step 2 (solve (US)*T = Cdgl for T, T^+ =(US)^+ * Cdgl)
            # US = np.dot(U, Sigma)
            # pUS = np.linalg.pinv(US)
            # pT = np.dot(pUS, Ckin)
            #
            # # step 3 (calculate factor C)
            # C = np.dot(US, pT)
            #
            # Cerr = C - Ckin  # %%%%error of kinetic fit (absolute)
            #
            # # calculate factor S
            # T = np.linalg.pinv(pT)
            # S = np.dot(T, Vt)
            # # RR = D - np.dot(C, S)
            #
            # # scaling
            # MaC = np.diag(
            #     np.max(C, axis=0) ** (-1)
            # )  # scaling factor for C (maximum of each profile = 1)
            # MaS = np.diag(
            #     np.max(S, axis=1) ** (-1)
            # )  # scaling factor for S (maximum of each profile = 1)
            # C = np.dot(C, MaC)
            # S = np.dot(MaS, S)
            #
            # # step 4 (apply constraints)
            # R = {}
            # idx = 1
            # # C>0

        #     if W[0] > -0.5:
        #         Temp = W[0] * np.minimum(0, C + self.epsC)
        #         R[idx] = Temp.flatten()
        #         idx += 1
        #
        #     # S>0
        #     if W[1] > -0.5:
        #         Temp = W[1] * np.minimum(0, S + self.epsA)
        #         R[idx] = Temp.flatten()
        #         idx += 1
        #
        #     # Reconstruction
        #     if W[2] > -0.5:
        #         Temp = W[2] * (np.dot(pT, T) - np.eye(self._n_components)
        #         R[idx] = Temp.flatten()
        #         idx += 1
        # # check if the kinetic model is defined
        # return _outfit
