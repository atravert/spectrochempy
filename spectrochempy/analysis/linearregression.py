# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of least squares Linear Regression.
"""
import traitlets as tr

from spectrochempy.analysis.abstractanalysis import LinearRegressionAnalysis

__all__ = ["LSTSQ", "NNLS"]
__configurables__ = ["LSTSQ", "NNLS"]


# ======================================================================================
# class LSTSQ
# ======================================================================================
class LSTSQ(LinearRegressionAnalysis):

    name = "LSTSQ"
    description = "Ordinary Least Squares Linear Regression"


# ======================================================================================
# class NNLS
# ======================================================================================
class NNLS(LinearRegressionAnalysis):
    name = "NNLS"
    description = "Non-Negative Least Squares Linear Regression"

    positive = tr.Bool(
        default_value=True,
        help="When set to ``True``, forces the coefficients to be positive. This"
        "option is only supported for dense arrays.",
    ).tag(config=True)
