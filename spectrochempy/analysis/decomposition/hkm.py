# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of Principal Component Analysis (using scikit-learn library)
"""

import traitlets as tr

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
    Multivariate Curve Resolution with Hard Kinetic Modeling (MCRHKM).

    Multivariate Curve Resolution based on truncated SVD and kintic models.

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
