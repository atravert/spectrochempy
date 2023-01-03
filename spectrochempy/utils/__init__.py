# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Package containing various utilities classes and functions.
"""
# some useful constants
# ------------------------------------------------------------------
# import numpy as np

# masked arrays
# ------------------------------------------------------------------
from numpy.ma.core import (
    masked as MASKED,
    nomask as NOMASK,
    MaskedArray,
    MaskedConstant,
)  # noqa: F401

# import util files content
# ------------------------------------------------------------------

from spectrochempy.utils.fake import *
from spectrochempy.utils.print import *
from spectrochempy.utils.file import *
from spectrochempy.utils.jsonutils import *
from spectrochempy.utils.misc import *
from spectrochempy.utils.packages import *
from spectrochempy.utils.plots import *
from spectrochempy.utils.system import *
from spectrochempy.utils.traits import *
from spectrochempy.utils.zip import *
from spectrochempy.utils.exceptions import *
from spectrochempy.utils.version import *
from spectrochempy.utils.print_versions import *
