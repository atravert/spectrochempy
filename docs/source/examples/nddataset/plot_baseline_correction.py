# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
NDDataset baseline correction
==============================

In this example, we perform a baseline correction of a 2D NDDataset
interactively, using the `multivariate` method and a `pchip` interpolation.

"""

###############################################################################
# As usual we start by importing the useful library, and at least  the
# spectrochempy scp.

from spectrochempy import scp
import os

###############################################################################
# Load data

nd = scp.NDDataset.read_omnic(os.path.join(scp.preferences.datadir,
                                           'irdata', 'NH4Y-activation.SPG'))

###############################################################################
# Do some slicing to keep only the interesting region

ndp = (nd - nd[-1])[:, 1290.0:5999.0]
# Important:  notice that we use floating point number
# integer would mean points, not wavenumbers!

###############################################################################
# Define the BaselineCorrection object.

ibc = scp.BaselineCorrection(ndp, axis=-1,
                         method='multivariate',
                         interpolation='pchip',
                         npc=5, zoompreview=3)

###############################################################################
# Launch the interactive view, using the `BaselineCorrection.run` method:

ranges = []  # not predefined range
span = ibc.run(*ranges)

scp.show()

###############################################################################
# print the corrected dataset

print(ibc.corrected)








# =============================================================================
if __name__ == '__main__':
    pass
