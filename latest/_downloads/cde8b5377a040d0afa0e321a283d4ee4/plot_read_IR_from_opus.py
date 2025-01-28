# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Loading Bruker OPUS files
==========================

Here we load an experimental Bruker OPUS files and plot it.

"""
# %%

import spectrochempy as scp

Z = scp.read_opus(
    ["test.0000", "test.0001", "test.0002", "test.0003"], directory="irdata/OPUS"
)
print(Z)

# %%
# plot it

_ = Z.plot()

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
