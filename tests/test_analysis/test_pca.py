# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B)
# See full LICENSE agreement in the root directory
# =============================================================================




""" Tests for the PCA module

"""
from spectrochempy.api import Pca

# test pca
#---------

def test_pca(IR_source_2D):

    source = IR_source_2D.copy()
    pca = Pca(source)

    print(pca)
    pca.printev(npc=5)

    assert str(pca)[:3] == '\nPC'

    #TODO: check those : it seems that colors are not correct!
    #pca.screeplot(npc=5)

    #pca.scoreplot(pcs=(0,1))
