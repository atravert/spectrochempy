# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================



from traitlets import TraitError

from spectrochempy.utils.testing import (raises)
from spectrochempy.utils.traittypes import HasTraits, Range


def test_range():

    class MyClass(HasTraits):
        r = Range()  # Initialized with some default values

    c = MyClass()
    c.r = [10, 5]
    assert c.r == [5,10]
    with raises(TraitError):
        c.r = [10, 5, 1]