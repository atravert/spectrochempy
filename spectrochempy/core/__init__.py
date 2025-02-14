# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Package defining the *core* methods of the  `SpectroChemPy` API.

Most the API methods such as plotting, processing, analysis, etc...

isort:skip_file
"""
# flake8: noqa

__all__ = []  # modified below

from spectrochempy.utils.timeutils import timeit

# ======================================================================================
# loading module libraries
# here we also construct the __all__ list automatically
# ======================================================================================

with timeit("application"):
    from spectrochempy.application import (
        version,
        release,
        copyright,
        license,
        release_date,
        authors,
        contributors,
        url,
        DEBUG,
        WARNING,
        ERROR,
        CRITICAL,
        INFO,
        error_,
        warning_,
        debug_,
        info_,
        preferences,
        plot_preferences,
        description,
        long_description,
        config_dir,
        config_manager,
        reset_preferences,
    )  # noqa: E402

    def set_loglevel(level=WARNING):
        if isinstance(level, str):
            import logging

            level = getattr(logging, level)
        preferences.log_level = level

    def get_loglevel():
        return preferences.log_level

    __all__ += [
        # Helpers
        "DEBUG",
        "WARNING",
        "ERROR",
        "CRITICAL",
        "INFO",
        "error_",
        "warning_",
        "debug_",
        "info_",
        "preferences",
        "plot_preferences",
        "config_manager",
        "config_dir",
        "reset_preferences",
        "set_loglevel",
        "get_loglevel",
        # Info
        "copyright",
        "version",
        "release",
        "license",
        "url",
        "release_date",
        "authors",
        "contributors",
        "description",
        "long_description",
    ]


# constants
# ---------
with timeit("constants"):
    from spectrochempy.utils.plots import show
    from spectrochempy.utils.constants import (
        MASKED,
        NOMASK,
        EPSILON,
        INPLACE,
    )  # noqa: E402
    from spectrochempy.utils.print_versions import show_versions  # noqa: E402

    __all__ += ["show", "MASKED", "NOMASK", "EPSILON", "INPLACE", "show_versions"]

# units
# -----
with timeit("units"):
    from spectrochempy.core.units import *  # noqa: E402,F403,F401

    __all__ += [
        "Unit",
        "Quantity",
        "ur",
        "set_nmr_context",
        "DimensionalityError",
    ]

# dataset
# -------
with timeit("dataset"):
    from spectrochempy.core.dataset import api  # noqa: E402
    from spectrochempy.core.dataset.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# plotters
# --------
with timeit("plotter"):
    from spectrochempy.core.plotters import api  # noqa: E402
    from spectrochempy.core.plotters.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# readers
# -------
with timeit("readers"):
    from spectrochempy.core.readers import api  # noqa: E402
    from spectrochempy.core.readers.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# writers
# -------
with timeit("writers"):
    from spectrochempy.core.writers import api  # noqa: E402
    from spectrochempy.core.writers.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# project
# -------
with timeit("project"):
    from spectrochempy.core.project.project import Project  # noqa: E402,F403,F401

    __all__ += ["Project"]

# script
# ------
with timeit("script"):
    from spectrochempy.core.script import *  # noqa: E402,F403,F401

    __all__ += ["Script", "run_script", "run_all_scripts"]

# widgets
# -------
with timeit("widgets"):
    from spectrochempy.widgets import api  # noqa: E402
    from spectrochempy.widgets.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__


# analysis
# --------
with timeit("analysis"):
    from spectrochempy.analysis import api  # noqa: E402
    from spectrochempy.analysis.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# processing
# ----------
with timeit("processing"):
    from spectrochempy.processing import api  # noqa: E402
    from spectrochempy.processing.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# START THE app
# -------------
with timeit("start app"):
    from spectrochempy.application import app

    _started = app.start()
