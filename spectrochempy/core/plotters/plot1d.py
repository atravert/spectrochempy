# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

"""
Module containing 1D plotting function(s)

"""
import numpy as np
from matplotlib.ticker import MaxNLocator

from spectrochempy.application import plotoptions
from spectrochempy.core.plotters.utils import make_label
from spectrochempy.utils import is_sequence


__all__ = ['plot_1D','plot_lines','plot_scatter', 'plot_multiple']

_methods = __all__[:]

# plot lines (default) --------------------------------------------------------

def plot_lines(source, **kwargs):
    """
    Plot a 1D dataset with solid lines by default.

    Alias of plot_1D (with `kind` argument set to ``lines``.

    """
    kwargs['kind'] = 'lines'
    ax = plot_1D(source, **kwargs)
    return ax


# plot scatter ----------------------------------------------------------------

def plot_scatter(source, **kwargs):
    """
    Plot a 1D dataset as a scatter plot (points can be added on lines).

    Alias of plot_1D (with `kind` argument set to ``scatter``.

    """
    kwargs['kind'] = 'scatter'
    ax = plot_1D(source, **kwargs)
    return ax


# plot lines (default) --------------------------------------------------------

def plot_lines(source, **kwargs):
    """
    Plot a 1D dataset with solid lines by default.

    Alias of plot_1D (with `kind` argument set to ``lines``.

    """
    kwargs['kind'] = 'lines'
    ax = plot_1D(source, **kwargs)
    return ax


# plot multiple ----------------------------------------------------------------

def plot_multiple(sources, kind='scatter', lines=True,
                  labels = None, **kwargs):
    """
    Plot a series of 1D datasets as a scatter plot
    (points can be added on lines).

    Parameters
    ----------

    sources : a list of ndatasets

    kind : `str` among [scatter, lines]

    lines : `bool`, optional, default=``True``

        if kind is scatter, this flag tells to draw also the lines
        between the marks.

    labels : a list of str, optional

        labels used for the legend.

    **kwargs : other parameters that will be passed to the plot1D function

    """
    if not is_sequence(sources):
        # we need a sequence. Else it is a single plot.
        return sources.plot(**kwargs)

    if not is_sequence(labels) or len(labels)!=len(sources):
        # we need a sequence of labels of same lentgh as sources
        raise ValueError('the list of labels must be of same length '
                         'as the sources list')

    for source in sources:
        if source.ndim > 1:
            raise NotImplementedError('plot multiple is designed to work on '
                                      '1D dataset only. you may achieved '
                                      'several plots with '
                                      'the `hold` parameter as a work around '
                                      'solution')

    hold = False
    for s in sources : #, colors, markers):

        ax = s.plot(kind= kind, lines=True, hold=hold, **kwargs)
        hold = True
        # hold is necessary for the next plot to say
        # that we will plot on the same figure

    legend = kwargs.get('legend', None)
    if legend is not None:
        leg = ax.legend(ax.lines, labels, shadow=True, loc=legend,
                        frameon=True, facecolor='lightyellow')
    return ax


# ------------------------------------------------------------------------------
# plot_1D
# ------------------------------------------------------------------------------
def plot_1D(source, **kwargs):
    """
    Plot of one-dimensional data

    Parameters
    ----------
    new: :class:`~spectrochempy.core.ddataset.nddataset.NDDataset` to plot

    kind: `str` [optional among ``lines`, ``scatter``]

    style : str, optional, default = 'notebook'
        Matplotlib stylesheet (use `available_style` to get a list of available
        styles for plotting

    reverse: `bool` or None [optional, default= None/False
        In principle, coordinates run from left to right, except for wavenumbers
        (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
        will try to guess. But if reverse is set, then this is the
        setting which will be taken into account.

    hold: `bool` [optional, default=`False`]

        If true hold the current figure and ax until a new plot is performed.

    data_only: `bool` [optional, default=`False`]

        Only the plot is done. No addition of axes or label specifications
        (current if any or automatic settings are kept.

    imag: `bool` [optional, default=`False`]

        Show imaginary part. By default only the real part is displayed.

    show_complex: `bool` [optional, default=`False`]

        Show both real and imaginary part.
        By default only the real part is displayed.

    dpi: int, optional
        the number of pixel per inches

    figsize: tuple, optional, default is (3.4, 1.7)
        figure size

    fontsize: int, optional
        font size in pixels, default is 10

    imag: bool, optional, default False
        By default real part is shown. Set to True to display the imaginary part

    xlim: tuple, optional
        limit on the horizontal axis

    zlim or ylim: tuple, optional
        limit on the vertical axis

    color or c: matplotlib valid color, optional

        color of the line #TODO: a list if several line

    linewidth or lw: float, optional

        line width

    linestyle or ls: str, optional

        line style definition

    xlabel: str, optional
        label on the horizontal axis

    zlabel or ylabel: str, optional
        label on the vertical axis

    showz: bool, optional, default=True
        should we show the vertical axis

    plot_model:Bool,

        plot model data if available

    modellinestyle or modls: str,

        line style of the model

    offset: float,

        offset of the model individual lines

    commands: str,

        matplotlib commands to be executed

    show_zero: boolean, optional

        show the zero basis

    savename: str,

        name of the file to save the figure

    savefig: Bool,

        save the fig if savename is defined,
        should be executed after all other commands

    vshift: float, optional
        vertically shift the line from its baseline


    kwargs : additional keywords

    """
    # where to plot?
    # ---------------
    new = source.copy()

    new._figure_setup(**kwargs)
    ax = new.ndaxes['main']

    # -------------------------------------------------------------------------
    # plot the source
    # -------------------------------------------------------------------------

    kind = kwargs.pop('kind','lines')
    lines = kwargs.pop('lines',False) # in case it is scatter,
                                      # we can also show the lines
    lines = kind=='lines' or lines
    scatter = kind=='scatter' and not lines
    scatlines = kind=='scatter' and lines

    show_complex = kwargs.pop('show_complex', False)

    color = kwargs.get('color', kwargs.get('c', None))    # default to rc
    lw = kwargs.get('linewidth', kwargs.get('lw', None))  # default to rc
    ls = kwargs.get('linestyle', kwargs.get('ls', None))  # default to rc

    marker = kwargs.get('marker', kwargs.get('m', None))  # default to rc
    markersize = kwargs.get('markersize', kwargs.get('ms', 5.))
    markevery = kwargs.get('markevery', kwargs.get('me', 1))

    # abscissa axis
    x = new.x

    # ordinates (by default we plot real part of the data)
    if not kwargs.pop('imag', False) or kwargs.get('show_complex', False):
        z = new.real
    else:
        z = new.imag

    # offset
    offset = kwargs.pop('offset', 0.0)
    z = z - offset

    # plot_lines
    # -----------------------------
    if scatlines:
        line, = ax.plot(x.data, z.masked_data,  markersize = markersize,
                                        markevery = markevery)
    elif scatter:
        line, = ax.plot(x.data, z.masked_data, lw=0,  markersize = markersize,
                                        markevery = markevery)
    elif lines:
        line, = ax.plot(x.data, z.masked_data)

    if show_complex and lines:
        zimag = new.imag
        ax.plot(x.data, zimag.masked_data, ls='--')

    if kwargs.get('plot_model', False):
        modeldata = new.modeldata                   #TODO: what's about mask?
        ax.plot(x.data, modeldata.T, ls=':', lw='2')   #TODO: improve this!!!

    # line attributes
    if lines and color:
        line.set_color(color)

    if lines and lw:
        line.set_linewidth(lw)

    if lines and ls:
        line.set_linestyle(ls)


    # -------------------------------------------------------------------------
    # axis limits and labels
    # -------------------------------------------------------------------------

    if kwargs.get('data_only', False):
        # if data only (we will not set axes and labels
        # it was probably done already in a previous plot
        new._plot_resume(source, **kwargs)
        return True

    # abscissa limits?
    xl = [x.data[0], x.data[-1]]
    xl.sort()
    xlim = list(kwargs.get('xlim', xl))
    xlim.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    # reversed axis?
    reverse = new.x.is_reversed
    if kwargs.get('reverse', reverse):
        xlim.reverse()

    # ordinates limits?
    amp = np.ma.ptp(z.masked_data) / 100.
    zl = [np.ma.min(z.masked_data)-amp, np.ma.max(z.masked_data)+amp]
    zlim = list(kwargs.get('zlim', kwargs.get('ylim', zl)))
    zlim.sort()

    # set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)

    number_x_labels = plotoptions.number_of_x_labels  # get from config
    number_y_labels = plotoptions.number_of_y_labels

    ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
    ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------

    # x label

    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(new.x, 'x')
    ax.set_xlabel(xlabel)

    # z label
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        zlabel = make_label(new, 'z')

    ax.set_ylabel(zlabel)

    # do we display the ordinate axis?
    if kwargs.get('show_z', True):
        ax.set_ylabel(zlabel)
    else:
        ax.set_yticks([])

    # do we display the zero line
    if kwargs.get('show_zero', False):
        ax.haxlines()

    new._plot_resume(source, **kwargs)

    return ax

if __name__ == '__main__':

    from spectrochempy.api import *
    source = NDDataset.read_omnic(
            os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))[:,::100]

    source.plot_stack()

    sources = [source[0], source[10], source[20], source[50], source[53]]
    labels = ['sample {}'.format(label) for label in
              ["S1", "S10", "S20", "S50", "S53"]]

    plot_multiple(kind = 'scatter', sources=sources, labels=labels, legend='best')


    plt.show()