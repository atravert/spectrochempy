# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module implement the PCA (Principal Component Analysis) class.
"""

__all__ = ['PCA']

__dataset_methods__ = ['PCA']

# ----------------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------------
import numpy as np
from scipy.special import gammaln
from traitlets import HasTraits, Instance
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from spectrochempy.dataset.nddataset import NDDataset, CoordSet
from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.core.analysis.svd import SVD
from spectrochempy.core.processors.npy import diag, dot
from spectrochempy.application import plotter_preferences
from spectrochempy.utils import docstrings, NRed, NBlue


# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------


# ============================================================================
# class PCA
# ============================================================================

class PCA(HasTraits):
    """
    Principal Component Analysis

    This class performs a Principal Component Analysis of a |NDDataset|,
    *i.e.*, a linear dimensionality reduction using Singular Value
    Decomposition (`SVD`)
    of the data to perform its projection to a lower dimensional space.

    The reduction of a dataset :math:`X` with shape (`M`,`N`) is achieved
    using the decomposition: :math:`X = S.L^T`, where
    :math:`S` is the score's matrix with shape (`M`, `n_pc`) and :math:`L^T` is
    the transposed loading's matrix with shape (`n_pc`, `N`).

    If the dataset `X` contains masked values, these values are silently
    ignored in the calculation.

    """
    _LT = Instance(NDDataset)
    _S = Instance(NDDataset)
    _X = Instance(NDDataset)

    ev = Instance(NDDataset)
    """|NDDataset| - Explained variances (The eigenvalues of the covariance 
    matrix)."""

    ev_ratio = Instance(NDDataset)
    """|NDDataset| - Explained variance per singular values."""

    ev_cum = Instance(NDDataset)
    """|NDDataset| - Cumulative Explained Variances."""

    # ........................................................................

    @docstrings.dedent
    def __init__(self, X,
                 centered=True,
                 standardized=False,
                 scaled = False):
        """
        Parameters
        ----------
        %(SVD.parameters.X)s
        centered : bool, optional, default:True
            If True the data are centered around the mean values:
            :math:`X' = X - mean(X)`.
        standardized : bool, optional, default:False
            If True the data are scaled to unit standard deviation:
            :math:`X' = X / \sigma`.
        scaled : bool, optional, default:False
            If True the data are scaled in the interval [0-1]:
            :math:`X' = (X - min(X)) / (max(X)-min(X))`

        Examples
        --------

        .. plot::
            :include-source:

            from spectrochempy.scp import *
            dataset = upload_IRIS()
            pca = PCA(dataset, centered=True)
            LT, S = pca.transform(n_pc='auto')
            _ = pca.screeplot()
            _ = pca.scoreplot(1,2, color_mapping='labels')
            show()

        """

        self._X = X

        Xsc = X.copy()

        # mean center the dataset
        # -----------------------
        self._centered = centered
        if centered:
            self._center = center = np.mean(X, axis=0)
            Xsc = X - center
            Xsc.title = "centered %s"% X.title

        # Standardization
        # ---------------
        self._standardized = standardized
        if standardized:
            self._std = np.std(Xsc, axis=0)
            Xsc /= self._std
            Xsc.title = "standardized %s" % Xsc.title

        # Scaling
        # -------
        self._scaled = scaled
        if scaled:
            self._min = np.min(Xsc, axis=0)
            self._ampl = np.ptp(Xsc, axis=0)
            Xsc -= self._min
            Xsc /= self._ampl
            Xsc.title = "scaled %s" % Xsc.title

        self._Xscaled = Xsc

        # perform SVD
        # -----------
        svd = svd = SVD(Xsc)
        sigma = diag(svd.s)
        U = svd.U
        VT = svd.VT

        # select n_pc loadings & compute scores
        # --------------------------------------------------------------------

        # loadings

        LT = VT
        LT.title = 'Loadings (L^T) of ' + X.name
        LT.history = 'created by PCA'

        # scores

        S = dot(U, sigma)
        S.title = 'scores (S) of ' + X.name
        S.coordset = CoordSet(X.y, Coord(None,
                          labels=['#%d' % (i + 1) for i in range(svd.s.size)],
                              title='principal component'))

        S.description = 'scores (S) of ' + X.name
        S.history = 'created by PCA'

        self._LT = LT
        self._S = S

        # other attributes
        # ----------------

        self.ev = svd.ev
        self.ev.x.title = 'PC #'

        self.ev_ratio= svd.ev_ratio
        self.ev_ratio.x.title = 'PC #'

        self.ev_cum = svd.ev_cum
        self.ev_cum.x.title = 'PC #'

        return

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __str__(self, n_pc=5):

        s = '\nPC\t\tEigenvalue\t\t%variance\t' \
            '%cumulative\n'
        s += '   \t\tof cov(X)\t\t per PC\t' \
             '     variance\n'
        for i in range(n_pc):
            tuple = (
            i, self.ev.data[i], self.ev_ratio.data[i], self.ev_cum.data[i])
            s += '#{}  \t{:8.3e}\t\t {:6.3f}\t      {:6.3f}\n'.format(*tuple)

        return s

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    def _get_n_pc(self, n_pc=None):

        max_n_pc = self.ev.size
        if n_pc is None:
            n_pc = max_n_pc
            return n_pc
        elif isinstance(n_pc, int):
            n_pc = min(n_pc, max_n_pc)
            return n_pc
        elif n_pc == 'auto':
            M, N = self._X.shape
            if M >= N:
                n_pc = self._infer_pc_()
                return n_pc
            else:
                warnings.warn('Cannot use `auto` if n_observations < '
                              'n_features. Try with threshold 0.9999')
                n_pc= 0.9999

        if 0 < n_pc < 1.0:
            # number of PC for which the cumulated explained variance is
            # less than a given ratio
            n_pc = np.searchsorted(self.ev_cum.data / 100., n_pc) + 1
            return n_pc
        else:
            raise ValueError('could not get a valid number of components')



    def _assess_dimension_(self, rank):
        """Compute the likelihood of a rank ``rank`` dataset
        The dataset is assumed to be embedded in gaussian noise having
        spectrum ``spectrum`` (here, the explained variances `ev` ).

        Parameters
        ----------
        rank : int
            Tested rank value.

        Returns
        -------
        float
            The log-likelihood.

        Notes
        -----
        This implements the method of Thomas P. Minka:
        Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604.
        Copied and modified from scikit-learn.decomposition.pca (BSD-3 license)

        """
        spectrum = self.ev.data
        M, N = self._X.shape

        if rank > len(spectrum):
            raise ValueError("The tested rank cannot exceed the rank of the"
                             " dataset")

        pu = -rank * np.log(2.)
        for i in range(rank):
            pu += (gammaln((N - i) / 2.) - np.log(np.pi) * (
            N - i) / 2.)

        pl = np.sum(np.log(spectrum[:rank]))
        pl = -pl * M / 2.

        if rank == N:
            pv = 0
            v = 1
        else:
            v = np.sum(spectrum[rank:]) / (N - rank)
            pv = -np.log(v) * M * (N - rank) / 2.

        m = N * rank - rank * (rank + 1.) / 2.
        pp = np.log(2. * np.pi) * (m + rank + 1.) / 2.

        pa = 0.
        spectrum_ = spectrum.copy()
        spectrum_[rank:N] = v
        for i in range(rank):
            for j in range(i + 1, len(spectrum)):
                pa += np.log((spectrum[i] - spectrum[j]) * (
                1. / spectrum_[j] - 1. / spectrum_[i])) + np.log(M)

        ll = pu + pl + pv + pp - pa / 2. - rank * np.log(M) / 2.

        return ll

    def _infer_pc_(self):
        """Infers the number of principal components.

        Notes
        -----
        Copied and modified from _infer_dimensions in
        scikit-learn.decomposition.pca (BSD-3 license).

        """
        n_ev = self.ev.size
        ll = np.empty(n_ev)
        for rank in range(n_ev):
            ll[rank] = self._assess_dimension_(rank)
        return ll.argmax()

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def transform(self, n_pc=None):
        """
        Apply the dimensionality reduction to the X dataset of shape [M, N].

        Loadings `L` with shape [``n_pc``, `N`] and scores `S`
        with shape [`M`, `n_pc`] are obtained using the following
        decomposition: :math:`X = S.L^T`.

        Parameters
        ----------
        n_pc : int, optional
            The number of principal components to compute. If not set all
            components are returned, except if n_pc is set to ``auto`` for
            an automatic determination of the number of components.

        Returns
        -------
        LT, S : |NDDataset| objects.
            n_pc loadings and their corresponding scores for each observations.


        """

        X = self._X

        # get n_pc (automatic or determined by the n_pc arguments)
        n_pc = self._get_n_pc(n_pc)

        # scores (S) and loading (L^T) matrices
        # ------------------------------------

        S = self._S[:, :n_pc]
        LT = self._LT[:n_pc]

        return LT, S


    def inverse_transform(self, n_pc=None):
        """
        Transform data back to the original space using the given number of
        PC's.

        The following matrice operation is performed: :math:`X' = S'.L'^T`
        where S'=S[:, n_pc] and L=L[:, n_pc].

        Parameters
        ----------
        n_pc : int, optional
            The number of PC to use for the reconstruction.

        Return
        ------
        X_reconstructed : |NDDataset|
            The reconstructed dataset based on n_pc principal components.

        """

        # get n_pc (automatic or determined by the n_pc arguments)
        n_pc = self._get_n_pc(n_pc)

        # reconstruct from scores and loadings using n_pc components
        S = self._S[:, :n_pc]
        LT = self._LT[:n_pc]

        X = dot(S, LT)

        # try to reconstruct something close to the original scaled,
        # standardized or centered data
        if self._scaled:
            X *= self._ampl
            X += self._min
        if self._standardized:
            X *= self._std
        if self._centered:
            X += self._center

        X.history = 'PCA reconstructed Dataset with {} principal ' \
                    'components'.format(n_pc)
        X.title = self._X.title
        return X

    def printev(self, n_pc=None):
        """prints figures of merit: eigenvalues and explained variance
        for the first n_pc PS's.

        Parameters
        ----------
        n_pc : int, optional
            The number of components to print.

        """
        # get n_pc (automatic or determined by the n_pc arguments)
        n_pc = self._get_n_pc(n_pc)

        print((self.__str__(n_pc)))

    def screeplot(self, n_pc=None, **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA.

        Parameters
        ----------
        n_pc: int
            Number of components to plot.

        """
        # get n_pc (automatic or determined by the n_pc arguments)
        n_pc = self._get_n_pc(n_pc)

        color1, color2 = kwargs.get('colors', [NBlue, NRed])
        pen = kwargs.get('pen', True)
        ylim1, ylim2 = kwargs.get('ylims', [(0,100), 'auto'])

        if ylim2 == 'auto':
            y1 = np.around(self.ev_ratio.data[0]*.95,-1)
            y2 = 101.
            ylim2 = (y1, y2)

        ax1 = self.ev_ratio[:n_pc].plot_bar(ylim = ylim1,
                                           color = color1,
                                           title='Scree plot')
        ax2 = self.ev_cum[:n_pc].plot_scatter(ylim = ylim2,
                                             color=color2,
                                             pen=True,
                                             markersize = 7.,
                                             twinx = ax1
                                            )
        ax1.set_title('Scree plot')
        return ax1, ax2


    def scoreplot(self, *pcs, colormap='viridis', color_mapping='index' ,
                  **kwargs):
        """
        2D or 3D scoreplot of samples.

        Parameters
        ----------
        *pcs: a series of int argument or a list/tuple
            Must contain 2 or 3 elements.
        colormap : str
            A matplotlib colormap.
        color_mapping : 'index' or 'labels'
            If 'index', then the colors of each n_scores is mapped sequentially
            on the colormap. If labels, the labels of the n_observation are
            used for color mapping.

        """

        if isinstance(pcs[0], (list,tuple, set)):
            pcs = pcs[0]

        # transform to internal index of component's index (1->0 etc...)
        pcs = np.array(pcs) - 1

        # colors
        if color_mapping == 'index':

            if np.any(self._S.y.data):
                colors = self._S.y.data
            else:
                colors = np.array(range(self._S.shape[0]))

        elif color_mapping == 'labels':

            labels = list(set(self._S.y.labels))
            colors = [labels.index(l) for l in self._S.y.labels]

        if len(pcs) == 2:
            # bidimentional score plot

            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
            ax.set_title('Score plot')

            ax.set_xlabel('PC# {} ({:.3f}%)'.format(
                                           pcs[0], self.ev_ratio.data[pcs[0]]))
            ax.set_ylabel('PC# {} ({:.3f}%)'.format(
                                           pcs[1], self.ev_ratio.data[pcs[1]]))
            axsc = ax.scatter( self._S.masked_data[:, pcs[0]],
                        self._S.masked_data[:, pcs[1]],
                        s=30,
                        c=colors,
                        cmap=colormap)

            number_x_labels = plotter_preferences.number_of_x_labels  # get
            # from config
            number_y_labels = plotter_preferences.number_of_y_labels
            # the next two line are to avoid multipliers in axis scale
            y_formatter = ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
            ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        if len(pcs) == 3:
            # tridimensional score plot
            fig = plt.figure(**kwargs)
            ax = plt.axes(projection='3d')
            ax.set_title('Score plot')
            ax.set_xlabel(
                    'PC# {} ({:.3f}%)'.format(pcs[0], self.ev_ratio.data[pcs[
                        0]]))
            ax.set_ylabel(
                    'PC# {} ({:.3f}%)'.format(pcs[1], self.ev_ratio.data[pcs[
                        1]]))
            ax.set_zlabel(
                    'PC# {} ({:.3f}%)'.format(pcs[2], self.ev_ratio.data[pcs[
                        2]]))
            axsc = ax.scatter(self._S.masked_data[:, pcs[0]],
                       self._S.masked_data[:, pcs[1]],
                       self._S.masked_data[:, pcs[2]],
                       zdir='z',
                       s=30,
                       c=colors,
                       cmap=colormap,
                       depthshade=True)

        if color_mapping == 'labels':
            import matplotlib.patches as mpatches

            leg= []
            for l in labels:
                i = labels.index(l)
                c = axsc.get_cmap().colors[int(255/(len(labels)-1)*i)]
                leg.append(mpatches.Patch(color=c,
                                          label=l))

            ax.legend(handles=leg, loc='best')




        return ax

# ============================================================================
if __name__ == '__main__':

    from spectrochempy.core import *

    dataset = upload_IRIS()
    pca = PCA(dataset, centered=True)
    LT, S = pca.transform(n_pc='auto')
    _ = pca.screeplot()
    _ = pca.scoreplot(1, 2, color_mapping='labels')
    show()