# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""Plugin module to extend NDDataset with the import methods method.
"""

__all__ = ['read_dsc']
__dataset_methods__ = __all__

import io
from warnings import warn
from datetime import datetime, timezone
import re

import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset, Coord
from spectrochempy.core.readers.importer import Importer, importermethod


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_dsc(*paths, **kwargs):
    """
    Read a steram dsc file exportyed as text file (extension ``.txt``) and return its
    content as dataset.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_dsc
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    timestamp: bool, optional
        returns the acquisition timestamp as Coord (Default=True).
        If set to false, returns the time relative to the acquisition time of the first data
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel', 'asc', 'setaram_dsc'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False)
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False)

    Examples
    ---------

    >>> import spectrochempy as scp
    >>> scp.read_dsc('msdata/ion_currents.asc')
    NDDataset: [float64] A (shape: (y:16975, x:10))

    Notes:
    ------

    See Also
    --------
    read : Read generic files.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    """
    kwargs['filetypes'] = ['Text Files (*.txt)']
    kwargs['protocol'] = ['setaram', 'txt']
    importer = Importer()
    return importer(*paths, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# Private methods
# ----------------------------------------------------------------------------------------------------------------------

@importermethod
def _read_dsc(*args, **kwargs):
    _, filename = args
    content = kwargs.get('content', False)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, 'r')

    lines = fid.readlines()
    fid.close()

    timestamp = kwargs.get('timestamp', True)

    # the list of channels is 2 lines after the line starting with "End Time"
    expt_name = lines[0]
    # read start Time
    start = lines[3].split()[-2:]
    start_date = datetime.strptime(start[0]+' '+start[1], '%d/%m/%Y %H:%M:%S')

    # the data start at line 11 (10 in python indexing)
    npoints = len(lines) - 10
    dates = []
    timestamps = np.zeros((npoints), dtype=float)
    temperatures = np.zeros((npoints))
    heatflow = np.zeros((npoints))

    for i in range(10, len(lines)):
        data = lines[i].replace(',','.').split()
        dates.append(start_date + timedelta(seconds=float(data[1])))
        timestamps[i-10] = dates[i-10].timestamp()
        temperatures[i - 10] = float(data[2])
        heatflow[i - 10] = float(data[3])

    dataset = NDDataset(heatflow, title = 'Heat Flow', units='mW')
    dataset.name = expt_name + ': heat flow'
    if timestamp:
        x1 = Coord(timestamps, title='Acquisition timestamp (UTC)', units='s', labels=(dates))
    else:
        x1 = Coord(timestamps - timestamps[0], title='Acquisition timestamp (UTC)', units='s', labels=(dates))
    x2 = Coord(temperatures, title='Temperature (°C)', units='degC')
    dataset.set_coordset(x = [x1, x2])

    # Set the NDDataset date
    dataset._date = datetime.now(timezone.utc)
    dataset._modified = dataset.date

    # Set origin, description and history
    dataset.history = f'{dataset.date}:imported from Setaram dsc text file {filename}'

    return dataset


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
