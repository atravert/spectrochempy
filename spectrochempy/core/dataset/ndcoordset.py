# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implements three classes |Coord|, |CoordSet| and |CoordRange|.
"""

__all__ = ['CoordSet']

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------
import copy
import warnings
import textwrap
import uuid

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from traitlets import HasTraits, List, Bool, Unicode, observe, All, validate, default, Instance, TraitError

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------
from .ndarray import NDArray, DEFAULT_DIM_NAME, HAS_PANDAS, HAS_XARRAY
from .ndcoord import Coord
from ...core import log
from ...utils import is_sequence, colored_output, convert_to_html


# ======================================================================================================================
# CoordSet
# ======================================================================================================================
class CoordSet(HasTraits):
    """
    A collection of Coord objects for a NDArray object with a validation
    method.
    """
    # Hidden attributes containing the collection of objects
    _id = Unicode()
    _coords = List(allow_none=True)
    
    _updated = Bool(False)
    # Hidden id and name of the object
    _id = Unicode()
    _name = Unicode()

    # Hidden attribute to specify if the collection is for a single dimension
    _is_same_dim = Bool(False)
    
    # other settings
    _copy = Bool(False)
    _sorted = Bool(True)
    _html_output = Bool(False)
    
    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def __init__(self, *coords, **kwargs):
        """
        Parameters
        ----------
        coords : |NDarray|, |NDArray| subclass or |CoordSet| sequence of objects.
            If an instance of CoordSet is found, instead of an array, this means
            that all coordinates in this coords describe the same axis.
            It is assumed that the coordinates are passed in the order of the
            dimensions of a nD numpy array (
            `row-major <https://docs.scipy.org/doc/numpy-1.14.1/glossary.html#term-row-major>`_
            order), i.e., for a 3d object: 'z', 'y', 'x'.
        x : |NDarray|, |NDArray| subclass or |CoordSet|
            A single coordinate associated to the 'x'-dimension.
            If a coord was already passed in the argument, this will overwrite
            the previous. It is thus not recommended to simultaneously use
            both way to initialize the coordinates to avoid such conflicts.
        y, z, u, ...: |NDarray|, |NDArray| subclass or |CoordSet|
            Same as `x` for the others dimensions.
        is_same_dim : bool, optional, default:False
            if true, all elements of coords describes a single dimension.
            By default, this is false, which means that each item describes
            a different dimension.

        """

        self._copy = kwargs.pop('copy', True)
        self._sorted = kwargs.pop('sorted', True)
        keepnames = kwargs.pop('keepnames', False)

        
        # initialise the coordinate list
        self._coords = []
        

        # First evaluate passed args
        # --------------------------
        
        # some cleaning
        if coords:

            if all([(isinstance(coords[i], (np.ndarray, NDArray, list, CoordSet))
                     or coords[i] is None) for i in range(len(coords))]):
                # Any instance of a NDArray can be accepted as coordinates for a dimension.
                # If an instance of CoordSet is found, this means that all
                # coordinates in this set describe the same axis
                coords = tuple(coords)

            elif is_sequence(coords) and len(coords) == 1:
                # if isinstance(coords[0], list):
                #     coords = (CoordSet(*coords[0], sorted=False),)
                # else:
                coords = coords[0]
                    
                if isinstance(coords, dict):
                    # we have passed a dict, postpone to the kwargs evaluation process
                    kwargs.update(coords)
                    coords = None

            else:
                raise ValueError('Did not understand the inputs')

        # now store the args coordinates in self._coords (validation is fired when this attribute is set)
        if coords:
            for coord in coords[::-1]:                      # we fill from the end of the list
                                                            # (in reverse order) because by convention when the
                                                            # names are not specified, the order of the
                                                            # coords follow the order of dims.
                if not isinstance(coord, CoordSet):
                    if isinstance(coord, list):
                        coord = CoordSet(*coord, sorted = False)
                    else:
                        coord = Coord(coord, copy=True)                    # be sure to cast to the correct type
                else:
                    coord = copy.deepcopy(coord)
                    
                if not keepnames:
                    coord.name = self.available_names.pop() # take the last available name of
                                                            # available names list
                                                            
                self._append(coord)                         # append the coord (but instead of append,
                                                            # use assignation -in _append - to fire the
                                                            # validation process )

        # now evaluate keywords argument
        # ------------------------------
        
        for key, coord in kwargs.items():
            # prepare values to be either Coord or CoordSet
            if isinstance(coord, (list, tuple)):
                coord = CoordSet(coord, sorted=False)           # make sure in this case it becomes a CoordSet instance
                
            elif isinstance(coord, np.ndarray) or coord is None:
                coord = Coord(coord, copy=True)                 # make sure it's a Coord
                                                                # (even if it is None -> Coord(None)

            # populate the coords with coord and coord's name.
            if isinstance(coord, (NDArray, Coord, CoordSet)):
                if key in self.available_names or (len(key)==2 and key.startswith('_') and key[1] in list("123456789")):
                    # ok we can find it as a canonical name:
                    # this will overwrite any already defined coord value
                    # which means also that kwargs have priority over args
                    coord.name = key
                    self._append(coord)
                    
                elif not self.is_empty and key in self.names:
                    # append when a coordinate with this name is already set in passed arg.
                    # replace it
                    idx = self.names.index(key)
                    coord.ame = key
                    self._coords[idx] = coord
                
                else:
                    raise KeyError(f'Probably an invalid key (`{key}`) for coordinates has been passed. '
                                   f'Valid keys are among :{DEFAULT_DIM_NAME}')
                
            else:
                raise ValueError(f'Probably an invalid type of coordinates has been passed: {key}:{coord} ')

        # store the item (validation will be performed)
        #self._coords = _coords

        # inform the parent about the update
        self._updated = True

        # set a notifier on the name traits name of each coordinates
        for coord in self._coords:
            if coord is not None:
                HasTraits.observe(coord, self._coords_update, '_name')

        # initialize the base class with the eventual remaining arguments
        super().__init__(**kwargs)

    # ..................................................................................................................
    def implements(self, name=None):
        # Rather than isinstance(obj,CoordSet) use object.implements('CoordSet')
        # This is useful to check type without importing the module
        if name is None:
            return ['CoordSet']
        else:
            return name == 'CoordSet'

    # ------------------------------------------------------------------------------------------------------------------
    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @validate('_coords')
    def _coords_validate(self, proposal):
        coords = proposal['value']
        if not coords:
            return None
        
        for id, coord in enumerate(coords):
            if coord and not isinstance(coord, (Coord, CoordSet)):
                raise TypeError('At this point all passed coordinates should be of type Coord or CoordSet!')
                # coord = Coord(coord)
            coords[id] = coord

        for coord in coords:
            if isinstance(coord, CoordSet):
                # it must be a single dimension axis
                # in this case we must have same length for all coordinates
                coord._is_same_dim = True
                
                #check this is valid in term of size
                try:
                    size = coord.sizes
                except ValueError:
                    raise
                
                # change the internal names
                n = len(coord)
                coord._set_names([f"_{i+1}" for i in range(n)])         # we must have  _1 for the first coordinates,
                                                                        # _2 the second, etc...
                coord._set_parent_dim(coord.name)
                
        # last check and sorting
        names= []
        for coord in coords:
            if coord.has_defined_name:
                names.append(coord.name)
            else:
                raise ValueError('At this point all passed coordinates should have a valid name!')
         
        if coords:
            if self._sorted:
                _sortedtuples = sorted((coord.name, coord) for coord in coords) # Final sort
                coords = list(zip(*_sortedtuples))[1]
            return list(coords) # be sure its a list not a tuple
        else:
            return None

    # ..................................................................................................................
    @default('_id')
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    # ------------------------------------------------------------------------------------------------------------------
    # Readonly Properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def available_names(self):
        """
        Chars that can be used for dimension name (DEFAULT_DIM_NAMES less those already in use)
        """
        _available_names = DEFAULT_DIM_NAME.copy()
        for item in self.names:
            if item in _available_names:
                _available_names.remove(item)
        return _available_names

    # ..................................................................................................................
    @property
    def coords(self):
        return self._coords
    
    # ..................................................................................................................
    @property
    def has_defined_name(self):
        """
        bool - True is the name has been defined

        """
        return not(self.name == self.id)
    
    # ..................................................................................................................
    @property
    def id(self):
        """
        str - Object identifier (Readonly property).
        """
        return self._id

    # ..................................................................................................................
    @property
    def is_empty(self):
        """bool - True if there is no coords defined (readonly
        property).
        """
        if self._coords:
            return len(self._coords) == 0
        else:
            return False

    # ..................................................................................................................
    @property
    def is_same_dim(self):
        """bool - True if the coords define a single dimension (readonly
        property).
        """
        return self._is_same_dim

    # ..................................................................................................................
    @property
    def sizes(self):
        """int or tuple of int - Sizes of the coord object for each dimension
        (readonly property). If the set is for a single dimension return a
        single size as all coordinates must have the same.
        """
        _sizes = []
        for i, item in enumerate(self._coords):
            _sizes.append(item.size)  # recurrence if item is a CoordSet

        if self.is_same_dim:
            _sizes = list(set(_sizes))
            if len(_sizes) > 1:
                raise ValueError('Coordinates must be of the same size for a dimension with multiple coordinates')
            return _sizes[0]
        return _sizes

    # alias
    size = sizes

    # ..................................................................................................................
    # @property
    # def coords(self):  #TODO: replace with itertiems, items etc ... to simulate a dict
    #     """list - list of the Coord objects in the current coords (readonly
    #     property).
    #     """
    #     return self._coords

    # ..................................................................................................................
    @property
    def names(self):
        """list - Names of the coords in the current coords (read only property)
        """
        _names = []
        if self._coords:
            for item in self._coords:
                if item.has_defined_name:
                    _names.append(item.name)
        return _names
    
    
    # ------------------------------------------------------------------------------------------------------------------
    # Mutable Properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self._id

    @name.setter
    def name(self, value):
        self._name = value


    # ..................................................................................................................
    @property
    def titles(self):
        """list - Titles of the coords in the current coords
        """
        _titles = []
        for item in self._coords:
            if isinstance(item, NDArray):
                _titles.append(item.title if item.title else item.name)  # TODO:name
            elif isinstance(item, CoordSet):
                _titles.append(
                    [el.title if el.title else el.name for el in item])  # TODO:name
            else:
                raise ValueError('Something wrong with the titles!')
        return _titles

    # ..................................................................................................................
    @property
    def labels(self):
        """list - Labels of the coordinates in the current coordset
        """
        return [item.labels for item in self]


    # ..................................................................................................................
    @property
    def units(self):
        """
        list - Units of the coords in the current coords

        """
        return [item.units for item in self]


    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def copy(self):
        """
        Make a disconnected copy of the current coords.

        Returns
        -------
        object
            an exact copy of the current object

        """
        return self.__copy__()
    
    # ..................................................................................................................
    def keys(self):
        """
        Alias for names
        
        Returns
        -------
        out : list
            list of all coordinates names
            
        """
        if self.names:
            return self.names
        else:
            return []
        
    
    # ..................................................................................................................
    def set(self, *args, **kwargs):
        """
        Set one or more coordinates in the current CoordSet
        
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        if not args and not kwargs:
            return
        
        if len(args)==1 and (is_sequence(args[0]) or isinstance(args[0], CoordSet)):
            args = args[0]
        
        if isinstance(args, CoordSet):
            kwargs.update(args.to_dict())
            args = ()
            
        if args:
            self._coords = [] #reset
            
        for i, item in enumerate(args[::-1]):
            item.name = self.available_names.pop()
            self._append(item)
        
        for k, item in kwargs.items():
            self[k] = item
        
    # ..................................................................................................................
    def set_titles(self, *args, **kwargs):
        """
        Set one or more coord title at once
        
        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's  name alhabetical order:
        e.g, the first title will be for the `x` coordinates, the second for the `y`, etc.
        
        Parameters
        ----------
        args : str(s)
            The list of titles to apply to the set of coordinates (they must be given according to the coordinate's name
            alphabetical order
        kwargs : str
            keyword attribution of the titles. The keys must be valid names among the coordinate's name list. This
            is the recommended way to set titles as this will be less prone to errors.
            
        Exemples
        --------

        """
        if len(args)==1 and (is_sequence(args[0]) or isinstance(args[0], CoordSet)):
            args = args[0]
            
        for i, item in enumerate(args):
            if not isinstance(self[i],CoordSet):
                self[i].title = item
            else:
                if is_sequence(item):
                    for i, v in enumerate(self[i]):
                        v.title = item[i]
    
        for k, item in kwargs.items():
            self[k].title = item

    # ..................................................................................................................
    def set_units(self, *args, **kwargs):
        """
        Set one or more coord units at once.
        
        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's name alhabetical order:
        e.g, the first units will be for the `x` coordinates, the second for the `y`, etc.
        
        Parameters
        ----------
        args : str(s)
            The list of units to apply to the set of coordinates (they must be given according to the coordinate's name
            alphabetical order
        kwargs : str
            keyword attribution of the units. The keys must be valid names among the coordinate's name list. This
            is the recommended way to set units as this will be less prone to errors.
        force: bool, optional, default=False
            whether or not the new units must be compatible with the current units. See the `Coord`.`to` method.
            
        Examples
        --------
        
        
        

        """
        force = kwargs.pop('force', False)
        
        if len(args)==1 and is_sequence(args[0]):
            args = args[0]
            
        for i, item in enumerate(args):
            if not isinstance(self[i],CoordSet):
                self[i].to(item, force=force, inplace=True)
            else:
                if is_sequence(item):
                    for i, v in enumerate(self[i]):
                        v.to(item[i], force=force, inplace=True)
        
        for k, item in kwargs.items():
            self[k].to(item, force=force, inplace=True)
            
    
    # ..................................................................................................................
    def to_dict(self):
        """
        Return a dict of the coordinates from the coordset
        
        Returns
        -------
        out: dict
            A dictionary where keys are the names of the coordinates, and the values the coordinates themselves
            
        """
        return dict(zip(self.names, self._coords))
    
    # ..................................................................................................................
    def to_index(self):
        """
        Convert all index coordinates into a `pandas.Index`

        Returns
        -------
        pandas.Index
            Index subclass corresponding to the outer-product of all dimension
            coordinates. This will be a MultiIndex if this object is has more
            than more dimension.

        """
        if HAS_PANDAS:
            import pandas as pd
        else:
            raise ImportError('Cannot perform this conversion as Pandas is not installed.')
        
        if len(self) == 0:
            raise ValueError('no valid index for a 0-dimensional object')
        
        elif len(self) == 1:
            return self[0].to_pandas()
        
        else:
            return pd.MultiIndex.from_product([item.data for item in self], names=self.titles)

    def update(self, **kwargs):
        """
        Update a specific coordinates in the CoordSet.

        Parameters
        ----------
        kwarg : Only keywords among the CoordSet.names are allowed - they denotes the name of a dimension.

        """
        dims = kwargs.keys()
        for dim in list(dims)[:]:
            if dim in self.names:
                # we can replace the given coordinates
                idx = self.names.index(dim)
                self[idx] = Coord(kwargs.pop(dim), name=dim)


    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    def _append(self, coord):
        # utility function to append coordinate with full validation
        if not isinstance(coord, tuple):
            coord = (coord, )
        if self._coords:
            self._coords = (*coord, ) + tuple(self._coords)                          # instead of append, fire the validation process
        else:
            self._coords = (*coord, )
            
    def _set_names(self, names):
        # utility function to change names of coordinates (in batch)
        # useful when a coordinate is a CoordSet itself
        for coord, name in zip(self._coords, names):
            coord.name = name
      
    def _set_parent_dim(self, name):
        # utility function to set the paretn name for sub coordset
        for coord in self._coords:
            coord._parent_dim = name

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @staticmethod
    def __dir__():
        return ['coords', 'is_same_dim', 'name']

    # ..................................................................................................................
    def __call__(self, *args, **kwargs):
        # allow the following syntax: coords(), coords(0,2) or
        coords = []
        axis = kwargs.get('axis', None)
        if args:
            for idx in args:
                coords.append(self[idx])
        elif axis is not None:
            if not is_sequence(axis):
                axis = [axis]
            for i in axis:
                coords.append(self[i])
        else:
            coords = self._coords
        if len(coords) == 1:
            return coords[0]
        else:
            return CoordSet(*coords)

    # ..................................................................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash(tuple(self._coords))

    # ..................................................................................................................
    def __len__(self):
        return len(self._coords)

    def __delattr__(self, item):
        if 'notify_change' in item:
            pass
        
        else:
            try:
                return self.__delitem__(item)
            except (IndexError, KeyError):
                raise AttributeError
    
    # ..................................................................................................................
    def __getattr__(self, item):
        # when the attribute was not found
        if '_validate' in item or '_changed' in item:
            raise AttributeError

        try:
            return self.__getitem__(item)
        except (IndexError, KeyError):
            raise AttributeError

    # ..................................................................................................................
    def __getitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                return self._coords.__getitem__(idx)

            # ok we did not find it!
            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(f"Getting a coordinate from its title. However `{index}` occurs several time. Only"
                                  f" the first occurence is returned!")
                return self._coords.__getitem__(self.titles.index(index))

            # may be it is a title or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    return item.__getitem__(item.titles.index(index))

            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.names:
                    # selection by subcoord name
                    return item.__getitem__(item.names.index(index))

            try:
                # let try with the canonical dimension names
                if index[0] in self.names:
                    # ok we can find it a a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == '_':
                        if isinstance(c, CoordSet):
                            c = c.__getitem__(index[1:])
                        else:
                            c = c.__getitem__(index[2:]) # try on labels
                    return c
            except IndexError:
                pass

            raise KeyError(f"Could not find `{index}` in coordinates names or titles")

        res = self._coords.__getitem__(index)
        if isinstance(index, slice):
            if isinstance(res, CoordSet):
                res = (res, )
            return CoordSet(*res, keepnames=True)
        else:
            return res

    # ..................................................................................................................
    def __setattr__(self, key, value):
        keyb = key[1:] if key.startswith('_') else key
        if keyb  in ['parent', 'copy', 'sorted', 'coords', 'updated', 'name',
                     'trait_values', 'trait_notifiers', 'trait_validators', 'cross_validation_lock', 'notify_change']:
            super().__setattr__(key, value)
            return
        
        try:
            self.__setitem__(key, value)
        except:
            super().__setattr__(key, value)
    
    
    # ..................................................................................................................
    def __setitem__(self, index, coord):
        coord = coord.copy(keepname=True)  # to avoid modifying the original
        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                coord.name = index
                self._coords.__setitem__(idx, coord)
                return

            # ok we did not find it!
            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(f"Getting a coordinate from its title. However `{index}` occurs several time. Only"
                                  f" the first occurence is returned!")
                index = self.titles.index(index)
                coord.name = self.names[index]
                self._coords.__setitem__(index, coord)
                return

            # may be it is a title or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    index = item.titles.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.names:
                    # selection by subcoord title
                    index = item.names.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return
                
            try:
                # let try with the canonical dimension names
                if index[0] in self.names:
                    # ok we can find it a a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == '_':
                        c.__setitem__(index[1:], coord)
                    return

            except KeyError:
                pass

            # add the new coordinates
            if index in self.available_names or \
                                            (len(index)==2 and index.startswith('_') and index[1] in list("123456789")):
                coord.name = index
                self._coords.append(coord)
                return
            
            else:
                raise KeyError(f"Could not find `{index}` in coordinates names or titles")
            
        self._coords[index] = coord

    # ..................................................................................................................
    def __delitem__(self, index):
    
        if isinstance(index, str):
        
            # find by name
            if index in self.names:
                idx = self.names.index(index)
                del self._coords[idx]
                return

            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                index = self.titles.index(index)
                self._coords.__delitem__(index)
                return
        
            # may be it is a title in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    return item.__delitem__(index)

            # let try with the canonical dimension names
            if index[0] in self.names:
                # ok we can find it a a canonical name:
                c = self._coords.__getitem__(self.names.index(index[0]))
                if len(index) > 1 and index[1] == '_':
                    if isinstance(c, CoordSet):
                        return c.__delitem__(index[1:])

            raise KeyError(f"Could not find `{index}` in coordinates names or titles")

    # ..................................................................................................................
    # def __iter__(self):
    #    for item in self._coords:
    #        yield item

    # ..................................................................................................................
    def __repr__(self):
        out = "CoordSet: [" + ', '.join(['{}'] * len(self._coords)) + "]"
        s = []
        for item in self._coords:
            if isinstance(item, CoordSet):
                s.append(f"{item.name}:" + repr(item).replace('CoordSet: ', ''))
            else:
                s.append(f"{item.name}:{item.title}")
        out = out.format(*s)
        return out

    # ..................................................................................................................
    def __str__(self):
        return repr(self)

    # ..................................................................................................................
    def _cstr(self, header='  coordinates: ... \n', print_size=True):
        
        #
        # out = header
        # for i, item in enumerate(self._coords):
        #     txt =""
        #     if print_size:
        #         txt = f'{item._str_shape().rstrip()}\n'
        #     txt += f'{item._cstr(print_size=False)}\n'
        #
        #     out += format(textwrap.indent(txt, ' ' * 4))
        #     out = out.replace('        title', f'({item.name}){" " * (6 - len(item.name))}title')

        txt = ''
        for idx, dim in enumerate(self.names):
            coord = getattr(self, dim)
            
            if coord:
                txt += ' DIMENSION `{}`\n'.format(dim)
                
                if isinstance(coord, CoordSet):
                    txt += '        index: {}\n'.format(idx)
                    if not coord.is_empty:
                        if print_size:
                            txt += f'{coord[0]._str_shape().rstrip()}\n'
                        coord._html_output = self._html_output
                        #txt += 'Multiple coord.\n'
                        for idx_s, dim_s in enumerate(coord.names):
                            c = getattr(coord, dim_s)
                            txt += f'          ({dim_s}) ...\n'
                            sub = c._cstr(header='  coordinates: ... \n', print_size=False) #, indent=4, first_indent=-6)
                            txt +=  f"{sub}\n"
                
                elif not coord.is_empty:
                    # coordinates if available
                    txt += '        index: {}\n'.format(idx)
                    coord._html_output = self._html_output
                    txt += '{}\n'.format(coord._cstr(header=header, print_size=print_size))
                
                
        txt = txt.rstrip()  # remove the trailing '\n'

        if not self._html_output:
            return colored_output(txt.rstrip())
        else:
            return txt.rstrip()

    # ..................................................................................................................
    def _repr_html_(self):
        return convert_to_html(self)
    
    # ..................................................................................................................
    def __deepcopy__(self, memo):
        coords = self.__class__(tuple(copy.deepcopy(ax, memo=memo) for ax in self), keepnames=True)
        coords.name = self.name
        return coords

    # ..................................................................................................................
    def __copy__(self):
        coords = self.__class__(tuple(copy.copy(ax) for ax in self), keepnames=True)
        coords.name = self.name
        return coords

        # ..................................................................................................................

    def __eq__(self, other):
        if other is None:
            return False
        try:
            return self._coords == other._coords
        except:
            return False

    # ..................................................................................................................
    def __ne__(self, other):
        return not self.__eq__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def _coords_update(self, change):
        # when notified that a coord name have been updated
        self._updated = True

    # ..................................................................................................................
    @observe(All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }
        log.debug('changes in CoordSet: %s to %s' % (change.name, change.new))
        if change.name == '_updated' and change.new:
            self._updated = False  # reset


# ======================================================================================================================
if __name__ == '__main__':
    pass