# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["Project"]

import datetime
from spectrochempy.core.dataset.nddataset import NDIO

from traitlets import HasTraits, Any, Dict


class Projectable:  # (Hastraits
    """
    Makes an object projectable

    add the name or date attibutes to an object so that it can be added to a Project.

    Parameters
    ----------
        obj
            the original object

    Attributes
        name
            str
            the name of the object
        obj: object
            the original object
        date: datetime
            the date of the object

    >>> a = 'hello'
    >>> a = Projectable(a)
    """

    def __init__(self, obj):

        self.obj = obj
        if hasattr(obj, "name"):
            self.name = obj.name
        else:
            self.name = _class_str(obj)

        if not hasattr(obj, "date"):
            self.date = datetime.datetime.now()
        else:
            self.date = obj.date


class Project(HasTraits, NDIO):
    """
    A manager for spectrochempy objects datasets and scripts.

    It can handle multiple datasets, sub-projects, and scripts in a main
    project.

    Parameters
    ----------
    *args : Series of objects, optional
        Argument type will be interpreted
        This is optional, as they can be added later.
    argnames : list, optional
        If not None, this list gives the names associated to each
        objects passed as args. It MUST be the same length that the
        number of args, or an error will be raised.
    name : str, optional
        The name of the project.
    description: str, optional
        The description of the project
    **meta : dict
        Any other attributes to described the project.

    Attributes
    ----------
    name : str
        Name of the dataset
    names : list of str
        List of names of the entries of the project
    all_names  : list of str
        List of all names of the entries, including those in subprojects
    objects: dict
        Dict of the objects of the project
    all_objects: dict
        Dict of all objects, including those in subprojects
    projects:
        List of projects included
    Dict of the (sub)projects of the project
    all_projects",

    See Also
    --------
    NDDataset : The main object containing arrays.
    Script : Executables scripts container.

    """

    _objects = Dict(Any)  # indique que _objects a

    def __init__(self, *objs, objnames=None, name=None, description=None):

        super().__init__()

        self.parent = None

        if name is not None:
            self.name = name
        else:
            self.name = "project"

        self.description = description

        self.objects = {}

        for i, object in enumerate(objs):
            if objnames is not None:
                name = objnames[i]
            else:
                name = None
            self.add(object, name=name)

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __contains__(self, item):
        if isinstance(item, str):
            if item in self.allnames:
                return True
            else:
                return any([item in project for project in self.projects])

    def __dir__(self):
        return [
            "name",
            "description",
            "objects",
        ]

    def __getattr__(self, name):

        if "_validate" in name or "_changed" in name:
            # this is for traits management
            return super().__getattribute__(name)

        if name in self.names:
            # this allows to access project object by attribute
            return self[name]

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError("The key must be a string.")
        if key not in self.objects.keys():
            raise KeyError("The key does not exist.")

        return self.objects[key]

    def __iter__(self):

        for object in self.objects:
            yield object

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("The key must be a string.")

        if key in self.names:
            self[key] = value
        else:
            self.add(value, name=key)

    def __repr__(self):
        n = len(self.names)
        if n == 0:
            return "Project (empty)"
        elif n == 1:
            return "Project (1 object)"
        else:
            return f"Project ({n} objects)"

    def __str__(self):
        def str_proj_(project, level=0):
            s = ""
            if level > 0:
                indent = "   " * level + "⤷ "
            else:
                indent = ""
            if self.is_empty:
                s += f"{indent}{project.name} (empty Project)\n"
            else:
                s += f"{indent}{project.name} (Project):\n"
                for name, obj in project.objects.items():
                    if not isinstance(obj, Project):
                        indent = "   " * level
                        s += f"{indent}   ⤷ {name} ({_class_str(obj)})\n"
                    else:
                        level += 1
                        s += str_proj_(obj, level=level)
            return s

        return str_proj_(self)

    # -------------------------------------------------------
    # public methods and attributes
    # -------------------------------------------------------

    def add(self, *objs, **kwargs):
        """
        Add a object(s) to the current project.

        Parameters
        ----------
        objs : |NDDataset|, |Project|, ...
            The objects to add

        Other parameters
        ----------------
        names: list of str, optional
            If provided the names will be used to name the entries in the project.

        name : str, optional
            If provided the names will be used to name the entries in the project.

        Notes
        -----
            If no name is provided, the name of the entry :
            - the object.name attribute if it exists
            - the name of the input variable

        Examples
        --------

        >>> ds1 = scp.NDDataset([1, 2, 3])
        >>> proj = scp.Project()
        >>> proj.add(ds1, name='Toto')
        """
        """
        parameters
        ----------
            objs: objects to add

        other parameters
        ----------------
            names: list of str, optional
                names of objects
            name: str, optional
                name of object
        """
        self._add_objects(*objs, **kwargs)

    def implements(self, name=None):
        """
        Utility to check if the current object implement `Project`.

        Rather than isinstance(obj, Project) use object.implements('Project').
        This is useful to check type without importing the module.
        """
        if name is None:
            return "Project"
        else:
            return name == "Project"

    def remove(self, name):
        """
        Remove object from the  project.

        Parameters
        ----------
        name: str
            The name of the object to remove

        Examples
        --------
        """
        self.objects.pop(name)

    @property
    def is_empty(self):
        if self.names == []:
            return True
        else:
            return False

    @property
    def names(self):
        return [key for key in self.objects.keys()]

    @property
    def projects(self):
        return {
            name: object
            for (name, object) in self.objects.items()
            if isinstance(object, Project)
        }

    # -------------------------------------------------------
    # private methods
    # -------------------------------------------------------

    def _add_objects(self, *objs, **kwargs):
        """
        Add a series of objects
        """
        names = kwargs.get("names", None)
        name = kwargs.get("name", None)

        if len(objs) > 1:
            for i, obj in enumerate(objs):
                if names is not None:
                    self._add_object(obj, names[i])
                else:
                    self._add_object(obj)

        else:  # a single object
            if names is not None:
                if len(names) == 1:
                    self._add_object(objs[0], names[0])
                else:
                    raise ValueError(
                        "the len of names must be the saem as the number of objects"
                    )
            if name is not None:
                self._add_object(objs[0], name)
            else:
                self._add_object(objs[0])

    def _add_object(self, obj, name=None):

        if not _is_projectable(obj):
            obj = Projectable(obj)

        if name is None:
            name = obj.name

        n = 1
        while name in self.names:
            # this name already exists
            name = f"{obj.name}-({n})"

        # add in objects dictionary
        if isinstance(obj, Projectable):
            self.objects[name] = obj.obj
        else:
            self.objects[name] = obj


# Utility functions


def _class_str(object):
    return str(type(object)).split("'")[-2].split(".")[-1]


def _is_projectable(obj):
    if hasattr(obj, "name"):
        return True
    else:
        return False
