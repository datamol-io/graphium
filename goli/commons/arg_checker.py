""" Argument checker module """
import collections
import numpy as np

# Global variable of accepted string types
KNOWN_TYPES = {
    "none": None,
    "str": str,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "int": int,
    "float": float,
    "complex": complex,
    "bool": bool,
    "callable": callable,
}


def _parse_type(type_to_validate, accepted_types):
    # Check if the provided type is accepted
    if (type_to_validate is not None) and (not isinstance(type_to_validate, accepted_types)):
        raise TypeError(
            "type_to_validate should be None, type or str. {} provided".format(type(type_to_validate))
        )
    if isinstance(type_to_validate, str):
        type_to_validate = type_to_validate.lower()
        if type_to_validate in KNOWN_TYPES.keys():
            type_to_validate = KNOWN_TYPES[type_to_validate]
        else:
            raise TypeError(
                "type_to_validate is not a known type. Known types are :"
                " \n{}\n Provided : \n{}".format(KNOWN_TYPES.keys(), type_to_validate)
            )
    return type_to_validate


def _enforce_iter_type(arg, enforce_type):
    # Cast the arg to be either a list or a tuple
    if enforce_type is not None:
        if (enforce_type == list) and (not isinstance(arg, list)):
            arg = list(arg)
        elif (enforce_type == tuple) and (not isinstance(arg, tuple)):
            arg = tuple(arg)
        elif enforce_type not in (list, tuple):
            raise TypeError('enforce_type should be None, "list" or "tuple", but is {}'.format(enforce_type))
    return arg


def check_arg_iterator(arg, enforce_type=None, enforce_subtype=None, cast_subtype: bool = True):
    r"""
    Verify if the type is an iterator. If it is `None`, convert to an empty list/tuple. If it is
    not a list/tuple/str, try to convert to an iterator. If it is a str or cannot be converted to
    an iterator, then put the `arg` inside an iterator.
    Possibly enforce the iterator type to `list` or `tuple`, if `enfoce_type` is not None.
    Possibly enforce the subtype to any given type if `enforce_subtype` is not None,
    and decide whether to cast the subtype or to throw an error.

    Parameters:
        arg (any type):
            The input to verify/convert to an iterator (list or tuple). If None, an empty iterator
            is returned.
        enforce_type (str or type):
            The type to enforce the iterator. The valid choices are :
            `None`, `list`, `tuple`, `'none'`, `'list'`, `'tuple'`.
            If `None`, then the iterator type is not enforced.

        enforce_subtype (type, np.dtype or str representing basic type):
            Verify if all the elements inside the iterator are the desired type.
            If `None`, then the sub-type is not enforced.
            Accepted strings are ['none', 'str', 'list', 'tuple', 'dict', 'int',
            'float', 'complex', 'bool', 'callable']

        cast_subtype:
            If True, then the type specified by `enforce_subtype` is used to cast the
            elements inside the iterator. If False, then an error is thrown if the
            types do not match.

    Returns:
        output (iterator):
            An iterator based on the input of the desired type (list or tuple) and
            the desired subtypes.

    """

    # If not list or tuple, put into a list
    if arg is None:
        arg = []
    elif isinstance(arg, str):
        arg = [arg]
    elif isinstance(arg, tuple):
        if enforce_type is None:
            enforce_type = tuple
        arg = list(arg)
    elif not isinstance(arg, (tuple, list)):
        try:
            arg = list(arg)
        except Exception:
            arg = [arg]

    output = arg

    # Make sure that enforce_type and enforce_subtype are a good inputs
    enforce_type = _parse_type(enforce_type, (type, str))
    enforce_subtype = _parse_type(enforce_subtype, (type, str, np.dtype))

    # Cast all the subtypes of the list/tuple into the desired subtype
    if enforce_subtype is not None:
        if enforce_type is None:
            arg2 = output
        elif not isinstance(output, enforce_type):
            arg2 = list(output)
        else:
            arg2 = output
        try:
            for idx, a in enumerate(output):
                if not isinstance(a, enforce_subtype):
                    if cast_subtype:
                        arg2[idx] = enforce_subtype(a)
                    else:
                        raise TypeError(
                            "iter subtype is {}, desired subtype is {}, "
                            "but cast_subtype is set to False".format(type(arg2[idx]), enforce_subtype)
                        )
        except Exception as e:
            raise TypeError(
                "iterator subtype is {} and cannot be casted to {}\n{}".format(type(a), enforce_subtype, e)
            )

        output = _enforce_iter_type(arg2, enforce_type)

    output = _enforce_iter_type(output, enforce_type)

    return output


def check_list1_in_list2(list1, list2, throw_error=True):
    r"""
    Verify if the list1 (iterator) is included in list2 (iterator). If not, raise an error.

    Parameters:
        list1, list2: list, tuple or object
            A list or tuple containing the elements to verify the inclusion.
            If an object is provided other than a list or tuple,
            then it is considered as a list of a single element.
        throw_error: bool
            Whether to throw an error if list1 is not in list2

    Returns:
        list1_in_list2: bool
            A boolean representing the inclusion of list1 in list2. It is returned if
            throw_error is set to false


    """

    list1 = check_arg_iterator(list1)
    list2 = check_arg_iterator(list2)

    # If all elements of list1 are not in list2, throw an error
    list1_in_list2 = all(elem in list2 for elem in list1)
    if not list1_in_list2 and throw_error:
        raise ValueError(
            ("Elements in list1 should be contained in list2." + "\n\nlist1 = {} \n\n list2 = {}").format(
                list1, list2
            )
        )

    return list1_in_list2


def check_columns_choice(dataframe, columns_choice, extra_accepted_cols=None, enforce_type="list"):
    r"""
    Verify if the choice of column `columns_choice` is inside the dataframe or
    the extra_accepted_cols. Otherwise, errors are thrown by the sub-functions.

    Parameters:
        dataframe: (pd.DataFrame)
            The dataframe on which to verify if the column choice is valid.
            columns_choice: str, iterator(str)
            The columns chosen from the dataframe
        extra_accepted_cols: str, iterator(str)
            A list

        enforce_type: str or type
            The type to enforce the iterator. The valid choices are :
            `None`, `list`, `tuple`, `'none'`, `'list'`, `'tuple'`.
            If `None`, then the iterator type is not enforced.


    Returns:
        output: iterator
            A str iterator based on the input of the desired type (list or tuple)

    """
    extra_accepted_cols = [] if extra_accepted_cols is None else extra_accepted_cols
    valid_columns = list(dataframe.columns)
    kwargs_iterator = {
        "enforce_type": enforce_type,
        "enforce_subtype": None,
        "cast_subtype": False,
    }
    columns_choice = check_arg_iterator(columns_choice, **kwargs_iterator)
    extra_accepted_cols = check_arg_iterator(extra_accepted_cols, **kwargs_iterator)
    valid_columns = check_arg_iterator(valid_columns, **kwargs_iterator)
    valid_columns_full = valid_columns + extra_accepted_cols
    check_list1_in_list2(columns_choice, valid_columns_full)

    return columns_choice
