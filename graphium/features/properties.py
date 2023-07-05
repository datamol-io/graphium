from typing import Union, List, Callable

import numpy as np
import datamol as dm

from rdkit.Chem import rdMolDescriptors as rdMD


def get_prop_or_none(
    prop: Callable, n: int, *args: Union[dm.Mol, str], **kwargs: Union[dm.Mol, str]
) -> Union[List[float], List[None]]:
    r"""
    return properties. If error, return list of `None` with lenght `n`.
    Parameters:
        prop: The property to compute.
        n: The number of elements in the property.
        *args: The arguments to pass to the property.
        **kwargs: The keyword arguments to pass to the property.
    Returns:
        The property or a list of `None` with lenght `n`.
    """
    try:
        return prop(*args, **kwargs)
    except RuntimeError:
        return [None] * n


def get_props_from_mol(
    mol: Union[dm.Mol, str],
    properties: Union[List[str], str] = "autocorr3d",
) -> np.ndarray:
    r"""
    Function to get a given set of desired properties from a molecule,
    and output a property list.

    Parameters:
        mol: The molecule from which to compute the properties.
        properties:
            The list of properties to compute for each molecule. It can be the following:

            - 'descriptors'
            - 'autocorr3d'
            - 'rdf'
            - 'morse'
            - 'whim'
            - 'all'

    Returns:
        props: np.array(float)
            The array of properties for the desired molecule
        classes_start_idx: list(int)
            The list of index specifying the start of each new class of
            descriptor or property. For example, if props has 20 elements,
            the first 5 are rotatable bonds, the next 8 are morse, and
            the rest are whim, then ``classes_start_idx = [0, 5, 13]``.
            This will mainly be useful to normalize the features of
            each class.
        classes_names: list(str)
            The name of the classes associated to each starting index.
            Will be usefull to understand what property is the network learning.

    """

    if isinstance(mol, str):
        mol = dm.to_mol(mol)

    if isinstance(properties, str):
        properties = [properties]

    properties = [p.lower() for p in properties]

    # Initialize arrays
    props = []  # Property vector for the features
    classes_start_idx = []  # The starting index for each property class
    classes_names = []

    # Generate a 3D structure for the molecule
    mol = dm.add_hs(mol)

    if ("autocorr3d" in properties) or ("all" in properties):
        # Some kind of 3D description of the molecule
        classes_names.append("autocorr3d")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcAUTOCORR3D, 80, mol))

    if ("rdf" in properties) or ("all" in properties):
        # The radial distribution function (better than the inertia)
        # https://en.wikipedia.org/wiki/Radial_distribution_function
        classes_names.append("rdf")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcRDF, 210, mol))

    if ("morse" in properties) or ("all" in properties):
        # Molecule Representation of Structures based on Electron diffraction descriptors
        classes_names.append("morse")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcMORSE, 224, mol))

    if ("whim" in properties) or ("all" in properties):
        # WHIM descriptors are 3D structural descriptors obtained from the
        # (x,y,z)‐atomic coordinates of a molecular conformation of a chemical,
        # and are used successfully in QSAR modelling.
        classes_names.append("whim")
        classes_start_idx.append(len(props))
        props.extend(get_prop_or_none(rdMD.CalcWHIM, 114, mol))

    return np.array(props), classes_start_idx, classes_names
