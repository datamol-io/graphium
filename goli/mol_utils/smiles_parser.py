import numpy as np
from rdkit import rdBase, Chem
from rdkit.Chem.Descriptors import ExactMolWt


def is_iterable(obj):
    r"""
    Return whether an object is iterable or not (list, numpy array, tuple, dict)

    Parameters:

        obj: any object
            The object to check if it is an iterable

    Returns:
        is_iterable: bool
            Whether obj is an iterable (list, numpy array, tuple, dict)

    """

    try:
        iter(obj)
        return True
    except:
        return False


class SuppressRdkit(object):
    r"""
    Class meant to suppress the rdkit errors using a
    `with` statement such that the error is only suppressed
    for the execution of specific lines of code.

    Parameters:

        mute_errors: bool, optional
            Whether to mute the rdkit messages or not

        error_str: str, optional
            The type of rdkit errors to mute

            - 'rdApp.*': Mute everything
            - 'rdApp.error': Mute errors only
            - any other rdApp type

            (Default='rdApp.*')

    Example:
        with SuppressRdkit(mute_errors=mute_errors) :
            mol = Chem.MolFromSmiles(smiles_str, sanitize=sanitize)
    """

    def __init__(self, mute_errors=True, error_str="rdApp.*"):
        self.mute_errors = mute_errors
        self.error_str = error_str

    def __enter__(self):
        if self.mute_errors:
            rdBase.DisableLog(self.error_str)

    def __exit__(self, type, value, traceback):
        if self.mute_errors:
            rdBase.EnableLog(self.error_str)


def is_smiles(smiles_str, sanitize=False, check_mass=True, mute_errors=True):
    r"""
    Verify whether the provided `smiles_str` is a valid molecular SMILES.
    If ``sanitize`` is ``False``, this function verifies only if the SMILES syntax is valid.
    It checks for unautaurized characters and parentheses closure in the string.
    If ``sanitize`` is ``True``, it checks also for the molecular structure and the
    molecular bonds.

    Attributes:
        smiles_str: str
            The SMILES to check for validity
        sanitize: bool, Optional
            If ``True``, the molecule will be generated to validate the bonds and structure.
            If ``False``, only the SMILES syntax is taken into account.
            Use ``False`` for faster computation, and ``True`` for more precise molecular
            validation.
            (Default = False)
        check_mass: bool, Optional
            If ``True``, it will verify that the mass of the molecule is greater
            than 0. It is only used if ``sanitize`` is set to ``True``, otherwise
            it is ignored.
            (Default = True)
        mute_errors: bool, Optional
            Whether to mute the errors from rdkit when a string does not parse
            (Default=True)
    Returns:
        is_valid_smiles: bool
            Whether the SMILES has a valid syntax and characters.

    See Also:
        ivbase.utils.commons.is_smiles_iterable:
        ivbase.utils.commons.any_smiles_iterable:
        ivbase.utils.commons.is_fingerprint:

    """

    is_valid_smiles = False

    if isinstance(smiles_str, str):
        mol = None

        with SuppressRdkit(mute_errors=mute_errors):
            try:
                mol = Chem.MolFromSmiles(smiles_str, sanitize=sanitize)
                if sanitize:
                    mass = ExactMolWt(mol)
                else:
                    mass = 1
            except:
                mass = 0
                pass

        if (mol is not None) and (mass > 0):
            is_valid_smiles = True

    return is_valid_smiles


def any_smiles_iterable(smiles, sanitize=False, check_mass=True, min_count=1, mute_errors=True):
    r"""
    Verify at least ``min_count`` of the provided ``smiles`` is a valid molecular SMILES.
    If ``sanitize`` is ``False``, this function verifies only if the SMILES syntax is valid.
    It checks for unautaurized characters and parentheses closure in the string.
    If ``sanitize`` is ``True``, it checks also for the molecular structure and the
    molecular bonds.

    Attributes:
        smiles: str, iterator(str) or numpy.ndarray(str)
            The iterator of SMILES to check for validity.
            If a string is passed, the smiles is put in a list.
        sanitize: bool, Optional
            If ``True``, the molecule will be generated to validate the bonds and structure.
            If ``False``, only the SMILES syntax is taken into account.
            Use ``False`` for faster computation, and ``True`` for more precise molecular
            validation.
            (Default = False)
        check_mass: bool, Optional
            If ``True``, it will verify that the mass of the molecule is greater
            than 0. It is only used if ``sanitize`` is set to ``True``, otherwise
            it is ignored.
            (Default = True)
        min_count: int, Optional
            The number of counts of smiles for the function output ``True``.
            (Default=1)
        mute_errors: bool, Optional
            Whether to mute the errors from rdkit when a string does not parse
            (Default=True)

    Returns:
        is_any_smiles: bool
            Whether there is at least 1 SMILES in the array

    See Also:
        ivbase.utils.commons.is_smiles:
        ivbase.utils.commons.is_smiles_iterable:
        ivbase.utils.commons.is_fingerprint:
        ivbase.utils.commons.all_smiles_iterable:

    """

    # Convert to an array
    smiles = np.asarray(smiles)
    if smiles.ndim == 0:
        smiles = np.asarray([smiles])

    # Return `True` if a number of elements bigger than min_count are smiles
    with SuppressRdkit(mute_errors=mute_errors):
        counter = 0
        for index in np.ndindex(smiles.shape):
            this_is_smiles = is_smiles(
                smiles[index],
                sanitize=sanitize,
                check_mass=check_mass,
                mute_errors=False,
            )
            counter += this_is_smiles
            if counter >= min_count:
                return True

    return False


def all_smiles_iterable(smiles, sanitize=False, check_mass=True, mute_errors=True):
    r"""
    Verify all of the provided ``smiles`` is a valid molecular SMILES.
    If ``sanitize`` is ``False``, this function verifies only if the SMILES syntax is valid.
    It checks for unautaurized characters and parentheses closure in the string.
    If ``sanitize`` is ``True``, it checks also for the molecular structure and the
    molecular bonds.

    Attributes:
        smiles: str, iterator(str) or numpy.ndarray(str)
            The iterator of SMILES to check for validity.
            If a string is passed, the smiles is put in a list.
        sanitize: bool, Optional
            If ``True``, the molecule will be generated to validate the bonds and structure.
            If ``False``, only the SMILES syntax is taken into account.
            Use ``False`` for faster computation, and ``True`` for more precise molecular
            validation.
            (Default = False)
        check_mass: bool, Optional
            If ``True``, it will verify that the mass of the molecule is greater
            than 0. It is only used if ``sanitize`` is set to ``True``, otherwise
            it is ignored.
            (Default = True)
        mute_errors: bool, Optional
            Whether to mute the errors from rdkit when a string does not parse
            (Default=True)

    Returns:
        is_all_smiles: bool
            Whether all the elements are valid smiles. The function returns as soon as a
            single invalid smiles is found.

    See Also:
        ivbase.utils.commons.is_smiles:
        ivbase.utils.commons.any_smiles_iterable:
        ivbase.utils.commons.is_smiles_iterable:
        ivbase.utils.commons.is_fingerprint:


    """

    # Convert to an array
    smiles = np.asarray(smiles)
    if smiles.ndim == 0:
        smiles = np.asarray([smiles])

    # Return `True` if a number of elements bigger than min_count are smiles
    with SuppressRdkit(mute_errors=mute_errors):
        for index in np.ndindex(smiles.shape):
            this_is_smiles = is_smiles(
                smiles[index],
                sanitize=sanitize,
                check_mass=check_mass,
                mute_errors=False,
            )

            if ~this_is_smiles:
                return False

    return True


def is_smiles_iterable(smiles, sanitize=False, check_mass=True, mute_errors=True, verbose=0):
    r"""
    Verify whether each of the provided `smiles` is a valid molecular SMILES.
    If ``sanitize`` is ``False``, this function verifies only if the SMILES syntax is valid.
    It checks for unautaurized characters and parentheses closure in the string.
    If ``sanitize`` is ``True``, it checks also for the molecular structure and the
    molecular bonds.

    Attributes:
        smiles: str, iterator(str) or numpy.ndarray(str)
            The iterator of SMILES to check for validity.
            If a string is passed, the smiles is put in a list.
        sanitize: bool, Optional
            If ``True``, the molecule will be generated to validate the bonds and structure.
            If ``False``, only the SMILES syntax is taken into account.
            Use ``False`` for faster computation, and ``True`` for more precise molecular
            validation.
            (Default = False)
        check_mass: bool, Optional
            If ``True``, it will verify that the mass of the molecule is greater
            than 0. It is only used if ``sanitize`` is set to ``True``, otherwise
            it is ignored.
            (Default = True)
        mute_errors: bool, Optional
            Whether to mute the errors from rdkit when a string does not parse
            (Default=True)
        verbose: int, Optional
            - 0: Don't print anything
            - 1: Print a summary of the number of good and bad SMILES.
            - 2: Print every 100,000 molecules tested.
            - 3: Print the bad smiles

            (Default = 0)

    Returns:
        is_smiles_array: bool, iterator(bool) or numpy.ndarray(bool)
            Whether each of the SMILES has a valid syntax and characters.
            The iterator or array has the same shape as the original input.
            If a single ``str`` is passed, then the output is a boolean

    See Also:
        ivbase.utils.commons.is_smiles:
        ivbase.utils.commons.any_smiles_iterable:
        ivbase.utils.commons.is_fingerprint:
        ivbase.utils.commons.all_smiles_iterable:


    """

    with SuppressRdkit(mute_errors=mute_errors):

        # If str or not iterable, compute a single boolean
        if isinstance(smiles, str) or not is_iterable(smiles):
            is_smiles_array = is_smiles(smiles, sanitize=sanitize, check_mass=check_mass, mute_errors=False)

        # If it is an iterable, return a boolean array iterable
        else:
            smiles = np.array(smiles)
            is_smiles_array = np.zeros_like(smiles, dtype=bool)
            count = 0
            for index in np.ndindex(smiles.shape):
                count += 1
                is_smiles_bool = is_smiles(
                    smiles[index],
                    sanitize=sanitize,
                    check_mass=check_mass,
                    mute_errors=False,
                )
                is_smiles_array[index] = is_smiles_bool

                # Display the number of checked molecules
                if (verbose >= 2) and ((count % 100000) == 0):
                    print("Done checking {} SMILES".format(count))

                # Display the invalid molecules
                if (verbose >= 3) and (not is_smiles_bool):
                    print('Bad SMILES "{}" detected at index {}'.format(smiles[index], index))

        # Display the number of valid/invalid molecules
        if verbose >= 1:
            num_valid = np.sum(is_smiles_array)
            num_invalid = len(is_smiles_array) - num_valid
            print(
                "Done checking for SMILES validity. \n{} valid smiles, {} invalid smiles.".format(
                    num_valid, num_invalid
                )
            )

    return is_smiles_array


def filter_bad_smiles(
    dataframe,
    smiles_cols,
    sanitize=False,
    return_valid_smiles_idx=False,
    reset_index=True,
    verbose=2,
):
    r"""
    Verify whether each of the provided `smiles` is a valid SMILES.
    Then remove all the rows with invalid smiles.
    If ``sanitize`` is ``False``, this function verifies only if the SMILES syntax is valid.
    It checks for unautaurized characters and parentheses closure in the string.
    If ``sanitize`` is ``True``, it checks also for the molecular structure and the
    molecular bonds.

    Parameters:
        dataframe: pd.DataFrame
            The dataframe on which the rows will be filtered.
        smiles_cols: list
            The name of the column to be used to filter the bad smiles.
        sanitize: bool, Optional
            If ``True``, the molecule will be generated to validate the bonds and structure.
            If ``False``, only the SMILES syntax is taken into account.
            Use ``False`` for faster computation, and ``True`` for more precise molecular
            validation.
            (Default = False)
        return_valid_smiles_idx: bool, Optional
            Whether to return the boolean array of valid indexes.
            (Default = False)
        reset_index: bool, Optional
            Whether to reset the indexes after the smiles are filtered.
            (Default = True)
        verbose: int, Optional
            - 0: Don't print anything
            - 1: Print a summary of the number of good and bad SMILES.
            - 2: Print progress every 100,000 molecules tested.
            - 3: Print the bad smiles

            (Default = 0)

    Returns:
        df: pd.DataFrame
            The DataFrame with the rows filtered according to the
            bad detected SMILES.
        is_smiles: numpy.ndarray(bool), Optional
            The array containing boolean ``True`` value at the indexes where
            the SMILES are valid, and ``False`` elsewhere.

    See Also:
        ivbase.transformers.features.molecules.is_smiles_iterable:

    """

    df = dataframe
    smiles_cols = [smiles_cols] if isinstance(smiles_cols, (str, int, float)) else smiles_cols
    is_smiles = np.ones((df.shape[0], len(smiles_cols)), dtype=bool)
    for ii, col in enumerate(smiles_cols):
        is_smiles[:, ii] = is_smiles_iterable(df[col].values, sanitize=sanitize, verbose=verbose)

    is_smiles_flat = np.all(is_smiles, axis=1)

    if not np.all(is_smiles_flat):
        df = df.iloc[is_smiles_flat, :]

    if reset_index:
        df = df.reset_index(drop=True)

    if return_valid_smiles_idx:
        return df, is_smiles_flat

    return df
