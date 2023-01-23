from __future__ import annotations
from typing import Union, Any, Callable, Dict, Hashable
from torch import Tensor


class DictTensor(dict):
    """
    A class that combines the functionality of `dict` and `torch.Tensor`.
    Specifically, it is a dict of Tensor, but it has all the methods and attributes of a Tensor.
    When a given method or attribute is called, it will be called on each element
    of the dict, and a new dict is returned.

    All methods from `torch.Tensor` and other functions are expected to work on `DictTensor`
    with the following rules:
        - The function receives one `DictTensor`: The function will be applied on
          all values of the dictionary. Examples are `torch.sum` or `torch.abs`.
        - The function receives two `DictTensor`: The function will be applied on
          each pair of Tensor with matching keys. If the keys don't match, an error
          will be thrown. Example are operators such as `dict_tensor1 + dict_tensor2`
          or `dict_tensor1 == dict_tensor2`.

    The output of the functions that act on the `torch.Tensor` level is a `dict[Any]`.
    If the output is a `dict[torch.Tensor]`, it is converted automatically to `DictTensor`.

    If a given method is available in both `dict` and `torch.Tensor`, then the one
    from `Tensor` is not available. Exceptions to the above rules are the `__dict__`
    and all comparison methods:

    - `__dict__`: Not available for this class.
    - `__lt__` or `<`: Supports two-way comparison between `DictTensor` and number, `torch.Tensor` or `DictTensor`.
    - `__le__` or `<=`: Supports two-way comparison between `DictTensor` and number, `torch.Tensor` or `DictTensor`.
    - `__eq__` or `==`: Supports two-way comparison between `DictTensor` and number, `torch.Tensor` or `DictTensor`.
    - `__ne__` or `!=`: Supports two-way comparison between `DictTensor` and number, `torch.Tensor` or `DictTensor`.
    - `__gt__` or `>`: Supports two-way comparison between `DictTensor` and number, `torch.Tensor` or `DictTensor`.
    - `__ge__` or `>=`: Supports two-way comparison between `DictTensor` and number, `torch.Tensor` or `DictTensor`.

    The only major function from `torch.Tensor` that doesn't work (to my knowledge) is the indexing.
    Indexing has to be done by manually looping the dictionary.

    Example:
        ```
        # Summing a float to a DictTensor
        dict_ten = DictTensor({
                "a": torch.zeros(5),
                "b": torch.zeros(2, 3),
                "c": torch.zeros(1, 2),
            })
        dict_ten + 2
        >>
        {'a': tensor([2., 2., 2., 2., 2.]),
        'b': tensor([[2., 2., 2.],
                [2., 2., 2.]]),
        'c': tensor([[2., 2.]])}

        # Summing a DictTensor to a DictTensor
        dict_ten = DictTensor({
                "a": torch.zeros(5),
                "b": torch.zeros(2, 3) + 0.1,
                "c": torch.zeros(1, 2) + 0.5,
            })
        dict_ten2 = DictTensor({
                "a": torch.zeros(5) + 0.2,
                "b": torch.zeros(2, 3) + 0.3,
                "c": torch.zeros(1, 2) + 0.4,
            })
        dict_ten + dict_ten2
        >>
        {'a': tensor([0.2000, 0.2000, 0.2000, 0.2000, 0.2000]),
        'b': tensor([[0.4000, 0.4000, 0.4000],
                [0.4000, 0.4000, 0.4000]]),
        'c': tensor([[0.9000, 0.9000]])}


        # Summing a accross the first axis
        dict_ten = DictTensor({
                "a": torch.zeros(5) + 0.1,
                "b": torch.zeros(2, 3) + 0.2,
                "c": torch.zeros(1, 2) + 0.5,
            })
        dict_ten.sum(axis=0)
        >>
        {'a': tensor(0.5000),
        'b': tensor([0.4000, 0.4000, 0.4000]),
        'c': tensor([0.5000, 0.5000])}

        # Getting the shape of each tensor
        dict_ten = DictTensor({
                "a": torch.zeros(5) + 0.1,
                "b": torch.zeros(2, 3) + 0.2,
                "c": torch.zeros(1, 2) + 0.5,
            })
        dict_ten.shape
        >>
        {'a': torch.Size([5]), 'b': torch.Size([2, 3]), 'c': torch.Size([1, 2])}
        ```
    """

    not_accepted_methods = {"__dict__"}
    from_tensor_methods = {
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
    }

    def _dict_func_wrapper(self, func: Callable):
        """
        A function that wraps another function to be called on each value of the `self` dictionary,
        then returns a new dictionary.
        If the first arg is also a `DictTensor`, then also loop it's values. This will allow to sum
        or multiply different dict tensors.
        If the output is a dict of tensors, convert it to `DictTensor`.

        Parameters:
            func: The function to be wrapped.
        """

        def wrap(first_input: Union[DictTensor, Any], *args, **kwargs):
            # The following lines are for when a function is 'bounded' and
            # is called form `torch.FUNCTION()` instead of `Tensor.FUNCTION()`.
            this_func, this_args, this_kwargs = func, args, kwargs
            if hasattr(func, "__self__"):
                if len(args) == 3:
                    this_func, this_class, this_args = args
                elif len(args) == 4:
                    this_func, this_class, this_args, this_kwargs = args
                first_input, this_args = this_args[0], this_args[1:]

            arg0 = args[0] if len(args) > 0 else None
            if isinstance(first_input, DictTensor):
                # Loop all the elements in the dict to apply the function
                out = {}
                for k, v in first_input.items():
                    if isinstance(arg0, DictTensor):
                        # If the first arg is also a `DictTensor`, loop it as well to apply
                        # the function to each pair of values from the `DictTensor`
                        assert set(first_input.keys()) == set(
                            arg0.keys()
                        ), f"Keys do not match. \nkeys1={first_input.keys()}\nkeys2={arg0.keys()}"
                        out[k] = this_func(v, arg0[k], *this_args[1:], **this_kwargs)
                    else:
                        # In the regular case, simply apply the function to each value of the `DictTensor`
                        out[k] = this_func(v, *this_args, **this_kwargs)
            elif isinstance(arg0, DictTensor):
                out = {}
                for k, v in arg0.items():
                    if isinstance(arg0, DictTensor):
                        # If the first arg is also a `DictTensor`, loop it as well to apply
                        # the function to each pair of values from the `DictTensor`
                        out[k] = this_func(arg0[k], v, *this_args[1:], **this_kwargs)
            else:
                raise TypeError("type `DictTensor` not found")

            out = self._to_dict_tensor(out, raise_if_type_error=False)
            return out

        return wrap

    def _create_property(self, prop_name: str):
        """
        A function that creates a property from a name.
        When the property is called, it will be called on each value of the `self` dictionary,
        then returns a new dictionary.
        """
        setattr(
            self.__class__,
            prop_name,
            property(fget=lambda self: {k: getattr(v, prop_name) for k, v in self.items()}),
        )

    def __init__(self, dic: Dict[Hashable, Tensor]):
        """
        Take a dictionary of `torch.Tensors`, and transform it into a `DictTensor`.
        Register all the required methods from `torch.Tensors`, but modify them
        to work on dictionary instead.
        """
        super().__init__(dic)

        # Assert that the dictionary is a dict of Tensor
        assert isinstance(dic, dict), f"Must be a dict, got {type(dic)}"
        assert all(
            [isinstance(val, Tensor) for val in dic.values()]
        ), f"Must only contain `torch.Tensor`, found {[type(v) for v in dic.values()]}"

        # From `torch.Tensor`, find the functions/methods/attributes to register
        tensor_func_names = {f for f in dir(Tensor)}  # if not ((f.endswith("_")) or (f.startswith("_")))}
        dict_func_names = {f for f in dir(dict)}  # if not ((f.endswith("_")) or (f.startswith("_")))}
        func_names_to_register = (
            tensor_func_names - dict_func_names - self.not_accepted_methods
        ) | self.from_tensor_methods

        # Loop all selected functions/methods/attributes to register them in the current class
        for func_name in func_names_to_register:
            func = getattr(Tensor, func_name)
            if isinstance(func, Callable):
                # Register the methods and functions
                setattr(DictTensor, func_name, self._dict_func_wrapper(func))
            else:
                # Register the attributes as properties
                self._create_property(func_name)

    @staticmethod
    def _to_dict_tensor(dic: Dict[Hashable, Tensor], raise_if_type_error: bool = True) -> DictTensor:
        """
        Convert a dictionary of tensors into a `DictTensor`
        """
        if all([isinstance(v, Tensor) for v in dic.values()]):
            dic = DictTensor(dic)
        else:
            if raise_if_type_error:
                raise TypeError("All values must be Tensors")
        return dic

    def apply(self, func: Callable, *args, **kwargs) -> Union[DictTensor, Dict[Any]]:
        """
        Apply a function on every Tensor "value" of the current dictionary

        Parameters:
            func: function to be called on each Tensor
            *args, **kwargs: Additional parameters to the function. Note that
                the Tensor should be the first input to the function.

        Returns:
            `DictTensor` if the output of the function is a `torch.Tensor`,
            or `dict[X]` if the output of the function is of type`X`.
        """
        out = {k: func(v, *args, **kwargs) for k, v in self.items()}
        out = self._to_dict_tensor(out, raise_if_type_error=False)
        return out
