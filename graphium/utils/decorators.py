"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals.
Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


class classproperty(property):
    r"""
    Decorator used to declare a class property, defined for the class
    without needing to instanciate an object.

    !!! Example

        ``` python linenums="1"
            @classproperty
            def my_class_property(cls):
                return 5
        ```
    """

    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()
