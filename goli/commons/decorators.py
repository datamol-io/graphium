class classproperty(property):
    r"""
    Decorator used to declare a static property, defined for the class
    without needing to instanciate an object.

    Example
    -----------

        .. code-block:: python

                @classproperty

                def my_static_property(cls):
                    return 5

    """

    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()
