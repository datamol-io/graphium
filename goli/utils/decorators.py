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
