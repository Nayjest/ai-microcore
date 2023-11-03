"""
Extended string used for API responses,
String by itself usually contains default(first) query result,
but full response data may be accessed via attributes.
"""
import inspect
import builtins


class ExtendedString(str):
    def __new__(cls, string: str, attrs: dict = None):
        """
        Allows string to have attributes.
        """
        obj = str.__new__(cls, string)
        if attrs:
            for k, v in attrs.items():
                setattr(obj, k, v)
        return obj

    def __getattr__(self, item):
        """
        Provides chaining of global functions
        """
        global_func = inspect.currentframe().f_back.f_globals.get(item) or vars(
            builtins
        ).get(item, None)
        if callable(global_func):

            def method_handler(*args, **kwargs):
                res = global_func(self, *args, **kwargs)
                if isinstance(res, str) and not isinstance(res, ExtendedString):
                    res = ExtendedString(res)
                return res

            return method_handler

        # If there's not a global function with that name, raise an AttributeError as usual
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )
