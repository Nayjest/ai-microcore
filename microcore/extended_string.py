"""
Extended string used for API responses,
String by itself usually contains default(first) query result, but full response data may be accessed via attributes.
"""


class ExtendedString(str):
    def __new__(cls, string: str, attrs: dict = None):
        obj = str.__new__(cls, string)
        if attrs:
            for k, v in attrs.items():
                setattr(obj, k, v)
        return obj
