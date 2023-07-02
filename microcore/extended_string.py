class ExtendedString(str):
    def __new__(cls, string: str, attrs: dict = None):
        obj = str.__new__(cls, string)
        if attrs:
            for k, v in attrs.items():
                setattr(obj, k, v)
        return obj
