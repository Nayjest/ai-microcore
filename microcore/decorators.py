def short_descr(description):
    def decorator(func):
        func.short_descr = description
        return func
    return decorator


def descr(description):
    def decorator(func):
        func.descr = description
        return func
    return decorator
