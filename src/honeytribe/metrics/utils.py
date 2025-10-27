def _name_function(name: str):
    def decorator(func):
        func.__name__ = name
        return func
    return decorator
