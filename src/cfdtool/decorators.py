# decorators.py
from functools import wraps
from cfdtool.quantities import Quantity

def enforce_dimensions(func):
    """
    装饰器，保留接口兼容性（当前无操作）。
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def check_input_dimension(expected_dimension):
    """
    装饰器，保留接口兼容性（当前无操作）。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_output_dimension(expected_dimension):
    """
    装饰器，保留接口兼容性（当前无操作）。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

def auto_calculate_dimension(operation):
    """
    装饰器，保留接口兼容性（当前无操作）。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
