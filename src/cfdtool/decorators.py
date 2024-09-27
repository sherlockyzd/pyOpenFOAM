# decorators.py
from functools import wraps
from config import ENABLE_DIMENSION_CHECK
from quantities import Quantity

def enforce_dimensions(func):
    """
    装饰器，用于在函数执行前后强制检查量纲一致性。
    """
    def wrapper(*args, **kwargs):
        if not ENABLE_DIMENSION_CHECK:
            return func(*args, **kwargs)
        result = func(*args, **kwargs)
        return result
    return wrapper

def check_input_dimension(expected_dimension):
    """
    装饰器，用于检查输入 Quantity 的量纲是否匹配。
    
    参数:
        expected_dimension (Dimension): 期望的量纲。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, Quantity):
                    if arg.dimension != expected_dimension:
                        raise ValueError(f"参数 {arg} 的量纲不匹配。期望量纲为 {expected_dimension}，实际量纲为 {arg.dimension}")
            for key, value in kwargs.items():
                if isinstance(value, Quantity):
                    if value.dimension != expected_dimension:
                        raise ValueError(f"参数 {key} 的量纲不匹配。期望量纲为 {expected_dimension}，实际量纲为 {value.dimension}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_output_dimension(expected_dimension):
    """
    装饰器，用于检查输出 Quantity 的量纲是否匹配。
    
    参数:
        expected_dimension (Dimension): 期望的量纲。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, Quantity):
                if result.dimension != expected_dimension:
                    raise ValueError(f"输出的量纲不匹配。期望量纲为 {expected_dimension}，实际量纲为 {result.dimension}")
            return result
        return wrapper
    return decorator

def auto_calculate_dimension(operation):
    """
    装饰器，用于自动计算输出量纲，适用于乘法或除法运算。
    
    参数:
        operation (str): 'multiply' 或 'divide'，表示要执行的操作。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取输入的 Quantity 对象
            quantities = [arg for arg in args if isinstance(arg, Quantity)]
            if len(quantities) < 2:
                raise ValueError("函数需要至少两个 Quantity 对象进行计算。")
            
            # 自动计算量纲
            if operation == 'multiply':
                result_dimension = quantities[0].dimension * quantities[1].dimension
            elif operation == 'divide':
                result_dimension = quantities[0].dimension / quantities[1].dimension
            else:
                raise ValueError("未知的操作类型：仅支持 'multiply' 或 'divide'")
            
            # 执行原始函数
            result_value = func(*args, **kwargs)
            
            return Quantity(result_value, result_dimension)
        
        return wrapper
    return decorator