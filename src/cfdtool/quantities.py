# quantities.py
import numpy as np
from config import ENABLE_DIMENSION_CHECK
from cfdtool.dimensions import Dimension,dimless
from typing import Callable, Any

class Quantity:
    """
    表示带有量纲的物理量。

    Attributes:
        value (numpy.ndarray): 数值部分，可变的。
        dimension (Dimension): 量纲部分，不可变的。
    """
    __array_priority__ = 20  # 设置优先级高于 numpy.ndarray

    __slots__ = ['__value', '__dimension']

    def __init__(self, value, dimension=Dimension()):
        """
        初始化 Quantity 对象。

        参数:
            value (int, float, list, tuple, numpy.ndarray): 数值，可以是标量或数组。
            dimension (Dimension): 量纲对象，表示物理量的量纲。
            在 __init__ 和 value 的 setter 中，对于标量值（int 和 float），建议统一存储为 float 类型。这有助于避免在后续的算术运算中出现类型不一致的问题。
        """
        if isinstance(value, np.ndarray):
             self.__value = value.astype(float)  # 确保数值为 float
        elif isinstance(value, (float, int)):
            self.__value = np.array([float(value)], dtype=float)  # 统一为 float
        elif isinstance(value, (list, tuple)):
            self.__value = np.array(value, dtype=float)  # 统一为 float
        else:
            raise TypeError("value 必须是 int、float、list、tuple 或 numpy.ndarray 类型。")
        
        if not isinstance(dimension, Dimension):
            raise TypeError("dimension 必须是 Dimension 类的实例。")
        
        self.__dimension = dimension  # Dimension 已经是不可变的
    
    def __dir__(self):
        """
        重写 __dir__ 方法，隐藏内部属性，使其不出现在 dir() 的结果中。
        """
        # return [attr for attr in super().__dir__() if not attr.startswith('_Quantity__')]
        return ['value', 'dimension']
    
    @property
    def value(self)-> np.ndarray:
        """
        公开只读属性，便于访问修改数值部分。
        获取数值部分。

        返回:
            numpy.ndarray: 数值数组。
        """
        return self.__value
    
    @value.setter
    def value(self, new_value):
        """
        设置数值部分。

        参数:
            new_value (int, float, list, tuple, numpy.ndarray): 新的数值，可以是标量或数组。
        """
        if isinstance(new_value, np.ndarray):
            self.__value = new_value.astype(float)  # 确保数值为 float
        elif isinstance(new_value, (int, float)):
            self.__value = np.array([float(new_value)], dtype=float)  # 统一为 float
        elif isinstance(new_value, (list, tuple)):
            self.__value = np.array(new_value, dtype=float)  # 统一为 float
        else:
            raise TypeError("new_value 必须是 int、float、list、tuple 或 numpy.ndarray 类型。")
    
    @property
    def dimension(self)-> Dimension:
        """
        此为只读属性，便于访问，不可修改！
        获取量纲部分。

        返回:
            Dimension: 量纲对象。
        """
        return self.__dimension

    # def __array__(self, dtype=None):
    #     """
    #     重载 __array__ 方法，以便在使用 NumPy 函数时返回数值部分。
    #     """
    #     if not ENABLE_DIMENSION_CHECK:
    #         return np.asarray(self.value, dtype=dtype)
    
    def _check_dimension(self, other):
        if ENABLE_DIMENSION_CHECK and self.dimension != other.dimension:
            raise ValueError(
                f"无法与不同量纲的 Quantity 对象相加/相减：{self.dimension} 和 {other.dimension}。"
            )
        
    # 一元负号
    def __neg__(self)-> 'Quantity':
        return Quantity(-self.value, self.dimension)
    
    def __add__(self, other)-> 'Quantity':
        """
        重载加法运算符。

        参数:
            other (Quantity): 另一个 Quantity 对象。

        返回:
            Quantity: 相加后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是 Quantity 实例。
            ValueError: 如果量纲不匹配且启用了量纲检查。
        """
        if isinstance(other, Quantity):
            self._check_dimension(other)
            return Quantity(self.value + other.value, self.dimension)
        elif isinstance(other, (int, float)) and self.dimension == dimless:
            return Quantity(self.value + other, dimless)
        else:
            raise TypeError("加法仅支持 Quantity 实例之间或无量纲 Quantity 与标量之间的相加。")
    
    def __radd__(self, other)-> 'Quantity':
        """
        重载右加法运算符，以支持 + Quantity。
        """
        return self.__add__(other)

    def __sub__(self, other)-> 'Quantity':
        """
        重载减法运算符。

        参数:
            other (Quantity 或 scalar): 另一个 Quantity 对象或无量纲的标量（int, float）。

        返回:
            Quantity: 相减后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是 Quantity 实例或无量纲的标量。
            ValueError: 如果量纲不匹配且启用了量纲检查。
        """
        if isinstance(other, Quantity):
            self._check_dimension(other)
            return Quantity(self.value - other.value, self.dimension)
        elif isinstance(other, (int, float)) and self.dimension == dimless:
            return Quantity(self.value - other, dimless)
        else:
            raise TypeError("减法仅支持 Quantity 实例之间或无量纲 Quantity 与标量之间的相减。")

    def __rsub__(self, other)-> 'Quantity':
        """
        重载右减法运算符，以支持 - Quantity。

        参数:
            other (scalar): 标量值。

        返回:
            Quantity: 相减后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是标量或无量纲 Quantity 对象。
        """    
        if isinstance(other, Quantity):
            return other.__sub__(self)
        elif isinstance(other, (int, float)) and self.dimension == dimless:
            return Quantity(other - self.value, dimless)
        else:
            raise TypeError("减法仅支持 Quantity 实例之间或无量纲 Quantity 与标量之间的相减。")
    
    def __mul__(self, other)-> 'Quantity':
        """
        重载乘法运算符。

        参数:
            other (Quantity 或 scalar): 另一个 Quantity 对象或标量。

        返回:
            Quantity: 相乘后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是 Quantity 或标量。
        """
        if isinstance(other, Quantity):
            new_value = self.value * other.value
            new_dimension = self.dimension * other.dimension
            return Quantity(new_value, new_dimension)
        elif isinstance(other, (int, float, np.ndarray)):
            new_value = self.value * other
            new_dimension = self.dimension
            return Quantity(new_value, new_dimension)
        else:
            raise TypeError("乘法仅支持 Quantity 对象或标量。")
    
    def __rmul__(self, other)-> 'Quantity':
        """
        重载右乘法运算符，以支持标量 * Quantity。

        参数:
            other (scalar): 标量值。

        返回:
            Quantity: 相乘后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是标量。
        """
        if isinstance(other, (int, float, np.ndarray)):
            return self.__mul__(other)
        else:
            raise TypeError("右乘法仅支持标量、数组。")
    
    def __truediv__(self, other)-> 'Quantity':
        """
        重载除法运算符。

        参数:
            other (Quantity 或 scalar): 另一个 Quantity 对象或标量。

        返回:
            Quantity: 相除后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是 Quantity 或标量。
        """
        if isinstance(other, Quantity):
            new_value = self.value / other.value
            new_dimension = self.dimension / other.dimension
            return Quantity(new_value, new_dimension)
        elif isinstance(other, (int, float)):
            new_value = self.value / other
            new_dimension = self.dimension
            return Quantity(new_value, new_dimension)
        else:
            raise TypeError("除法仅支持 Quantity 对象或标量。")
    
    def __rtruediv__(self, other)-> 'Quantity':
        """
        重载右除法运算符，以支持标量 / Quantity。

        参数:
            other (scalar): 标量值。

        返回:
            Quantity: 相除后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是标量。
        """
        if isinstance(other, Quantity):
            new_value = other.value / self.value
            new_dimension = other.dimension / self.dimension
        elif isinstance(other, (int, float)):
            new_value = other / self.value
            new_dimension = dimless/self.dimension
        else:
            raise TypeError("右除法仅支持 Quantity 实例或标量。")
        return Quantity(new_value, new_dimension)
    
    def __iadd__(self, other)-> 'Quantity':
        """
        重载原地加法运算符（+=）。

        参数:
            other (Quantity 或 scalar): 另一个 Quantity 对象或标量。

        返回:
            self: 修改后的 Quantity 对象。
        
        异常:
            TypeError: 如果 other 不是 Quantity 实例或标量。
            ValueError: 如果量纲不匹配且启用了量纲检查。
        """
        if isinstance(other, Quantity):
            self._check_dimension(other)
            self.value += other.value
        elif isinstance(other, (int, float)) and self.dimension == dimless:
            self.value += other
        else:
            raise TypeError("原地加法仅支持 Quantity 实例之间或无量纲 Quantity 与标量之间的相加。")
        return self    
    def __isub__(self, other)-> 'Quantity':
        """
        重载原地减法运算符（-=）。

        参数:
            other (Quantity 或 scalar): 另一个 Quantity 对象或标量。

        返回:
            self: 修改后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是 Quantity 实例或标量。
            ValueError: 如果量纲不匹配且启用了量纲检查。
        """
        if isinstance(other, Quantity):
            self._check_dimension(other)
            self.value -= other.value
        elif isinstance(other, (int, float)) and self.dimension == dimless:
            self.value -= other
        else:
            raise TypeError("原地减法仅支持 Quantity 实例之间或无量纲 Quantity 与标量之间的相减。")
        return self
    
    def __imul__(self, other)-> 'Quantity':
        """
        重载原地乘法运算符（*=）。
        *=乘法运算量纲会改变，因此仅支持无量纲的 Quantity 对象或标量。
        """
        if isinstance(other, Quantity) and other.dimension == dimless:
            self.value *= other.value
        elif isinstance(other, (int, float)):
            self.value *= other
        else:
            raise TypeError("原地乘法仅支持 Quantity 对象或标量。")
        return self
    
    def __itruediv__(self, other)-> 'Quantity':
        """
        重载原地除法运算符（/=）。
        /=除法运算量纲会改变，因此仅支持无量纲的 Quantity 对象或标量。
        """
        if isinstance(other, Quantity) and other.dimension == dimless:
            self.value /= other.value
        elif isinstance(other, (int, float)):
            self.value /= other
        else:
            raise TypeError("原地除法仅支持 Quantity 对象或标量。")
        return self

    def __pow__(self, power)-> 'Quantity':
        """
        重载幂运算符。

        参数:
            power (int): 幂次。

        返回:
            Quantity: 幂运算后的 Quantity 对象。

        异常:
            TypeError: 如果 power 不是整数。
        """
        if not isinstance(power, (int, float)):
            raise TypeError("幂运算的指数必须是数值类型。")
        new_value = self.value ** power
        new_dimension = self.dimension ** power
        return Quantity(new_value, new_dimension)
    
    def __repr__(self)-> str:
        """
        返回对象的官方字符串表示，优化以避免处理大型数组时过慢。

        返回:
            str: 对象的字符串表示。
        """
        max_elements = 10  # 最大显示的元素数量
        if self.value.size > max_elements:
            # 只显示前几个元素，并指示省略
            displayed_value = f"{self.value[:max_elements]}... (total {self.value.size} elements)"
        else:
            displayed_value = self.value
        # return f"Quantity(value={displayed_value}, dimension={self.dimension})"
        return f"Quantity(value={displayed_value}, shape={self.__value.shape}, dimension={self.dimension})"

    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __eq__(self, other)-> bool:
        """
        重载等于运算符。

        参数:
            other (Quantity): 另一个 Quantity 对象。

        返回:
            bool: 如果数值和量纲都相同，返回 True，否则返回 False。
        """
        if not isinstance(other, Quantity):
            return False
        return np.array_equal(self.value, other.value) and self.dimension == other.dimension
    
    def __getitem__(self, key)-> 'Quantity':
        """
        重载切片操作符，以支持返回带量纲的 Quantity 对象。

        参数:
            key (int, slice): 切片索引或整数索引。

        返回:
            Quantity: 切片后的 Quantity 对象。
        """
        # 使用 self.value[key] 获取切片后的数值部分
        sliced_value = self.__value[key]
        # print(f"切片键: {key}, 切片后的值: {sliced_value}")
        # 不在 __init__ 中重新创建数组，直接传入现有的 ndarray
        return Quantity(sliced_value, self.dimension)
    
    def __setitem__(self, key, value):
        """
        重载赋值操作符，以支持直接修改数组部分的值。

        参数:
            key (int, slice): 要修改的索引或切片。
            value (int, float, list, numpy.ndarray): 新的值，可以是标量或数组。
        """
        # 这里检查传入值是否与当前存储的数据类型兼容
        if isinstance(value, Quantity):
            # 如果传入的是 Quantity 对象，确保量纲相同
            self._check_dimension(value)
            self.__value[key] = value.value
        elif isinstance(value, (int, float)) and self.dimension == dimless:
            self.__value[key] = float(value)
        # elif isinstance(value, (list, tuple, np.ndarray)):
        #     self.__value[key] = np.array(value, dtype=float)
        else:
            raise TypeError("赋值值必须是 Quantity 实例、标量或数组。")

    
    # def magnitude(self):
    #     """
    #     获取数值部分的拷贝。

    #     返回:
    #         numpy.ndarray: 数值数组的拷贝。
    #     """
    #     return self.value.copy()
    
    def copy(self)-> 'Quantity':
        """
        创建 Quantity 对象的副本。

        返回:
            Quantity: 副本对象。
        """
        return Quantity(self.value.copy(), self.dimension)
    
    def apply(self, func: Callable[..., Any], *args, **kwargs)-> 'Quantity':
        """
        对 Quantity 的数值部分应用一个函数，并更新数值部分。

        参数:
            func (Callable): 要应用的函数，例如 np.squeeze。
            *args: 传递给 func 的位置参数。
            **kwargs: 传递给 func 的关键字参数。
        """
        if not callable(func):
            raise TypeError("func 必须是可调用的函数。")
        
        new_value = func(self.value, *args, **kwargs)
        self.value = new_value
        return self
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs)-> 'Quantity':
        if method != '__call__':
            return NotImplemented

        # 分离 Quantity 对象和其他输入
        values = []
        dimensions = []
        for input_ in inputs:
            if isinstance(input_, Quantity):
                values.append(input_.value)
                dimensions.append(input_.dimension)
            else:
                values.append(input_)
                dimensions.append(dimless)  # 假设标量为无量纲

        # 处理不同的 ufunc
        if ufunc in (np.add, np.subtract):
            # 检查所有 Quantity 对象的量纲是否相同
            base_dim = dimensions[0]
            for dim in dimensions[1:]:
                if dim != base_dim:
                    raise ValueError(f"无法对不同量纲的 Quantity 对象进行运算：{dimensions}")
            result_value = getattr(ufunc, method)(*values, **kwargs)
            return Quantity(result_value, base_dim)
        
        elif ufunc in (np.multiply, np.divide):
            # 计算新的量纲
            if ufunc == np.multiply:
                new_dimension = dimensions[0]
                for dim in dimensions[1:]:
                    new_dimension = new_dimension * dim
            else:  # np.divide
                new_dimension = dimensions[0]
                for dim in dimensions[1:]:
                    new_dimension = new_dimension / dim
            result_value = getattr(ufunc, method)(*values, **kwargs)
            return Quantity(result_value, new_dimension)

        elif ufunc == np.power:
            # 处理幂运算，假设幂次为标量
            if not isinstance(inputs[1], (int, float)):
                raise TypeError("幂次必须是标量")
            power = inputs[1]
            new_dimension = dimensions[0] ** power
            result_value = getattr(ufunc, method)(inputs[0].value, power, **kwargs)
            return Quantity(result_value, new_dimension)
        
        else:
            # 对于其他未处理的 ufunc，返回 NotImplemented
            return NotImplemented
        
    def __array_function__(self, func, types, args, kwargs) -> 'Quantity':
        # Check if the function should be handled
        if not all(issubclass(t, Quantity) for t in types):
            return NotImplemented
        
        # 提取所有 Quantity 对象的 value
        new_args = []
        new_kwargs = {}
        for i, arg in enumerate(args):
            if isinstance(arg, Quantity):
                new_args.append(arg.value)
            else:
                new_args.append(arg)
        
        for key, value in kwargs.items():
            if isinstance(value, Quantity):
                new_kwargs[key] = value.value
            else:
                new_kwargs[key] = value
                
        # Categorize functions
        elementwise_funcs = {
            np.sin, np.cos, np.tan, np.arcsin, np.arccos, 
            np.arctan, np.arctan2, np.arcsinh, np.arccosh, np.arctanh,
            np.exp, np.log, np.sqrt, np.abs, np.sign,
            # Add more element-wise functions as needed
        }

        aggregation_funcs = {
            np.sum, np.mean, np.min, np.max, np.prod, np.std, np.var,
            # Add more aggregation functions as needed
        }

        reshaping_funcs = {
            np.squeeze, np.expand_dims, np.reshape,
            # Add more reshaping functions as needed
        }

        if func in elementwise_funcs:
            # Apply the function to the value
            result_value = func(*new_args, **new_kwargs)
            # Handle dimension changes based on function semantics
            # Example: trigonometric functions require dimensionless inputs
            trigonometric_funcs = {np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.arctan2, np.arcsinh, np.arccosh, np.arctanh}
            if func in trigonometric_funcs:
                quantity = args[0]
                if quantity.dimension.dimensions==dimless:
                    # Result is dimensionless
                    return Quantity(result_value, Quantity.dimless)
                else:
                    raise ValueError(f"Cannot apply {func.__name__} to Quantity with dimension {quantity.dimension}")
            elif func==np.sqrt:
                # Absolute value and sign functions preserve dimensions
                return Quantity(result_value, args[0].dimension**0.5)
            else:
                # For other element-wise functions, assume dimensions are preserved
                # This may not be accurate for all functions and may need customization
                return Quantity(result_value, args[0].dimension)
        
        elif func in aggregation_funcs:
            # Apply the function to the value
            result_value = func(*new_args, **new_kwargs)
            # Aggregation typically reduces dimensions, but depends on context
            # Here, assume dimensions are preserved
            return Quantity(result_value, args[0].dimension)
        
        elif func in reshaping_funcs:
            # Apply the function to the value
            result_value = func(*new_args, **new_kwargs)
            # Reshaping doesn't change dimensions
            return Quantity(result_value, args[0].dimension)
        
        else:
            # For other functions, attempt a generic application
            try:
                result_value = func(*new_args, **new_kwargs)
                if isinstance(result_value, np.ndarray) or isinstance(result_value, (int, float)):
                    # Assume dimensions are preserved
                    return Quantity(result_value, args[0].dimension)
                else:
                    return NotImplemented
            except:
                return NotImplemented