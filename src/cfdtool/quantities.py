# quantities.py

import numpy as np
from config import ENABLE_DIMENSION_CHECK
from cfdtool.dimensions import Dimension

class Quantity:
    """
    表示带有量纲的物理量。
    """
    __slots__ = ['_value', '_dimension']
    
    def __init__(self, value, dimension=Dimension()):
        """
        初始化物理量。
        
        参数：
        - value: 数值，可以是标量或 NumPy 数组。
        - dimension: Dimension 实例，表示物理量的量纲。
        """
        self._value = np.array(value, dtype=float)
        self._dimension = dimension
    
    @property
    def value(self):
        return self._value
    
    @property
    def dimension(self):
        return self._dimension
    
    def __add__(self, other):
        if not isinstance(other, Quantity):
            raise TypeError("Addition is only supported between Quantity instances.")
        if ENABLE_DIMENSION_CHECK and self.dimension != other.dimension:
            raise ValueError(f"Cannot add quantities with different dimensions: {self.dimension} and {other.dimension}")
        return Quantity(self.value + other.value, self.dimension)
    
    def __sub__(self, other):
        if not isinstance(other, Quantity):
            raise TypeError("Subtraction is only supported between Quantity instances.")
        if ENABLE_DIMENSION_CHECK and self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract quantities with different dimensions: {self.dimension} and {other.dimension}")
        return Quantity(self.value - other.value, self.dimension)
    
    def __mul__(self, other):
        if isinstance(other, Quantity):
            new_dim = self.dimension * other.dimension
            return Quantity(self.value * other.value, new_dim)
        elif isinstance(other, (int, float)):
            return Quantity(self.value * other, self.dimension)
        else:
            raise TypeError("Multiplication is only supported with Quantity or scalar.")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Quantity):
            new_dim = self.dimension / other.dimension
            return Quantity(self.value / other.value, new_dim)
        elif isinstance(other, (int, float)):
            return Quantity(self.value / other, self.dimension)
        else:
            raise TypeError("Division is only supported with Quantity or scalar.")
    
    def __pow__(self, power):
        new_dim = self.dimension ** power
        return Quantity(self.value ** power, new_dim)
    
    def __repr__(self):
        return f"Quantity(value={self.value}, dimension={self.dimension})"
    
    def magnitude(self):
        """
        返回数值部分，忽略量纲。
        """
        return self.value.copy()
