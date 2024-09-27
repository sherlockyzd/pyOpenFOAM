# quantities.py
import numpy as np
from config import ENABLE_DIMENSION_CHECK
from cfdtool.dimensions import Dimension

class Quantity:
    """
    表示带有量纲的物理量。

    Attributes:
        value (numpy.ndarray): 数值部分，可变的。
        dimension (Dimension): 量纲部分，不可变的。
    """
    __slots__ = ['__value', '__dimension']

    def __init__(self, value, dimension=Dimension()):
        """
        初始化 Quantity 对象。

        参数:
            value (int, float, list, tuple, numpy.ndarray): 数值，可以是标量或数组。
            dimension (Dimension): 量纲对象，表示物理量的量纲。
        """
        if isinstance(value, (int, float)):
            self.__value = np.array([value], dtype=float)
        elif isinstance(value, (list, tuple, np.ndarray)):
            self.__value = np.array(value, dtype=float)
        else:
            raise TypeError("value 必须是 int、float、list、tuple 或 numpy.ndarray 类型。")
        
        if not isinstance(dimension, Dimension):
            raise TypeError("dimension 必须是 Dimension 类的实例。")
        
        self.__dimension = dimension  # Dimension 已经是不可变的
    
    def __dir__(self):
        """
        重写 __dir__ 方法，隐藏内部属性，使其不出现在 dir() 的结果中。
        """
        return [attr for attr in super().__dir__() if not attr.startswith('_Quantity__')]
    
    @property
    def value(self):
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
        if isinstance(new_value, (int, float)):
            self.__value = np.array([new_value], dtype=float)
        elif isinstance(new_value, (list, tuple, np.ndarray)):
            self.__value = np.array(new_value, dtype=float)
        else:
            raise TypeError("new_value 必须是 int、float、list、tuple 或 numpy.ndarray 类型。")
    
    @property
    def dimension(self):
        """
        此为只读属性，便于访问，不可修改！
        获取量纲部分。

        返回:
            Dimension: 量纲对象。
        """
        return self.__dimension
    
    def __add__(self, other):
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
        if not isinstance(other, Quantity):
            raise TypeError("加法仅支持 Quantity 实例之间的相加。")
        
        if ENABLE_DIMENSION_CHECK and self.dimension != other.dimension:
            raise ValueError(
                f"无法相加不同量纲的 Quantity 对象：{self.dimension} 和 {other.dimension}。"
            )
        
        new_value = self.value + other.value
        new_dimension = self.dimension  # 量纲相同
        
        return Quantity(new_value, new_dimension)
    
    def __sub__(self, other):
        """
        重载减法运算符。

        参数:
            other (Quantity): 另一个 Quantity 对象。

        返回:
            Quantity: 相减后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是 Quantity 实例。
            ValueError: 如果量纲不匹配且启用了量纲检查。
        """
        if not isinstance(other, Quantity):
            raise TypeError("减法仅支持 Quantity 实例之间的相减。")
        
        if ENABLE_DIMENSION_CHECK and self.dimension != other.dimension:
            raise ValueError(
                f"无法相减不同量纲的 Quantity 对象：{self.dimension} 和 {other.dimension}。"
            )
        
        new_value = self.value - other.value
        new_dimension = self.dimension  # 量纲相同
        
        return Quantity(new_value, new_dimension)
    
    def __mul__(self, other):
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
        elif isinstance(other, (int, float)):
            new_value = self.value * other
            new_dimension = self.dimension
            return Quantity(new_value, new_dimension)
        else:
            raise TypeError("乘法仅支持 Quantity 对象或标量。")
    
    def __rmul__(self, other):
        """
        重载右乘法运算符，以支持标量 * Quantity。

        参数:
            other (scalar): 标量值。

        返回:
            Quantity: 相乘后的 Quantity 对象。

        异常:
            TypeError: 如果 other 不是标量。
        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
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
    
    def __pow__(self, power):
        """
        重载幂运算符。

        参数:
            power (int): 幂次。

        返回:
            Quantity: 幂运算后的 Quantity 对象。

        异常:
            TypeError: 如果 power 不是整数。
        """
        if not isinstance(power, int):
            raise TypeError("幂运算的指数必须是整数。")
        
        new_value = self.value ** power
        new_dimension = self.dimension ** power
        return Quantity(new_value, new_dimension)
    
    def __repr__(self):
        """
        返回对象的官方字符串表示。

        返回:
            str: 对象的字符串表示。
        """
        return f"Quantity(value={self.value}, dimension={self.dimension})"
    
    def magnitude(self):
        """
        获取数值部分的拷贝。

        返回:
            numpy.ndarray: 数值数组的拷贝。
        """
        return self.value.copy()
    
    def copy(self):
        """
        创建 Quantity 对象的副本。

        返回:
            Quantity: 副本对象。
        """
        return Quantity(self.value.copy(), self.dimension)
    
    def __eq__(self, other):
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

'''
# 示例
from quantities import Quantity
from dimensions import Dimension

# 使用量纲数组创建 Quantity 对象
velocity = Quantity([10.0, 20.0], Dimension([0, 1, -1, 0, 0, 0, 0]))  # [m/s]
pressure = Quantity([101325], Dimension([1, -1, -2, 0, 0, 0, 0]))  # [kg/(m·s²)]
density = Quantity([1.225], Dimension([1, -3, 0, 0, 0, 0, 0]))  # [kg/m³]

# 打印 Quantity 对象
print(velocity)  # 输出: Quantity(value=[10. 20.], dimension=Dimension([0, 1, -1, 0, 0, 0, 0]))
print(pressure)  # 输出: Quantity(value=[101325], dimension=Dimension([1, -1, -2, 0, 0, 0, 0]))
print(density)   # 输出: Quantity(value=[1.225], dimension=Dimension([1, -3, 0, 0, 0, 0, 0]))

# 创建 Quantity 对象
phi = Quantity([1.0, 2.0, 3.0], mass_dim / time_dim)  # [kg/s]
print(phi)  # 输出: Quantity(value=[1. 2. 3.], dimension=Dimension([1, 0, -1, 0, 0, 0, 0]))

# 加法操作（量纲匹配）
phi_new = Quantity([4.0, 5.0, 6.0], mass_dim / time_dim)  # [kg/s]
phi_sum = phi + phi_new
print(phi_sum)  # 输出: Quantity(value=[5. 7. 9.], dimension=Dimension([1, 0, -1, 0, 0, 0, 0]))

# 乘法操作（自动计算量纲）
force = phi * Quantity([10, 20, 30], velocity_dim)  # [kg/s] * [m/s] = [kg·m/s²]
print(force)  # 输出: Quantity(value=[10. 40. 90.], dimension=Dimension([1, 1, -2, 0, 0, 0, 0]))

# 错误赋值（量纲不匹配）
try:
    wrong_phi = Quantity([7.0, 8.0, 9.0], pressure_dim)  # [kg/(m·s²)]
    phi + wrong_phi  # 尝试相加不同量纲的 Quantity 对象
except ValueError as e:
    print(e)  # 输出: 无法相加不同量纲的 Quantity 对象：Dimension(M^1 L^0 T^-1) 和 Dimension(M^1 L^-1 T^-2)

# 创建二维数组的 Quantity 对象
phi_matrix = Quantity([[1.0, 2.0], [3.0, 4.0]], mass_dim / time_dim)  # [kg/s]
print(phi_matrix)
# 输出:
# Quantity(value=[[1. 2.]
#                [3. 4.]], dimension=Dimension([1, 0, -1, 0, 0, 0, 0]))

# 创建二维数组的速度 Quantity 对象
velocity_matrix = Quantity([[5.0, 6.0], [7.0, 8.0]], velocity_dim)  # [m/s]
print(velocity_matrix)
# 输出:
# Quantity(value=[[5. 6.]
#                [7. 8.]], dimension=Dimension([0, 1, -1, 0, 0, 0, 0]))

# 乘法操作（自动计算量纲）
force_matrix = phi_matrix * velocity_matrix  # [kg/s] * [m/s] = [kg·m/s²]
print(force_matrix)
# 输出:
# Quantity(value=[[10. 12.]
#                [21. 32.]], dimension=Dimension([1, 1, -2, 0, 0, 0, 0]))

# 创建三维数组的 Density Quantity 对象
density_tensor = Quantity([
    [[1.225, 1.225], [1.225, 1.225]],
    [[1.225, 1.225], [1.225, 1.225]]
], density_dim)  # [kg/m³]
print(density_tensor)
# 输出:
# Quantity(value=[[[1.225 1.225]
#                [1.225 1.225]]
# 
#               [[1.225 1.225]
#                [1.225 1.225]]], dimension=Dimension([1, -3, 0, 0, 0, 0, 0]))


'''