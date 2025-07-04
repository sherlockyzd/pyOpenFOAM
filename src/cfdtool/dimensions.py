# value.py
import numpy as np

class Dimension:
    '''
    在 OpenFOAM 中，`value` 用于定义物理量的维度。每个物理量的维度是一个具有七个分量的数组，分别表示它在以下七个基础单位下的幂次[M, L, T, Θ, I, N, J]：
    1. 质量（mass）
    2. 长度（length）
    3. 时间（time）
    4. 温度（temperature）
    5. 电流（electric current）
    6. 物质的量（amount of substance）
    7. 发光强度（luminous intensity）
    这些单位遵循国际单位制（SI）。`value` 主要用于确保单位的正确性，并帮助 OpenFOAM 在计算过程中执行单位检查。
    value [M L T Θ I N J] 每个字母表示一个单位的幂次，按照如下解释：
    - `M`: 质量，单位 kg
    - `L`: 长度，单位 m
    - `T`: 时间，单位 s
    - `Θ`: 温度，单位 K
    - `I`: 电流，单位 A
    - `N`: 物质的量，单位 mol
    - `J`: 发光强度，单位 cd
    在 OpenFOAM 中，物理量的维度通常写在文件开头。
    速度的维度（单位 m/s）value [0 1 -1 0 0 0 0];
    压力的维度（单位 N/m² 或者 kg/(m·s²)）value [1 -1 -2 0 0 0 0];
    体积：`[0 3 0 0 0 0 0]` （m³）
    力：`[1 1 -2 0 0 0 0]` （N 或 kg·m/s²）
    密度：`[1 -3 0 0 0 0 0]` （kg/m³）
    通过这种方式，OpenFOAM 可以确保所有物理量的单位在计算中保持一致。如果在模拟过程中出现单位不匹配的问题，OpenFOAM 将给出错误信息。
    '''
    __slots__ = ['_value'] # 使用 __slots__ 限制实例属性，避免动态添加属性，提高性能和内存效率

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)) and len(args[0]) == 7:
            dim_list = args[0]
        elif 'dim_list' in kwargs and isinstance(kwargs['dim_list'], (list, tuple, np.ndarray)) and len(kwargs['dim_list']) == 7:
            dim_list = kwargs['dim_list']
        else:
            dim_list = [kwargs.get(key, 0) for key in ['M', 'L', 'T', 'Theta', 'I', 'N', 'J']]
        
        if len(dim_list) != 7:
            raise ValueError("dim_list must be a length-7 list, tuple, or NumPy array representing [M, L, T, Θ, I, N, J].")
        
        # 设置属性时确保只允许在初始化时赋值，后续不可修改
        value=np.array(dim_list, dtype=np.float64)
        # 将数组设置为只读，确保不可变性
        value.flags.writeable = False
        object.__setattr__(self, '_value', value)
        # self._value.flags.writeable = False
    
    @property
    def value(self):
        '''只读属性，返回维度数组'''
        return self._value
    
    def __dir__(self):
        '''控制 dir() 输出的属性，隐藏 _value，只暴露 value'''
        return ['value']
    # def __dir__(self):
    #     '''控制 dir() 输出的属性，隐藏 _value，只暴露 value 和其他属性'''
    #     not_visible_attr = ['_value','special variables']  # 不可见属性列表
    #     base_attrs = super().__dir__()  # 获取所有现有属性
    #     return [attr for attr in base_attrs if attr not in not_visible_attr]  # 过滤掉 _value

    def __setattr__(self, name, value):
        # 禁止修改现有属性
        if name == 'value':
            raise AttributeError(f"Cannot modify the '{name}' attribute, it is read-only.")
        super().__setattr__(name, value)

    def __mul__(self, other)-> 'Dimension':
        if not isinstance(other, Dimension):
            raise TypeError("Multiplication is only supported between Dimension instances.")
        # new_dims = self.value + other.value
        return Dimension(self.value + other.value)
    
    def __truediv__(self, other)-> 'Dimension':
        if not isinstance(other, Dimension):
            raise TypeError("Division is only supported between Dimension instances.")
        # new_dims = self.value - other.value
        return Dimension(self.value - other.value)
    
    def __pow__(self, power)-> 'Dimension':
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be an integer or float.")
        # new_dims = self.value * power
        return Dimension(self.value * power)
    
    def __eq__(self, other)-> bool:
        # if not isinstance(other, Dimension):
        #     return False
        return isinstance(other, Dimension) and np.allclose(self.value, other.value, atol=1e-10)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self)-> str:
        labels = ['M', 'L', 'T', 'Θ', 'I', 'N', 'J']
        # components = []
        # for label, power in zip(labels, self.value):
        #     if np.isclose(power, 0.0):
        #         continue
        #     elif np.isclose(power, 1.0):
        #         components.append(f"{label}")
        #     elif np.isclose(power, -1.0):
        #         components.append(f"{label}^-1")
        #     else:
        #         components.append(f"{label}^{power}")
        # return " ".join(components) if components else "Dimensionless"
        # return f"Dimension({self.value.tolist()})"
        # return "Dimension(" + " ".join(self.__str__) + ")"
        components = [f"{label}^{power}" if not np.isclose(power, 1.0) else label 
                     for label, power in zip(labels, self.value) if not np.isclose(power, 0.0)]
        return " ".join(components) if components else "Dimensionless"
    
    def copy(self)-> 'Dimension':
        return Dimension(self.value.copy())
    
    def grad(self)-> 'Dimension':
        # 复制 dimensions 对象，确保可修改
        value_copy = self.value.copy()
        value_copy[1] -= 1
        return Dimension(value_copy)

# Define common dimensions

dimless = Dimension()  # Dimensionless

mass_dim = Dimension(M=1)                                          # Mass [kg]
length_dim = Dimension(dim_list=[0, 1, 0, 0, 0, 0, 0])             # Length [m]
time_dim = Dimension(T=1)                                          # Time [s]
temperature_dim = Dimension(dim_list=[0, 0, 0, 1, 0, 0, 0])        # Temperature [K]
current_dim = Dimension(I=1)                                       # Electric Current [A]
amount_dim = Dimension(dim_list=[0, 0, 0, 0, 0, 1, 0])             # Amount of Substance [mol]
luminous_intensity_dim = Dimension(J=1)                            # Luminous Intensity [cd]

# Composite dimensions

velocity_dim = length_dim / time_dim             # Velocity [m/s]
acceleration_dim = length_dim / time_dim**2      # Acceleration [m/s²]
force_dim = mass_dim * acceleration_dim          # Force [kg·m/s²]
pressure_dim = force_dim / length_dim**2         # Pressure [kg/(m·s²)]
density_dim = mass_dim / length_dim**3           # Density [kg/m³]
volume_dim = length_dim**3                       # Volume [m³]
area_dim = length_dim**2                         # Area [m²]
flux_dim = mass_dim / time_dim                   # Flux [kg/s]