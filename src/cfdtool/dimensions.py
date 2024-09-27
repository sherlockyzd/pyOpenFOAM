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
    __slots__ = ['value']

    def __init__(self, M=0, L=0, T=0, Theta=0, I=0, N=0, J=0, dim_list=None):
        if dim_list is not None:
            if not isinstance(dim_list, (list, tuple, np.ndarray)) or len(dim_list) != 7:
                raise ValueError("dim_list 必须是长度为7的列表、元组或NumPy数组，表示 [M, L, T, Θ, I, N, J]。")
            self.value = np.array(dim_list, dtype=np.int32)
        else:
            self.value = np.array([M, L, T, Theta, I, N, J], dtype=np.int32)
        
        # 将数组设置为只读，确保不可变性
        self.value.flags.writeable = False
    
    def __mul__(self, other):
        if not isinstance(other, Dimension):
            raise TypeError("Multiplication is only supported between Dimension instances.")
        new_dims = self.value + other.value
        return Dimension(dim_list=new_dims)
    
    def __truediv__(self, other):
        if not isinstance(other, Dimension):
            raise TypeError("Division is only supported between Dimension instances.")
        new_dims = self.value - other.value
        return Dimension(dim_list=new_dims)
    
    def __pow__(self, power):
        if not isinstance(power, int):
            raise TypeError("Power must be an integer.")
        new_dims = self.value * power
        return Dimension(dim_list=new_dims)
    
    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return False
        return np.array_equal(self.value, other.value)
    
    def __str__(self):
        labels = ['M', 'L', 'T', 'Theta', 'I', 'N', 'J']
        dim_str = " ".join(f"{label}^{power}" if power != 0 else "" for label, power in zip(labels, self.value))
        return " ".join(filter(None, dim_str.split()))
    
    def __repr__(self):
        return f"Dimension({self.value.tolist()})"

# 定义常用量纲

dimless = Dimension()

mass_dim = Dimension(M=1)                                          # 质量 [kg]
length_dim = Dimension(dim_list=[0, 1, 0, 0, 0, 0, 0])             # 长度 [m]
time_dim = Dimension(T=1)                                          # 时间 [s]
temperature_dim = Dimension(dim_list=[0, 0, 0, 1, 0, 0, 0])        # 温度 [K]
current_dim = Dimension(I=1)                                       # 电流 [A]
amount_dim = Dimension(dim_list=[0, 0, 0, 0, 0, 1, 0])             # 物质的量 [mol]
luminous_intensity_dim = Dimension(J=1)                            # 发光强度 [cd]

# 复合量纲

velocity_dim = length_dim / time_dim             # 速度 [m/s]
acceleration_dim = length_dim / time_dim**2      # 加速度 [m/s²]
force_dim = mass_dim * acceleration_dim          # 力 [kg·m/s²]
pressure_dim = force_dim / length_dim**2         # 压力 [kg/(m·s²)]
density_dim = mass_dim / length_dim**3           # 密度 [kg/m³]
volume_dim = length_dim**3                        # 体积 [m³]
area_dim = length_dim**2                          # 面积 [m²]
flux_dim = mass_dim / time_dim                    # 通量 [kg/s]