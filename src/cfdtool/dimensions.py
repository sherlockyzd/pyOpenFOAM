# dimensions.py

class Dimension:
    """
    表示物理量的量纲。
    量纲以七个基础单位的幂次表示：[M, L, T, Θ, I, N, J]
    """
    def __init__(self, M=0, L=0, T=0, Theta=0, I=0, N=0, J=0):
        self.dimensions = (M, L, T, Theta, I, N, J)
    
    def __mul__(self, other):
        if not isinstance(other, Dimension):
            raise TypeError("Multiplication is only supported between Dimension instances.")
        return Dimension(*(a + b for a, b in zip(self.dimensions, other.dimensions)))
    
    def __truediv__(self, other):
        if not isinstance(other, Dimension):
            raise TypeError("Division is only supported between Dimension instances.")
        return Dimension(*(a - b for a, b in zip(self.dimensions, other.dimensions)))
    
    def __pow__(self, power):
        return Dimension(*(a * power for a in self.dimensions))
    
    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return False
        return self.dimensions == other.dimensions
    
    def __str__(self):
        labels = ['M', 'L', 'T', 'Theta', 'I', 'N', 'J']
        return " ".join(f"{label}^{power}" if power != 0 else "" for label, power in zip(labels, self.dimensions)).strip()
    
    def __repr__(self):
        return f"Dimension{self.dimensions}"

# 定义常用量纲

dimless = Dimension()

mass_dim = Dimension(M=1)               # 质量 [kg]
length_dim = Dimension(L=1)             # 长度 [m]
time_dim = Dimension(T=1)               # 时间 [s]
temperature_dim = Dimension(Theta=1)    # 温度 [K]
current_dim = Dimension(I=1)            # 电流 [A]
amount_dim = Dimension(N=1)             # 物质的量 [mol]
luminous_intensity_dim = Dimension(J=1) # 发光强度 [cd]

# 复合量纲

velocity_dim = length_dim / time_dim             # 速度 [m/s]
acceleration_dim = length_dim / time_dim**2      # 加速度 [m/s²]
force_dim = mass_dim * acceleration_dim          # 力 [kg·m/s²]
pressure_dim = force_dim / length_dim**2         # 压力 [kg/(m·s²)]
density_dim = mass_dim / length_dim**3           # 密度 [kg/m³]
volume_dim = length_dim**3                        # 体积 [m³]
area_dim = length_dim**2                          # 面积 [m²]
