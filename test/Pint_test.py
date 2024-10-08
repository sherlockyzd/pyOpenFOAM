import pint
import numpy as np

# 创建 UnitRegistry
ureg = pint.UnitRegistry()

# 创建 Quantity 对象
length = 5 * ureg.meter
time = 10 * ureg.second
velocity = length / time  # 计算速度

print(f"长度: {length}")          # 输出: 长度: 5 meter
print(f"时间: {time}")            # 输出: 时间: 10 second
print(f"速度: {velocity}")        # 输出: 速度: 0.5 meter / second

# 数组操作
length_array = np.array([1, 2, 3]) * ureg.meter
scaled_length = length_array * 2
print(f"缩放后的长度数组: {scaled_length}")  # 输出: 缩放后的长度数组: [2 4 6] meter

# 单位转换
length_cm = length.to(ureg.centimeter)
print(f"长度（厘米）: {length_cm}")  # 输出: 长度（厘米）: 500.0 centimeter

# 自定义函数应用
def multiply_length(q, factor):
    return q * factor

new_length = multiply_length(length, 3)
print(f"新的长度: {new_length}")  # 输出: 新的长度: 15 meter

# 无量纲操作
dimensionless_quantity = 2 * ureg.dimensionless
result = length * dimensionless_quantity
print(f"结果: {result}")  # 输出: 结果: 10 meter

# 错误操作示例
try:
    invalid = length + time
except pint.DimensionalityError as e:
    print(f"错误: {e}")  # 输出: 错误: Cannot add 'meter' and 'second'

# 自定义单位
ureg.define('lightyear = 9.461e15 * meter')
distance = 2 * ureg.lightyear
print(f"距离: {distance}")  # 输出: 距离: 2 lightyear

# 单位转换
distance_km = distance.to(ureg.kilometer)
print(f"距离（千米）: {distance_km}")  # 输出: 距离（千米）: 1.8922e+19 kilometer

pass
