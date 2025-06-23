import os
import sys
import numpy as np
# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the path to the pyFVM directory
src_path = os.path.abspath(os.path.join(current_dir,'..', 'src'))
print(f"src 路径: {src_path}")  # 打印路径，确保正确
sys.path.insert(0, src_path)

from cfdtool.quantities import Quantity as Q_
import cfdtool.dimensions as dm
import cfdtool.Math as mth


local_centre = np.zeros(3)  # 标量
face_centroid = Q_([2, 3, 4], dm.Dimension([0, 1, 0, 0, 0, 0, 0]))  # 带量纲
# 触发 __add__，这应该抛出错误
# result1 = face_centroid + local_centre  # 调用 __add__

# 触发 __radd__，这也应该抛出错误
# result2 = local_centre + face_centroid  # 调用 __radd__
LL=dm.Dimension(L=1)
MM=dm.Dimension(M=1)
LL*MM
q0=dm.dimless
dm.mass_dim
dm.Dimension(dim_list=[0, 1, 0, 0, 0, 0, 0])
dm.length_dim*dm.pressure_dim
q1=Q_([1, 2, 3, 4, 5],dm.length_dim)
q2=Q_([30, 40, 50, 40, 44],dm.length_dim)
q3=Q_([3, 4, 5, 4, 44],dm.pressure_dim)
q4=q1+q2
q5=q1*q2
q6=q1/q2


print(q2)
q2+=q1
print(q1)
print(q2)
q2[3:]+=q1[3:]
print(q2)
q2[3:].value+=q1[3:].value
print(q2)



print('--------------------------------------------------\n')
q7=q3
q8=Q_(q3.value,q3.dimension)
print(q3)
q7.value[2]=33.3
print(q7)
print(q3)
print(q8)
print('--------------------------------------------------\n')
q8.value=q3.value
q3.value[2]=111
print(q7)
print(q8)


print(np.linalg.norm(q1))
# print(q1.apply(np.linalg.norm))
print(q1)
print(q2)
q2[2:5]=q1[2:5]
print(q2)
q2 += q1
print(q2)
q1.value[2]=300
print(q2)
q2[2:5]=q6[2:5]
# q5=q1-q3