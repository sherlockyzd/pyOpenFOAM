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
result2 = local_centre + face_centroid  # 调用 __radd__

q1=Q_([1, 2, 3, 4, 5],dm.length_dim)
q2=Q_([3, 4, 5, 4, 44],dm.length_dim)
q3=Q_([3, 4, 5, 4, 44],dm.pressure_dim)
q4=q1+q2
q6=q1*q2
q7=q1/q2

np.linalg.norm(q1.value)
q1.apply_function(np.linalg.norm)
q2[2:5]=q1[2:5]
q1.value[2]=300
q2[2:5]=q6[2:5]
# q5=q1-q3