import numpy as np

a = np.zeros(5)
idx = np.array([1, 1, 2, 3])
v   = np.ones_like(idx)

# 用高级索引做 +=
a[idx] += v
# 实际等价于：
#   tmp = a[idx]         # tmp = [0,0,0,0]
#   tmp += v             # tmp = [1,1,1,1]
#   a[idx] = tmp         # a[1]=1; a[2]=1; a[3]=1  （两次写 a[1] 都写成 1）
print(a)  # [0., 1., 1., 1., 0.]

# 而 np.add.at 会真正对重复位置做多次累加：
a = np.zeros(5)
np.add.at(a, idx, 1)
print(a)  # [0., 2., 1., 1., 0.]


a = np.zeros(5)
a[idx]=a[idx] + v
print(a)  # [0., 2., 1., 1., 0.]