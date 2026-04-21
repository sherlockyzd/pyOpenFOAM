import numpy as np
from cfdtool.backend import be


def cfdMag(valueVector):
    return cfdMag_np(valueVector)

def cfdMag_np(valueVector):
    """计算向量模（L2 范数）。

    支持单个向量（1D）或向量列表（2D）。
    通过 backend.norm 实现后端无关。
    """

        # raise TypeError("输入参数必须是一个 NumPy 数组")
    if valueVector.ndim == 1:
        return be.norm(valueVector)
    elif valueVector.ndim == 2:
        return be.norm(valueVector, axis=1)
    else:
        raise ValueError("valueVector 必须是一维或二维的 NumPy 数组。")

def cfdDot(Sf, U_f):
    return cfdDot_np(Sf, U_f)

def cfdDot_np(Sf, U_f):
    """
    计算每个面的面积向量 Sf 与速度向量 U_f 的点积。
    
    参数：
    - Sf: 面面积向量数组，形状为 (..., dim)。
    - U_f: 面上的速度向量数组，形状为 (..., dim)。
    
    返回：
    - flux: 每个面的通量值数组，形状为 (...,)。
    
    支持一维、二维和三维向量。通过 backend.dot 实现后端无关。
    """

        # raise ValueError(f"Shape mismatch: Sf.shape={Sf.shape} and U_f.shape={U_f.shape} must be the same.")
    return be.dot(Sf, U_f)

def cfdUnit(vector):
    """
    将输入的每个向量标准化为单位向量（模为1）。
    通过 backend.norm 实现后端无关。
    """

        # raise TypeError("输入参数必须是一个 NumPy 数组")
    epsilon = 1e-10
    norms = be.norm(vector, axis=1)
    norms = be.where(norms == 0, epsilon, norms)
    return vector / norms[:, np.newaxis]

def cfdScalarList(*args):
    """初始化标量列表。通过 backend.zeros/full 实现后端无关。"""

        # return []
    if len(args) == 1:
        n = args[0]
        the_scalar_list = be.zeros(n)
    elif len(args) == 2:
        n = args[0]
        value = args[1]
        the_scalar_list = be.full(n, value)
    return the_scalar_list

def cfdResidual(rc, method='norm'):
    """计算残差。通过 backend 实现后端无关。"""
    # be = get_backend()
    if method == 'RMS':
        rc_res = be.sqrt(be.mean(rc ** 2))
    elif method == 'norm':
        rc_res = be.norm(rc)
    elif method == 'mean':
        rc_res = be.mean(be.abs(rc))
    elif method == 'max':
        rc_res = be.max(be.abs(rc))
    elif method == 'sum':
        rc_res = be.sum(be.abs(rc))
    return rc_res
