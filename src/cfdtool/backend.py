"""
backend.py — 数组计算后端抽象层

设计原则：
    - Backend 定义"做什么"的接口，不关心数组具体类型（ndarray/jax.Array/cupy.ndarray）
    - NumpyBackend 是默认实现，直接委托给 numpy/scipy
    - 未来可添加 JaxBackend / CupyBackend 实现 GPU 或 JIT 加速
    - 所有 FVM 算子（Math / Interpolate / Gradient / Assemble）通过 backend 实例操作数组

使用方式：
    # 用户代码只需一行导入即可使用：
    from cfdtool.backend import be

    x = be.zeros((100, 3))   # 创建数组
    mag = be.norm(x, axis=1) # 计算模

    # 切换后端：修改 src/config.py 中的 cfdBackend = 'jax'
    # 或运行时切换：from cfdtool.backend import set_backend; set_backend(JaxBackend())
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence, Optional
import numpy as np


# ============================================================
# 全局后端管理
# ============================================================

_current_backend: "Backend" = None  # type: ignore


def get_backend() -> "Backend":
    """获取当前全局后端实例。首次调用时默认创建 NumpyBackend。"""
    global _current_backend
    if _current_backend is None:
        _current_backend = NumpyBackend()
    return _current_backend


def set_backend(backend: "Backend") -> None:
    """设置全局后端实例，同时更新模块级 be 引用。"""
    global _current_backend, be
    _current_backend = backend
    be = backend


def backend_name() -> str:
    """返回当前后端名称。"""
    return get_backend().name


def _auto_init_backend() -> "Backend":
    """
    根据 config.cfdBackend 自动初始化全局后端。
    由模块加载时调用，用户无需关心。
    """
    try:
        from config import cfdBackend
    except ImportError:
        cfdBackend = 'numpy'

    if cfdBackend == 'jax':
        try:
            from cfdtool.backend_jax import JaxBackend
            return JaxBackend()
        except ImportError:
            import warnings
            warnings.warn("cfdBackend='jax' 但 jax 未安装，回退到 numpy", stacklevel=2)
            # return NumpyBackend()
            raise IOError("cfdBackend='jax' 但 jax 未安装，请安装 jax 或修改 cfdBackend 设置")
    else:
        return NumpyBackend()


# ============================================================
# Backend 抽象基类
# ============================================================

class Backend(ABC):
    """
    数组计算后端的抽象接口。

    子类需实现所有 @abstractmethod 方法。
    算术运算符（+、-、*、/、**）由数组类型自身支持，Backend 不封装。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称，如 'numpy', 'jax', 'cupy'"""
        ...

    # ----------------------------------------------------------
    # 1. 数组创建
    # ----------------------------------------------------------
    @abstractmethod
    def zeros(self, shape: Sequence[int], dtype=np.float64) -> Any:
        """创建全零数组"""
        ...

    @abstractmethod
    def full(self, shape: Sequence[int], fill_value: float, dtype=np.float64) -> Any:
        """创建填充指定值的数组"""
        ...

    @abstractmethod
    def copy(self, x: Any) -> Any:
        """深拷贝数组"""
        ...

    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        """创建与 x 同形状的全零数组"""
        ...

    @abstractmethod
    def full_like(self, x: Any, fill_value: float) -> Any:
        """创建与 x 同形状、填充指定值的数组"""
        ...

    @abstractmethod
    def arange(self, stop: int) -> Any:
        """等价于 np.arange(stop)，返回 0 到 stop-1 的一维整数数组"""
        ...

    # ----------------------------------------------------------
    # 2. 逐元素数学函数
    # ----------------------------------------------------------
    @abstractmethod
    def abs(self, x: Any) -> Any:
        """逐元素绝对值"""
        ...

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        """逐元素平方根"""
        ...

    # ----------------------------------------------------------
    # 3. 向量运算
    # ----------------------------------------------------------
    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        """
        向量点积：沿最后一轴做内积。
        a, b 形状为 (..., dim)，返回形状为 (...)。
        等价于 np.einsum('...i,...i->...', a, b)
        """
        ...

    @abstractmethod
    def norm(self, x: Any, axis: Optional[int] = None) -> Any:
        """
        计算范数（L2）。
        axis=None 时返回标量（全数组 Frobenius 范数）。
        axis=int 时沿指定轴归约。
        """
        ...

    # ----------------------------------------------------------
    # 4. 规约运算
    # ----------------------------------------------------------
    @abstractmethod
    def sum(self, x: Any, axis: Optional[int] = None) -> Any:
        """求和"""
        ...

    @abstractmethod
    def max(self, x: Any, axis: Optional[int] = None) -> Any:
        """最大值"""
        ...

    @abstractmethod
    def min(self, x: Any, axis: Optional[int] = None) -> Any:
        """最小值"""
        ...

    @abstractmethod
    def mean(self, x: Any, axis: Optional[int] = None) -> Any:
        """均值"""
        ...

    # ----------------------------------------------------------
    # 5. Scatter 操作（原子化索引累加）
    # ----------------------------------------------------------
    @abstractmethod
    def add_at(self, arr: Any, indices: Any, values: Any) -> Any:
        """
        等价于 np.add.at(arr, indices, values)，返回修改后的数组。

        NumpyBackend 原地修改并返回自身（零开销）。
        JaxBackend 返回新数组（不可变语义）。
        调用方应始终使用返回值：arr = be.add_at(arr, idx, val)。
        """
        ...

    @abstractmethod
    def subtract_at(self, arr: Any, indices: Any, values: Any) -> Any:
        """
        等价于 np.subtract.at(arr, indices, values)，返回修改后的数组。
        调用方应始终使用返回值：arr = be.subtract_at(arr, idx, val)。
        """
        ...

    # ----------------------------------------------------------
    # 6. 形状 / 重排操作
    # ----------------------------------------------------------
    @abstractmethod
    def transpose(self, x: Any, axes: Optional[Sequence[int]] = None) -> Any:
        """转置。axes=None 时等价于 np.transpose(x)"""
        ...

    @abstractmethod
    def squeeze(self, x: Any, axis: Optional[int] = None) -> Any:
        """去除长度为 1 的维度"""
        ...

    @abstractmethod
    def column_stack(self, arrays: Sequence[Any]) -> Any:
        """等价于 np.column_stack：将 1D/2D 数组水平堆叠"""
        ...

    @abstractmethod
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """
        通用 Einstein 求和。
        JAX 原生支持 jnp.einsum，因此保留此接口。
        """
        ...

    # ----------------------------------------------------------
    # 7. 条件操作
    # ----------------------------------------------------------
    @abstractmethod
    def set_at(self, arr: Any, indices: Any, values: Any) -> Any:
        """
        等价于 arr[indices] = values，返回修改后的数组。

        NumpyBackend 原地修改并返回自身（零开销）。
        JaxBackend 返回新数组（不可变语义）。
        调用方应始终使用返回值：arr = be.set_at(arr, idx, val)。
        """
        ...

    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """
        等价于 np.where(condition, x, y)。
        condition 为真时取 x，否则取 y。
        """
        ...


# ============================================================
# NumpyBackend — 默认实现（生产级）
# ============================================================

class NumpyBackend(Backend):
    """
    NumPy 后端实现。

    所有方法直接委托给 numpy，零抽象开销（函数调用仅在模块加载时绑定）。
    """

    def __init__(self):
        import numpy as _np
        self._np = _np

    @property
    def name(self) -> str:
        return "numpy"

    # --- 数组创建 ---
    def zeros(self, shape, dtype=np.float64):
        return self._np.zeros(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=np.float64):
        return self._np.full(shape, fill_value, dtype=dtype)

    def copy(self, x):
        return x.copy()

    def zeros_like(self, x):
        return self._np.zeros_like(x)

    def full_like(self, x, fill_value):
        return self._np.full_like(x, fill_value)

    def arange(self, stop):
        return self._np.arange(stop)

    # --- 逐元素 ---
    def abs(self, x):
        return self._np.abs(x)

    def sqrt(self, x):
        return self._np.sqrt(x)

    # --- 向量运算 ---
    def dot(self, a, b):
        return self._np.einsum('...i,...i->...', a, b)

    def norm(self, x, axis=None):
        return self._np.linalg.norm(x, axis=axis)

    # --- 规约 ---
    def sum(self, x, axis=None):
        return self._np.sum(x, axis=axis)

    def max(self, x, axis=None):
        return self._np.max(x, axis=axis)

    def min(self, x, axis=None):
        return self._np.min(x, axis=axis)

    def mean(self, x, axis=None):
        return self._np.mean(x, axis=axis)

    # --- Scatter ---
    def add_at(self, arr, indices, values):
        self._np.add.at(arr, indices, values)
        return arr

    def subtract_at(self, arr, indices, values):
        self._np.subtract.at(arr, indices, values)
        return arr

    # --- 形状 / 重排 ---
    def transpose(self, x, axes=None):
        return self._np.transpose(x, axes=axes)

    def squeeze(self, x, axis=None):
        return self._np.squeeze(x, axis=axis)

    def column_stack(self, arrays):
        return self._np.column_stack(arrays)

    def einsum(self, subscripts, *operands):
        return self._np.einsum(subscripts, *operands)

    # --- 索引赋值 ---
    def set_at(self, arr, indices, values):
        arr[indices] = values
        return arr

    # --- 条件 ---
    def where(self, condition, x, y):
        return self._np.where(condition, x, y)


# ============================================================
# 模块级单例：所有 FVM 文件只需 from cfdtool.backend import be
# ============================================================

_current_backend = _auto_init_backend()
be = _current_backend
