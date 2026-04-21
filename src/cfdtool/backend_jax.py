"""
backend_jax.py — JAX 后端实现

基于 jax.numpy 实现与 NumpyBackend 相同的接口。
支持 jax.jit 编译加速，未来可扩展到 GPU (jax GPU backend)。

使用方式：
    from cfdtool.backend_jax import JaxBackend
    from cfdtool.backend import set_backend

    set_backend(JaxBackend())  # 一行切换

注意事项：
    1. JAX 数组不可变，add_at / subtract_at 返回新数组
    2. jax.jit 要求所有 shape 在编译时确定（或通过 static_argnums 标注）
    3. 随机数需要显式 PRNG key
    4. jax.numpy 兼容大部分 numpy API，但不支持所有高级索引模式
"""

from __future__ import annotations
from typing import Any, Sequence, Optional
import numpy as np
from cfdtool.backend import Backend


class JaxBackend(Backend):
    """
    JAX 后端实现。

    所有方法委托给 jax.numpy，支持 jax.jit 编译加速。
    首次调用时会触发 XLA 编译（较慢），后续调用走编译缓存（极快）。
    """

    def __init__(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        self._jnp = jnp

    @property
    def name(self) -> str:
        return "jax"

    # --- 数组创建 ---
    def zeros(self, shape, dtype=np.float64):
        return self._jnp.zeros(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=np.float64):
        return self._jnp.full(shape, fill_value, dtype=dtype)

    def copy(self, x):
        # jnp.array(x) 创建可变副本（DeviceArray）
        return self._jnp.array(x)

    def zeros_like(self, x):
        return self._jnp.zeros_like(x)

    def full_like(self, x, fill_value):
        return self._jnp.full_like(x, fill_value)

    def arange(self, stop):
        return self._jnp.arange(stop)

    # --- 逐元素 ---
    def abs(self, x):
        return self._jnp.abs(x)

    def sqrt(self, x):
        return self._jnp.sqrt(x)

    # --- 向量运算 ---
    def dot(self, a, b):
        return self._jnp.einsum('...i,...i->...', a, b)

    def norm(self, x, axis=None):
        # jnp.linalg.norm 兼容 numpy 的 axis 参数
        return self._jnp.linalg.norm(x, axis=axis)

    # --- 规约 ---
    def sum(self, x, axis=None):
        return self._jnp.sum(x, axis=axis)

    def max(self, x, axis=None):
        return self._jnp.max(x, axis=axis)

    def min(self, x, axis=None):
        return self._jnp.min(x, axis=axis)

    def mean(self, x, axis=None):
        return self._jnp.mean(x, axis=axis)

    def _to_jax_indices(self, indices):
        """将 numpy 数组 / Python list / range 等转为 JAX 数组索引。
        slice / tuple-of-slice 等原生支持的索引原样返回。
        """
        # slice 和 None 不需要转换
        if isinstance(indices, (slice, type(None))):
            return indices
        # 元组：逐元素处理（混合 slice 和数组的情况，如 (slice(None,N), slice(None))）
        if isinstance(indices, tuple):
            return tuple(self._to_jax_indices(x) for x in indices)
        if isinstance(indices, self._jnp.ndarray):
            return indices
        return self._jnp.array(indices)

    # --- Scatter（不可变语义，返回新数组）---
    def add_at(self, arr, indices, values):
        """
        JAX 不可变 scatter-add。
        等价于 numpy 的 np.add.at(arr, indices, values)，
        但返回新数组而非原地修改。
        """
        return arr.at[self._to_jax_indices(indices)].add(values)

    def subtract_at(self, arr, indices, values):
        """JAX 不可变 scatter-subtract，返回新数组。"""
        return arr.at[self._to_jax_indices(indices)].subtract(values)

    # --- 形状 / 重排 ---
    def transpose(self, x, axes=None):
        return self._jnp.transpose(x, axes=axes)

    def squeeze(self, x, axis=None):
        return self._jnp.squeeze(x, axis=axis)

    def column_stack(self, arrays):
        return self._jnp.column_stack(arrays)

    def einsum(self, subscripts, *operands):
        # JAX 原生支持 jnp.einsum，且可被 jax.jit 编译
        return self._jnp.einsum(subscripts, *operands)

    # --- 索引赋值（不可变语义，返回新数组）---
    def set_at(self, arr, indices, values):
        """JAX 不可变索引赋值，返回新数组。"""
        return arr.at[self._to_jax_indices(indices)].set(values)

    # --- 条件 ---
    def where(self, condition, x, y):
        return self._jnp.where(condition, x, y)
