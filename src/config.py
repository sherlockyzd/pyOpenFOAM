# config.py — pyOpenFOAM 全局配置

# 流动类型：True=可压缩，False=不可压缩
cfdIsCompressible = False
# 非线性修正
pp_nonlinear_corrected = False

# 数组计算后端：'numpy'（默认）| 'jax'（需 pip install jax jaxlib）
cfdBackend = 'numpy'
# cfdBackend = 'jax'  # 切换到 JAX 后端进行测试
