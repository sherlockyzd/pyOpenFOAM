<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:003545&height=180&section=header&text=pyOpenFOAM&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=32&desc=Python%20Finite%20Volume%20Method%20CFD%20Solver&descSize=16&descColor=90CAF9" alt="pyOpenFOAM Banner"/>
</p>

<p align="center">
  <a href="https://github.com/sherlockyzd/pyOpenFOAM/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8+-green.svg" alt="Python"/></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Backend-013243?logo=numpy" alt="NumPy"/></a>
  <a href="https://github.com/google/jax"><img src="https://img.shields.io/badge/JAX-Backend-FF6F00?logo=jax" alt="JAX"/></a>
  <a href="https://github.com/sherlockyzd/pyOpenFOAM"><img src="https://img.shields.io/badge/CFD-FVM_Solver-00B4D8" alt="CFD"/></a>
</p>

<p align="center">
  <b>中文</b> | <a href="#english-version">English</a>
</p>

---

## pyOpenFOAM

**基于有限体积方法（FVM）的 Python 计算流体动力学求解器**，实现了完整的网格拓扑处理、方程组装、线性求解、边界条件设置和结果可视化。兼容 OpenFOAM 网格格式，支持 NumPy / JAX 双后端。

### 主要特性

- **多维网格支持** — 结构化与非结构化网格，多种边界条件
- **多种线性求解器** — AMG、PCG、ILU、SOR
- **稳态/瞬态模拟** — SIMPLE / PISO 算法
- **双后端架构** — NumPy（默认）和 JAX，一行配置切换
- **自动残差绘图** — 模拟结束自动保存 `residualHistory.png`
- **OpenFOAM 兼容** — 原生支持 OpenFOAM 网格格式和算例结构

### 快速开始

```bash
git clone https://github.com/sherlockyzd/pyOpenFOAM.git
cd pyOpenFOAM

conda create --name pyOF python=3.10 numpy matplotlib scipy
conda activate pyOF
```

```bash
# 顶盖驱动方腔流（结构化网格）
cd example/cavity && python pyFVMScript.py

# 90° 弯管流动（非结构化网格）
cd example/elbow && python pyFVMScript.py

# 法兰盘传热
cd example/flange && python pyFVMScript.py
```

### 基本使用

```python
from pyFVM.Region import Region

case = Region("/path/to/case")
case.RunCase()
# 模拟结束后，算例目录下自动生成 residualHistory.png
```

### 后端切换

修改 `src/config.py` 即可：

```python
cfdBackend = 'numpy'   # 默认，生产运行最快
# cfdBackend = 'jax'   # JAX 后端（需 pip install jax jaxlib）
```

| 后端 | 性能 | 适用场景 |
|:---:|:---:|---|
| **NumPy** | cavity ~6s | 日常开发、生产计算 |
| **JAX** | 慢 2-3x（无 JIT） | 自动微分、GPU 并行、JIT 编译 |

JAX 后端为以下能力铺路：
- `jax.jit` — 编译计算图 → GPU 加速
- `jax.grad` — 自动微分 → 反演优化、数据同化
- `jax.vmap` / `jax.pmap` — 向量化与多设备并行

### 支持的物理现象

- 动量方程（Navier-Stokes）
- 连续性方程（质量守恒）
- 能量方程（传热）
- 压力-速度耦合（SIMPLE / PISO）

### 算例

| 算例 | 类型 | 网格 | 说明 |
|:---:|:---:|:---:|---|
| `cavity` | 不可压缩流 | 结构化 | 经典顶盖驱动方腔基准 |
| `elbow` | 不可压缩流 | 非结构化 | 90° 弯管内部流动 |
| `flange` | 传热 | 结构化 | 法兰盘热传导模拟 |

### 项目结构

```
pyOpenFOAM/
├── src/
│   ├── cfdtool/                # 工具模块
│   │   ├── backend.py          # 后端抽象 + NumpyBackend（24 API）
│   │   ├── backend_jax.py      # JaxBackend 实现
│   │   ├── config.py           # 后端选择
│   │   ├── Solve.py            # 线性求解器
│   │   └── cfdPlot.py          # 残差绘图
│   └── pyFVM/                  # 核心 FVM
│       ├── Region.py           # 仿真控制器
│       ├── Assemble.py         # 方程组装
│       ├── Polymesh.py         # 网格处理
│       └── ...
├── example/
│   ├── cavity/                 # 方腔流
│   ├── elbow/                  # 弯管流
│   └── flange/                 # 传热
├── LICENSE
└── README.md
```

### 参考与致谢

- Moukalled, F., Mangani, L., & Darwish, M. *The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab*. Springer.
- OpenFOAM® — 开源 CFD 工具箱

### 许可证

[MIT License](LICENSE)

---

<p align="center">
  如果觉得有用，欢迎 <a href="https://github.com/sherlockyzd/pyOpenFOAM"><b>⭐ Star</b></a> 支持一下！
</p>

---
---

<p align="center">
  <b>English</b> | <a href="#pyopenfoam">中文</a>
</p>

## English Version

**pyOpenFOAM** is a Python-based Computational Fluid Dynamics (CFD) solver implementing the Finite Volume Method (FVM). It provides a complete toolkit for mesh topology, equation assembly, linear solving, boundary conditions, and result visualization. Compatible with OpenFOAM mesh formats, supporting NumPy / JAX dual backends.

### Key Features

- **Multi-dimensional mesh** — Structured and unstructured, various boundary conditions
- **Multiple linear solvers** — AMG, PCG, ILU, SOR
- **Steady-state & transient** — SIMPLE / PISO algorithms
- **Dual-backend architecture** — NumPy (default) and JAX, switch with one line
- **Auto residual plotting** — Saves `residualHistory.png` after simulation
- **OpenFOAM compatible** — Native support for OpenFOAM mesh format and case structure

### Quick Start

```bash
git clone https://github.com/sherlockyzd/pyOpenFOAM.git
cd pyOpenFOAM

conda create --name pyOF python=3.10 numpy matplotlib scipy
conda activate pyOF
```

### Example Cases

| Case | Type | Mesh | Description |
|:---:|:---:|:---:|---|
| `cavity` | Incompressible | Structured | Lid-driven cavity benchmark |
| `elbow` | Incompressible | Unstructured | 90-degree pipe elbow flow |
| `flange` | Heat transfer | Structured | Thermal conduction in a flange |

### License

[MIT License](LICENSE)
