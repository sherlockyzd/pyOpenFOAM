# pyOpenFOAM: Python Finite Volume Method CFD Solver

[中文版本](#中文版本) | [English Version](#english-version)

---

## English Version

### 🌊 Overview

**pyOpenFOAM** is a Python-based Computational Fluid Dynamics (CFD) library implementing the Finite Volume Method (FVM), designed for simulating fluid flow and heat transfer phenomena. The solver provides a complete toolset for mesh topology, equation assembly, linear solving, boundary conditions, and result visualization.

The solver adopts OpenFOAM mesh format and references discretization schemes from *"The Finite Volume Method in Computational Fluid Dynamics"* (Moukalled et al.).

### ✨ Key Features

- **Multi-dimensional mesh support** with various boundary conditions
- **Multiple linear solvers**: AMG, PCG, ILU, SOR
- **Steady-state and transient simulation** (SIMPLE / PISO algorithms)
- **Multi-backend architecture**: NumPy (default) and JAX backends with a single-line config switch
- **Automatic residual plotting**: Saves `residualHistory.png` after simulation
- **OpenFOAM compatibility**: Native support for OpenFOAM mesh formats and case structure
- **Dimensional consistency management**: Automatic physical dimension checking

### 🚀 Quick Start

#### Prerequisites

- Python 3.8+ (recommended: Python 3.10)
- NumPy, SciPy, Matplotlib

#### Installation

```bash
# Clone the repository
git clone https://github.com/sherlockyzd/pyOpenFOAM.git
cd pyOpenFOAM

# Create and activate conda environment
conda create --name pyOF python=3.10 numpy matplotlib scipy
conda activate pyOF
```

#### Run Examples

```bash
# Lid-driven cavity flow (structured mesh)
cd example/cavity
python pyFVMScript.py

# Elbow pipe flow (unstructured mesh)
cd ../elbow
python pyFVMScript.py

# Heat transfer in flange
cd ../flange
python pyFVMScript.py
```

### ⚡ Backend Switching (NumPy / JAX)

pyOpenFOAM supports multiple array computation backends via a unified `Backend` abstraction layer. Switch backends by editing `src/config.py`:

```python
# src/config.py
cfdBackend = 'numpy'   # Default, fast for production runs
# cfdBackend = 'jax'   # Enable JAX backend (requires: pip install jax jaxlib)
```

**NumPy backend**: Production-ready, optimal performance (~6 s for cavity example).

**JAX backend**: Currently ~2-3x slower without JIT, but enables:
- `jax.jit` — compile compute graphs for GPU acceleration
- `jax.grad` — automatic differentiation for inverse problems / data assimilation
- `jax.vmap` / `jax.pmap` — vectorized and multi-device parallelism

Install JAX:
```bash
pip install jax jaxlib
```

### 💻 Basic Usage

```python
from pyFVM.Region import Region

case_path = "/path/to/your/case"
cfd_case = Region(case_path)
cfd_case.RunCase()
# After simulation: residualHistory.png is saved in the case directory
```

### 📁 Project Structure

```
pyOpenFOAM/
├── src/
│   ├── cfdtool/                # Utilities
│   │   ├── backend.py          # Backend ABC + NumpyBackend (24 APIs)
│   │   ├── backend_jax.py      # JaxBackend implementation
│   │   ├── config.py           # Backend selection (numpy/jax)
│   │   ├── Interpolate.py      # Interpolation schemes
│   │   ├── cfdPlot.py          # Residual history plotting
│   │   ├── Solve.py            # Linear solvers (scipy.sparse)
│   │   └── IO.py               # File I/O
│   └── pyFVM/                  # Core FVM implementation
│       ├── Region.py           # Main simulation controller
│       ├── Model.py            # Physics model definitions
│       ├── Assemble.py         # Equation assembly
│       ├── Polymesh.py         # Mesh handling
│       ├── Coefficients.py     # Matrix assembly (acnb/ldu/csr/coo)
│       ├── Fluxes.py           # Flux computation
│       ├── Gradient.py         # Gradient computation
│       ├── Interpolate.py      # Face interpolation
│       ├── Field.py            # Field data structure
│       ├── Equation.py         # Equation management
│       └── FoamDictionaries.py # OpenFOAM file parsing
├── example/
│   ├── cavity/                 # Lid-driven cavity flow (structured)
│   ├── elbow/                  # Pipe elbow flow (unstructured)
│   └── flange/                 # Heat transfer case
└── README.md
```

### 🔬 Supported Physics

- **Momentum equations** (Navier-Stokes)
- **Continuity equation** (mass conservation)
- **Energy equation** (heat transfer)
- **Pressure-velocity coupling** (SIMPLE / PISO algorithms)

### 📊 Example Cases

| Case | Type | Mesh | Description |
|------|------|------|-------------|
| `cavity` | Incompressible | Structured | Classic lid-driven cavity benchmark |
| `elbow` | Incompressible | Unstructured | 90-degree pipe elbow flow |
| `flange` | Heat transfer | Structured | Thermal conduction in a flange |

### 🤝 Contributing

Contributions are welcome:
- 🐛 Bug reports and fixes
- 📖 Documentation improvements
- 🚀 Performance optimizations (JIT, GPU)
- ✨ New features and solvers
- 🧪 Test cases and validation

### 📄 License

[MIT License](LICENSE)

### 📧 Contact

- **Email**: yuzd17@tsinghua.org.cn
- **Issues**: [GitHub Issues](https://github.com/sherlockyzd/pyOpenFOAM/issues)

---

## 中文版本

### 🌊 项目概述

**pyOpenFOAM** 是一个基于有限体积方法（FVM）的计算流体动力学（CFD）库，用于模拟流体流动和热传递现象。该库提供了一套完整的工具来处理网格拓扑关系、方程组装、线性求解、边界条件设置和结果可视化。

求解器采用 OpenFOAM 网格格式，主要离散格式参考了《The Finite Volume Method in Computational Fluid Dynamics》一书。

### ✨ 主要特性

- **多维网格支持**，兼容多种边界条件
- **多种线性求解器**：AMG、PCG、ILU、SOR
- **稳态和瞬态模拟**（SIMPLE / PISO 算法）
- **多后端架构**：NumPy（默认）和 JAX 后端，一行配置切换
- **自动残差绘图**：模拟结束后保存 `residualHistory.png`
- **OpenFOAM 兼容**：原生支持 OpenFOAM 网格格式和算例结构
- **量纲一致性管理**：物理量纲自动检查

### 🚀 快速开始

#### 环境要求

- Python 3.8+（推荐 Python 3.10）
- NumPy、SciPy、Matplotlib

#### 安装

```bash
git clone https://github.com/sherlockyzd/pyOpenFOAM.git
cd pyOpenFOAM

# 创建并激活 conda 环境
conda create --name pyOF python=3.10 numpy matplotlib scipy
conda activate pyOF
```

#### 运行算例

```bash
# 顶盖驱动方腔流（结构化网格）
cd example/cavity
python pyFVMScript.py

# 90° 弯管流动（非结构化网格）
cd ../elbow
python pyFVMScript.py

# 法兰盘传热
cd ../flange
python pyFVMScript.py
```

### ⚡ 后端切换（NumPy / JAX）

pyOpenFOAM 通过统一的 `Backend` 抽象层支持多种数组计算后端。修改 `src/config.py` 即可切换：

```python
# src/config.py
cfdBackend = 'numpy'   # 默认，生产运行性能最优
# cfdBackend = 'jax'   # 启用 JAX 后端（需先：pip install jax jaxlib）
```

| 后端 | 性能 | 适用场景 |
|------|------|---------|
| **NumPy** | 最快（cavity ~6s） | 日常开发、生产计算 |
| **JAX** | 慢 2-3x（无 JIT） | 自动微分、GPU 并行、JIT 编译 |

JAX 后端安装：
```bash
pip install jax jaxlib
```

JAX 后端的真正价值在于为以下能力铺路：
- `jax.jit` — 编译计算图实现 GPU 加速
- `jax.grad` — 自动微分，支持反演优化、数据同化
- `jax.vmap` / `jax.pmap` — 向量化与多设备并行

### 💻 基本使用

```python
from pyFVM.Region import Region

case_path = "/path/to/case"
cfd_case = Region(case_path)
cfd_case.RunCase()
# 模拟结束后，算例目录下自动生成 residualHistory.png
```

### 📁 项目结构

```
pyOpenFOAM/
├── src/
│   ├── cfdtool/                # 工具模块
│   │   ├── backend.py          # 后端抽象基类 + NumpyBackend（24 个 API）
│   │   ├── backend_jax.py      # JaxBackend 实现
│   │   ├── config.py           # 后端选择配置（numpy/jax）
│   │   ├── Interpolate.py      # 插值格式
│   │   ├── cfdPlot.py          # 残差历史绘图
│   │   ├── Solve.py            # 线性求解器（scipy.sparse）
│   │   └── IO.py               # 文件读写
│   └── pyFVM/                  # 核心 FVM 实现
│       ├── Region.py           # 主仿真控制器
│       ├── Model.py            # 物理模型定义
│       ├── Assemble.py         # 方程组装
│       ├── Polymesh.py         # 网格处理
│       ├── Coefficients.py     # 矩阵装配（acnb/ldu/csr/coo）
│       ├── Fluxes.py           # 通量计算
│       ├── Gradient.py         # 梯度计算
│       ├── Field.py            # 场数据结构
│       ├── Equation.py         # 方程管理
│       └── FoamDictionaries.py # OpenFOAM 文件解析
├── example/
│   ├── cavity/                 # 顶盖驱动方腔流（结构化网格）
│   ├── elbow/                  # 90° 弯管流动（非结构化网格）
│   └── flange/                 # 法兰盘传热
└── README.md
```

### 🔬 支持的物理现象

- **动量方程**（Navier-Stokes 方程）
- **连续性方程**（质量守恒）
- **能量方程**（传热）
- **压力-速度耦合**（SIMPLE / PISO 算法）

### 📊 算例

| 算例 | 类型 | 网格 | 说明 |
|------|------|------|------|
| `cavity` | 不可压缩流 | 结构化 | 经典顶盖驱动方腔基准算例 |
| `elbow` | 不可压缩流 | 非结构化 | 90° 弯管内部流动 |
| `flange` | 传热 | 结构化 | 法兰盘热传导模拟 |

### 🤝 贡献指南

欢迎各种形式的贡献：
- 🐛 错误报告与修复
- 📖 文档改进
- 🚀 性能优化（JIT、GPU）
- ✨ 新功能与求解器
- 🧪 测试算例与验证

### 📄 开源许可

[MIT 许可证](LICENSE)

### 📧 联系方式

- **邮箱**：yuzd17@tsinghua.org.cn
- **问题反馈**：[GitHub Issues](https://github.com/sherlockyzd/pyOpenFOAM/issues)

---

**⭐ 觉得有用就 Star 一下吧！**
