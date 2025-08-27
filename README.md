# pyOpenFOAM: Python Finite Volume Method CFD Solver

[中文版本](#中文版本) | [English Version](#english-version)

---

## English Version

### 🌊 Overview

**pyOpenFOAM** is a Python-based Computational Fluid Dynamics (CFD) library implementing the Finite Volume Method (FVM), designed for simulating fluid flow and heat transfer phenomena. This educational-oriented solver provides a complete toolset for handling mesh topology relationships, equation solving, boundary condition setup, and result visualization.

The solver library adopts OpenFOAM mesh format files and implements discretization schemes and solving approaches based on the principles from "*The Finite Volume Method in Computational Fluid Dynamics*". While currently developed entirely on the Python platform with NumPy, the computational performance has room for improvement - we welcome community contributions to enhance performance while maintaining the project's educational value for CFD practitioners and researchers.

### ✨ Key Features

- **Multi-dimensional mesh support** with various boundary conditions
- **Multiple linear solvers**: AMG (Algebraic Multigrid), PCG (Preconditioned Conjugate Gradient), ILU (Incomplete LU Decomposition), SOR (Successive Over-Relaxation)
- **Steady-state and transient simulation** capabilities
- **Rich interpolation and residual calculation** tools
- **Custom equation and coefficient** configuration support
- **Dimensional consistency management**: Automatic physical dimension checking and validation system ensuring correct dimensional relationships in all mathematical operations
- **OpenFOAM compatibility**: Native support for OpenFOAM mesh formats and case structure

### 🚀 Quick Start

#### Prerequisites
- Python 3.8+ (recommended: Python 3.10)
- NumPy, SciPy, Matplotlib

#### Installation & Running

```bash
# Clone the repository
git clone https://github.com/sherlockyzd/pyOpenFOAM.git
cd pyOpenFOAM

# Create and activate conda environment
conda create --name pyOpenFOAM python=3.10 numpy matplotlib scipy
conda activate pyOpenFOAM

# Run cavity flow example
cd example/cavity
python pyFVMScript.py
```

#### Alternative pip installation
```bash
pip install numpy scipy matplotlib
```

### 💻 Basic Usage

1. **Initialize simulation region**:
```python
from pyFVM.Region import Region

case_path = "/path/to/case"
cfd_case = Region(case_path)
```

2. **Run simulation**:
```python
cfd_case.RunCase()
```

3. **Visualization**: Use built-in plotting tools or export data for ParaView

### 📁 Project Structure

```
pyOpenFOAM/
├── src/                    # Source code
│   ├── pyFVM/             # Core FVM implementation
│   │   ├── Region.py      # Main simulation controller
│   │   ├── Model.py       # Physics model definitions
│   │   ├── Polymesh.py    # Mesh handling
│   │   ├── Solve.py       # Linear solvers
│   │   └── ...
│   └── cfdtool/           # Utilities and tools
├── example/               # Test cases
│   ├── cavity/           # Lid-driven cavity flow
│   └── flange/           # Heat transfer case
└── test/                 # Unit tests
```

### 🔬 Supported Physics

- **Momentum equations** (Navier-Stokes)
- **Continuity equation** (mass conservation)
- **Energy equation** (heat transfer)
- **Pressure-velocity coupling** (SIMPLE/PISO algorithms)

### 🛠 Development & Architecture

The solver follows a modular architecture:

- **Region**: Main controller managing the entire CFD simulation
- **Model**: Defines physical equations and boundary conditions
- **Polymesh**: Handles mesh operations and topology
- **Coefficients**: Matrix assembly for linear systems
- **Solvers**: Multiple algorithms for solving linear equations
- **Field**: Data structure for physical quantities with dimension checking

### 📊 Example Cases

#### Lid-Driven Cavity Flow
Classic benchmark case demonstrating incompressible flow in a square cavity with moving top wall.

#### Heat Transfer in Flange
Demonstrates heat conduction simulation with complex geometry and thermal boundary conditions.

### 🤝 Contributing

We welcome contributions in all forms:
- 🐛 Bug reports and fixes
- 📖 Documentation improvements
- 🚀 Performance optimizations
- ✨ New features and solvers
- 🧪 Test cases and validation

Please submit Pull Requests through GitHub.
**⭐ Star this repository if you find it useful for your CFD learning journey!**

### 📄 License

This project is open-source software under the [MIT License](LICENSE).

### 📧 Contact

- **Email**: yuzd17@tsinghua.org.cn
- **Issues**: Please use GitHub Issues for questions and suggestions
- **Discussions**: Welcome to join development discussions

---

## 中文版本

### 🌊 项目概述

**pyOpenFOAM** 是一个基于有限体积方法（FVM）的计算流体动力学（CFD）库，用于模拟流体流动和热传递现象。该库提供了一套完整的工具来处理网格拓扑关系、方程求解、边界条件设置和结果可视化。

该求解器库采用OpenFOAM网格格式文件，主要离散格式和求解方法参考了《The Finite Volume Method in Computational Fluid Dynamics》一书的原理。pyOpenFOAM目前完全基于Python平台开发，虽然计算性能有待提升，但该项目具有重要的教育意义，对于CFD从业者理解数值离散方法以及科研学术工作具有重要价值。

### ✨ 主要特性

- **多维网格支持**，兼容多种边界条件
- **多种线性求解器**：AMG（代数多重网格法）、PCG（预条件共轭梯度法）、ILU（不完全LU分解）、SOR（逐次超松弛迭代法）
- **稳态和瞬态模拟**能力
- **丰富的插值和残差计算**工具
- **自定义方程和系数**配置支持
- **量纲一致性管理**：实现物理量纲自动检查和验证系统，确保数学运算中的量纲关系正确
- **OpenFOAM兼容性**：原生支持OpenFOAM网格格式和算例结构

### 🚀 快速开始

#### 环境要求
- Python 3.8+（推荐：Python 3.10）
- NumPy、SciPy、Matplotlib

#### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/sherlockyzd/pyOpenFOAM.git
cd pyOpenFOAM
# 创建并激活conda环境
conda create --name pyOpenFOAM python=3.10 numpy matplotlib scipy
conda activate pyOpenFOAM
# 运行方腔流算例
cd example/cavity
python pyFVMScript.py
```
如果你想用PETCs, 输入
```bash
conda install petsc petsc4py
```
如果你想用cuda PETCs（gpu并行版）, 输入
```bash
nvidia-smi #check the cuda version
nvcc --version
conda install -c conda-forge petsc=*=*cuda* petsc4py
# For CUDA 12, run:    
conda install cuda-cudart cuda-version=12
# For CUDA 13, run:    
conda install cuda-cudart cuda-version=13
echo 'export PETSC_OPTIONS="-use_gpu_aware_mpi 0"' >> ~/.bashrc
source ~/.bashrc
# conda install cuda-cudart cuda-version=12 #cuda version=12.7
# echo 'export OMPI_MCA_opal_cuda_support=true' >> ~/.bashrc
# echo 'export UCX_MEMTYPE_CACHE=n' >> ~/.bashrc
# source ~/.bashrc
# pip install jax[cuda]
```


#### pip安装方式
```bash
pip install numpy scipy matplotlib
```

### 💻 基本使用

1. **初始化模拟区域**：
```python
from pyFVM.Region import Region

case_path = "/path/to/case"
cfd_case = Region(case_path)
```

2. **运行算例**：
```python
cfd_case.RunCase()
```

3. **结果可视化**：使用内置绘图工具或导出数据到ParaView

### 📁 项目结构

```
pyOpenFOAM/
├── src/                    # 源代码
│   ├── pyFVM/             # 核心FVM实现
│   │   ├── Region.py      # 主要仿真控制器
│   │   ├── Model.py       # 物理模型定义
│   │   ├── Polymesh.py    # 网格处理
│   │   ├── Solve.py       # 线性求解器
│   │   └── ...
│   └── cfdtool/           # 实用工具
├── example/               # 测试算例
│   ├── cavity/           # 方腔流算例
│   └── flange/           # 传热算例
└── test/                 # 单元测试
```

### 🔬 支持的物理现象

- **动量方程**（Navier-Stokes方程）
- **连续性方程**（质量守恒）
- **能量方程**（传热）
- **压力-速度耦合**（SIMPLE/PISO算法）

### 🛠 开发架构

求解器采用模块化架构：

- **Region**：主控制器，管理整个CFD仿真
- **Model**：定义物理方程和边界条件
- **Polymesh**：处理网格操作和拓扑关系
- **Coefficients**：线性系统的矩阵装配
- **Solvers**：多种线性方程求解算法
- **Field**：带量纲检查的物理量数据结构

### 📊 案例

#### 顶盖驱动流
openfoam中的一个经典基准案例（cavity），展示带移动顶盖方形空腔中的不可压缩流动.

#### 法兰盘换热
演示具有复杂几何形状和热边界条件的法兰盘热传导模拟。

### 🤝 贡献指南

我们欢迎任何形式的贡献：
- 🐛 错误报告和修复
- 📖 文档改进
- 🚀 性能优化
- ✨ 新功能和求解器
- 🧪 测试算例和验证

请通过GitHub提交Pull Requests。

### 📄 开源许可

本项目采用 [MIT许可证](LICENSE) 开源。

### 📧 联系方式

- **邮箱**：yuzd17@tsinghua.org.cn
- **问题反馈**：请使用GitHub Issues提问和建议
- **技术交流**：欢迎参与开发讨论

### 🎯 项目愿景

我们致力于创建一个**教育友好**且**功能完整**的CFD求解器，帮助学生和研究者：
- 深入理解有限体积方法的数值实现
- 学习CFD算法的底层原理
- 开发和验证新的数值方法
- 为更高性能的CFD工具奠定基础


---

**⭐ Star this repository if you find it useful for your CFD learning journey!**

*本项目正在持续开发中，欢迎加入我们一起完善这个教育性CFD工具！*