
# pyOpenFOAM: Python Finite Volume Method Solver Like OpenFoam

pyOpenFOAM 是一个基于有限体积方法（FVM）的计算流体动力学（CFD）库，用于模拟流体流动和热传递。该库提供了一套完整的工具来处理网格拓扑关系、方程求解、边界条件设置和结果可视化。该求解器库采用OpenFOAM网格格式文件，主要离散格式和求解也参考了书籍《The Finite Volume Method in Computational Fluid Dynamics》。pyOpenFOAM目前完全基于Python平台开发，尽管也用到了Numpy库，但计算性能依然受限，亟待大家的共同开发和加入来提高程序的性能，但该代码库依然具有教育意义，对于从事CFD行业理解CFD数值离散以及从事科研学术依然具有重要意义。

## 主要特性

- 支持多维网格和多种边界条件
- 多种求解器选项（待验证准确性），包括AMG(代数多重网格法)、PCG(带预分解器共轭梯度法)、ILU(不完全LU分解)和SOR(高斯赛德尔迭代法)
- 稳态和瞬态模拟支持
- 丰富的插值和残差计算工具
- 支持自定义方程和系数设置
- 本项目实现了一个量纲管理机制，用于在计算流体力学（CFD）模拟过程中自动检查和维护物理量的量纲一致性。通过集成该机制，可以确保各种物理运算（如加减乘除）中的量纲关系正确，提升代码的健壮性和可靠性。

## 安装

可以通过以下方式安装 pyOpenFOAM：

```bash
git clone https://github.com/sherlockyzd/pyOpenFOAM.git
cd pyOpenFOAM
```

## 使用方法

1. **初始化模拟区域**: 创建 `Region` 对象，设置案例路径。

```python
from pyFVM.Region import Region

case_path = "/path/to/case"
cfd_case = Region(case_path)
```

2. **运行案例**: 调用 `RunCase` 方法启动模拟。

```python
cfd_case.RunCase()
```

3. **结果可视化**: 使用内置的可视化工具或导出数据到ParaView。

## 文件结构及调用关系
```
|—— 0
|    |—— p
|    |—— T
|    |—— U
|—— constant
|    |—— g
|    |—— polyMesh
|        |—— boundary
|        |—— faces
|        |—— neighbour
|        |—— owner
|        |—— points
|    |—— transportProperties
|—— convergence
|    |—— convergenceUp.out
|—— hola.foam
|—— pyFVM
|    |—— Assemble.py
|    |—— cfdGetTools.py
|    |—— cfdPlot.py
|    |—— Coefficients.py
|    |—— Equation.py
|    |—— Field.py
|    |—— Fluxes.py
|    |—— FoamDictionaries.py
|    |—— Gradient.py
|    |—— Interpolate.py
|    |—— IO.py
|    |—— Math.py
|    |—— Model.py
|    |—— Polymesh.py
|    |—— Region.py
|    |—— Solve.py
|    |—— Time.py
|—— pyFVMScript.py
|—— system
|    |—— blockMeshDict
|    |—— controlDict
|    |—— fvSchemes
|    |—— fvSolution
|—— test.py
```
### 1. pyFVMScript.py
- **角色**: 主执行脚本。
- **功能**: 创建 `Region` 实例并运行案例。
- **调用**:
  - `Region.Region(os.getcwd())`: 创建区域实例。
  - `cfdCase.RunCase()`: 运行案例。

### 2. Region.py
- **角色**: 模拟区域类。
- **功能**: 管理整个CFD模拟案例。
- **调用**:
  - `Polymesh.Polymesh(self)`: 创建网格实例。
  - `FoamDictionaries.FoamDictionaries(self)`: 创建字典实例。
  - `Model.Model(self)`: 创建模型实例。
  - `Time.Time(self)`: 创建时间实例。
  - `Coefficients.Coefficients(self)`: 创建系数实例。
  - `Fluxes.Fluxes(self)`: 创建通量实例。
  - `Assemble.Assemble(self, iTerm)`: 创建组装实例。
  - `Solve.cfdSolveEquation(self, theEquationName, iComponent)`: 求解方程。

### 3. Polymesh.py
- **角色**: 网格类。
- **功能**: 处理网格相关操作。
- **调用**:
  - `self.cfdReadPointsFile()`: 读取点文件。
  - `self.cfdReadFacesFile()`: 读取面文件。
  - `self.cfdReadOwnerFile()`: 读取所有者文件。
  - `self.cfdReadNeighbourFile()`: 读取邻居文件。
  - `self.cfdReadBoundaryFile()`: 读取边界文件。

### 4. FoamDictionaries.py
- **角色**: 字典类。
- **功能**: 读取和操作OpenFOAM字典文件。
- **调用**:
  - `self.cfdReadControlDictFile()`: 读取控制字典文件。
  - `self.cfdReadFvSchemesFile()`: 读取有限体积方案文件。
  - `self.cfdReadFvSolutionFile()`: 读取有限体积解决方案文件。
  - `self.cfdReadGravity()`: 读取重力文件。
  - `self.cfdReadTurbulenceProperties()`: 读取湍流属性文件。

### 5. Model.py
- **角色**: 模型类。
- **功能**: 定义CFD模拟的模型。
- **调用**:
  - `Equation.Equation(fieldName)`: 创建方程实例。
  - `self.DefineMomentumEquation()`: 定义动量方程。
  - `self.DefineContinuityEquation()`: 定义连续性方程。
  - `self.DefineEnergyEquation()`: 定义能量方程。

### 6. Equation.py
- **角色**: 方程类。
- **功能**: 表示CFD模拟中的一个方程。
- **调用**:
  - `self.initializeResiduals()`: 初始化残差。
  - `self.setTerms(terms)`: 设置方程项。

### 7. Coefficients.py
- **角色**: 系数类。
- **功能**: 设置求解方程组所需的系数。
- **调用**:
  - `self.setupCoefficients()`: 设置系数数组。

### 8. Fluxes.py
- **角色**: 通量类。
- **功能**: 管理CFD模拟中的通量计算。
- **调用**:
  - `self.cfdZeroCoefficients()`: 将系数数组置零。

### 9. Assemble.py
- **角色**: 组装类。
- **功能**: 组装CFD模拟中的方程。
- **调用**:
  - `self.cfdAssembleEquation()`: 组装方程。

### 10. Solve.py
- **角色**: 求解器类。
- **功能**: 求解方程。
- **调用**:
  - `self.cfdSolveEquation()`: 求解方程。

### 11. Field.py
- **角色**: 场类。
- **功能**: 创建和初始化CFD场数据结构。
- **调用**:
  - `self.cfdUpdateScale()`: 更新场的比例尺。

### 12. Gradient.py
- **角色**: 梯度类。
- **功能**: 计算指定场的梯度。
- **调用**:
  - `self.cfdComputeGradientGaussLinear0()`: 使用高斯线性方法计算梯度。

### 13. cfdGetTools.py
- **角色**: 工具函数类。
- **功能**: 提供辅助函数，如插值、残差计算等。

### 14. Interpolate.py
- **角色**: 插值函数类。
- **功能**: 提供插值函数，用于场数据的插值计算。

### 15. Math.py
- **角色**: 数学函数类。
- **功能**: 提供数学工具，如向量运算、单位向量计算等。

### 16. IO.py
- **角色**: 输入输出类。
- **功能**: 提供输入输出工具，用于文件读写和错误处理。

## 贡献

我们欢迎任何形式的贡献，包括代码、文档、测试和反馈。请通过GitHub提交Pull Requests。

## 许可

pyFVM 是开源软件，采用 [MIT 许可证](LICENSE)。