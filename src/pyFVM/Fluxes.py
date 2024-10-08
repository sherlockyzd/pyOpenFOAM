import numpy as np
from cfdtool.quantities import Quantity as Q_
import cfdtool.dimensions as dm

class Fluxes():
    
    def __init__(self,Region):
        '''
        这段Python代码定义了一个名为 `Fluxes` 的类，它用于在计算流体动力学（CFD）模拟中设置和管理通量（fluxes）。以下是对类的构造器和初始化的通量属性的详细解释：
        1. **类定义**：
        - `class Fluxes():` 定义了一个名为 `Fluxes` 的类。

        2. **构造器**：
        - `def __init__(self, Region):` 构造器接收一个参数 `Region`，它是一个包含模拟区域信息的对象。

        3. **初始化属性**：
        - 构造器中初始化了一系列与通量相关的属性。这些属性用于存储面通量和体积通量的系数和值。

        4. **面通量属性**：
        - `self.FluxCf`: 面通量的线性化系数，用于单元格C（感兴趣单元格）。
        - `self.FluxFf`: 面通量的线性化系数，用于邻近单元格。
        - `self.FluxVf`: 面通量的非线性系数。
        - `self.FluxTf`: 总面通量，等于 `FluxCf * phiC + FluxFf * phiF + FluxVf`。

        5. **体积通量属性**：
        - `self.FluxC`: 体积通量的线性化系数。
        - `self.FluxV`: 体积通量，等于源值乘以单元格体积（`Q_C^phi * Vc`）。
        - `self.FluxT`: 总体积通量。
        - `self.FluxC_old`: 上一时间步的体积通量。

        6. **初始化数组大小**：
        - 所有通量数组的大小都是基于网格的面数量 `theNumberOfFaces` 和单元格数量 `theNumberOfElements`。

        7. **注释的方法**：
        - 代码中注释掉了 `setupFluxes` 方法，这个方法可能被用于更复杂的初始化逻辑，但在当前构造器中并未使用。

        ### 注意事项：
        - 类的构造器在类单独的方法中实现。
        - 构造器中使用 `Region.mesh.numberOfFaces` 和 `Region.mesh.numberOfElements` 获取网格的面和单元格的数量。
        - 所有通量数组都初始化为全零数组，使用 `np.zeros()` 创建。
        - 代码中的注释提供了对每个通量数组用途的说明，有助于理解每个属性的预期用途。

        `Fluxes` 类是CFD模拟中用于管理通量计算的关键组件，它提供了一种机制来存储和操作面通量和体积通量的系数，这些是求解流体动力学方程时的重要数据。通过这种方式，可以方便地访问和更新通量信息，以实现模拟的数值求解。
        '''
        # self.region=Region
        self.setupFluxes(Region)

    def setupFluxes(self,Region,**kwargs):
        theNumberOfFaces=Region.mesh.numberOfFaces
        theNumberOfElements=Region.mesh.numberOfElements
        self.FluxCf = {}  # 面通量的线性化系数，用于单元格 C（感兴趣单元格）
        self.FluxFf = {}  # 面通量的线性化系数，用于邻近单元格
        self.FluxVf = {}  # 面通量的非线性系数
        self.FluxTf = {}  # 总面通量，等于 FluxCf * phiC + FluxFf * phiF + FluxVf

        self.FluxC = {}   # 体积通量的线性化系数
        self.FluxV = {}   # 体积通量，等于源值乘以单元格体积（Q_C^phi * Vc）
        self.FluxT = {}   # 总体积通量
        self.FluxC_old = {}  # 上一时间步的体积通量
        # 为每个方程初始化通量
        for equation_name in Region.model.equations:
            CoffDim = Region.model.equations[equation_name].CoffDim
            Dim=Region.fluid[equation_name].phi.dimension*CoffDim
            #值保存在面心上
            #face fluxes
            # face flux linearization coefficients for cell C (cell of interest)
            self.FluxCf[equation_name]=Q_(np.zeros((theNumberOfFaces),dtype=float),CoffDim)
            # face flux linearization coefficients for neighbouring cell
            self.FluxFf[equation_name]=Q_(np.zeros((theNumberOfFaces),dtype=float),CoffDim)
            # non-linear face coefficients 
            self.FluxVf[equation_name]=Q_(np.zeros((theNumberOfFaces),dtype=float),Dim)
            # total face flux (equal to FluxCf*phiC+FluxFf*phiF+FluxVf)
            self.FluxTf[equation_name]=Q_(np.zeros((theNumberOfFaces),dtype=float),Dim)

            #值保存在体心上
            #Volume fluxes (treated as source terms)
            self.FluxC[equation_name]=Q_(np.zeros((theNumberOfElements),dtype=float),CoffDim)
            # volume fluxes from previous time step
            self.FluxC_old[equation_name]=Q_(np.zeros((theNumberOfElements),dtype=float),CoffDim)
            # volume flux equal to source value times cell volume (Q_{C}^{phi} * Vc)
            self.FluxV[equation_name]=Q_(np.zeros((theNumberOfElements),dtype=float),Dim)
            self.FluxT[equation_name]=Q_(np.zeros((theNumberOfElements),dtype=float),Dim)

        print('fluxes success!')
    
    def cfdZeroElementFLUXCoefficients(self,equation_name):
        # print('Inside cfdZeroElementFLUXCoefficients')
        self.FluxC[equation_name].value.fill(0)
        self.FluxV[equation_name].value.fill(0)
        self.FluxT[equation_name].value.fill(0)
        self.FluxC_old[equation_name].value.fill(0)

    def cfdZeroFaceFLUXCoefficients(self,equation_name):
        # print('Inside cfdZeroFaceFLUXCoefficients')
        self.FluxCf[equation_name].value.fill(0)
        self.FluxVf[equation_name].value.fill(0)
        self.FluxTf[equation_name].value.fill(0)
        self.FluxFf[equation_name].value.fill(0)




