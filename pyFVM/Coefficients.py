import numpy as np

class Coefficients():
    
    def __init__(self,Region):
        '''
        这段Python代码定义了一个名为`Coefficients`的类，它用于设置计算流体动力学（CFD）模拟中求解方程组所需的系数。以下是对类的构造器和`setupCoefficients`方法的详细解释：

        1. **类定义**：
        - `class Coefficients():` 定义了一个名为`Coefficients`的类。

        2. **构造器**：
        - `def __init__(self, Region):` 构造器接收一个参数`Region`，它是一个包含模拟区域信息的对象。

        3. **实例属性**：
        - `self.region`: 存储传入的`Region`实例，作为类的局部属性。

        4. **初始化系数方法**：
        - `def setupCoefficients(self, **kwargs):` 定义了一个实例方法，用于设置求解方程所需的系数数组。接收任意数量的关键字参数。

        5. **处理关键字参数**：
        - 如果没有提供关键字参数（`len(kwargs) == 0`），则使用`self.region.mesh.elementNeighbours`作为连接性信息。

        6. **设置连接性数组**：
        - `self.theCConn`: 存储网格元素的邻居元素列表，与`polyMesh.elementNeighbours`结构相同。

        7. **计算每个元素的邻居数量**：
        - `self.theCSize`: 一个数组，包含每个元素的邻居元素数量。

        8. **初始化系数数组**：
        - `self.ac`: 一个数组，包含每个单元格中心对通量项的贡献，通常是常数和恒定的扩散系数。
        - `self.ac_old`: 与`self.ac`类似，但用于上一时间步。
        - `self.bc`: 一个数组，包含边界条件对通量项的贡献。

        9. **设置邻接元素系数**：
        - `self.anb`: 一个列表的列表，为每个元素的每个邻居元素设置系数。

        10. **初始化其他数组**：
            - `self.dc`: 一个数组，可能用于存储与对流或扩散相关的系数。
            - `self.rc`: 一个数组，可能用于存储源项或其他常数项。
            - `self.dphi`: 一个数组，可能用于存储场变量的增量。

        ### 注意事项：
        - 类的构造器中调用了`setupCoefficients`方法，但没有提供任何关键字参数，因此方法会使用`Region`实例的网格连接性信息。
        - `self.theCConn`和`self.theCSize`基于网格的拓扑结构进行初始化，这对于后续计算邻居元素的贡献是必要的。
        - 代码中的注释提供了一些额外信息，例如`self.ac`和`self.ac_old`的区别，以及可能需要进一步确认的点。
        - `self.anb`的初始化使用了列表推导式，为每个元素创建了一个长度等于其邻居数量的零列表。

        `Coefficients`类是CFD模拟中用于管理方程求解过程中的系数的一个辅助类。通过这种方式，可以方便地访问和操作与网格拓扑结构相关的系数。
        '''
        
        ## local attribute of simulation's region instance
        # self.region=Region
        # self.setupCoefficients()

    # def setupCoefficients(self,**kwargs):
    #     """Setups empty arrays containing the coefficients (ac and bc) required to solve the system of equations
    #     """
    #     if len(kwargs)==0:
            
            ## (list of lists) identical to polyMesh.elementNeighbours. Provides a list where each index represents an element in the domain. Each index has an associated list which contains the elements for which is shares a face (i.e. the neighouring elements).
        self.theCConn = Region.mesh.elementNeighbours
        
        ## array containing the number of neighbouring elements for each element in the domain
        self.theCSize = np.zeros((len(self.theCConn)))
        
        for iElement,value in enumerate(self.theCConn):
            self.theCSize[iElement]=len(self.theCConn[iElement])
               
        theNumberOfElements=len(self.theCConn)

        self.NumberOfElements=theNumberOfElements
        
        ## array of cell-centered contribution to the flux term. These are constants and constant diffusion coefficients and therefore act as 'coefficients' in the algebraic equations. See p. 229 Moukalled.
        self.ac=np.zeros((theNumberOfElements))
        
        ## see ac, however this is for the previous timestep? Check this later when you know more. 
        self.ac_old=np.zeros((theNumberOfElements))
        
        ## array of the boundary condition contributions to the flux term.
        self.bc=np.zeros((theNumberOfElements))
        
        self.anb=[]
        
        for iElement in range(theNumberOfElements):
            
            #easiest way to make a list of zeros of defined length ...
            listofzeros = [0]*int(self.theCSize[iElement])
            self.anb.append(listofzeros) 
        # self.dc=np.zeros((theNumberOfElements))
        # self.rc=np.zeros((theNumberOfElements))
        self.dphi=np.zeros((theNumberOfElements))

    def cfdZeroCoefficients(self):
# %==========================================================================
# % Routine Description:
# %   This function zeros the coefficients
# %--------------------------------------------------------------------------
# % Get info
        # theNumberOfElements = self.NumberOfElements

         ## array of cell-centered contribution to the flux term. These are constants and constant diffusion coefficients and therefore act as 'coefficients' in the algebraic equations. See p. 229 Moukalled.
        self.ac.fill(0)
        ## see ac, however this is for the previous timestep? Check this later when you know more. 
        self.ac_old.fill(0)
        ## array of the boundary condition contributions to the flux term.
        self.bc.fill(0)
        ## reset the anb list of lists
        for iElement in range(self.NumberOfElements):
            self.anb[iElement] = [0] * int(self.theCSize[iElement])
        self.dphi.fill(0)