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
        self.MatrixFormat = Region.mesh.MatrixFormat # 默认格式
        self.NumberOfElements=int(Region.mesh.numberOfElements)
        self._sparse_matrix_structure_needs_update = True
        ## see ac, however this is for the previous timestep? Check this later when you know more. 
        # self.ac_old=np.zeros((self.NumberOfElements),dtype=np.float64)
        # self._init_csr_format(Region)
        # 根据格式初始化不同的数据结构
        self._initialize_matrix_structure(Region)

    # def setupCoefficients(self,**kwargs):
    #     """Setups empty arrays containing the coefficients (ac and bc) required to solve the system of equations
    #     """
    #     if len(kwargs)==0:
        ## (list of lists) identical to polyMesh.elementNeighbours. Provides a list where each index represents an element in the domain. Each index has an associated list which contains the elements for which is shares a face (i.e. the neighbouring elements).
        ## array of the boundary condition contributions to the flux term.
        self.bc=np.zeros((self.NumberOfElements),dtype=np.float64)
        self.dphi=np.zeros((self.NumberOfElements),dtype=np.float64)
        

    def _initialize_matrix_structure(self, Region):
        """Initialize the matrix structure based on the specified format."""
        """根据MatrixFormat初始化对应的数据结构"""
        self._A_sparse_needs_update = True
        # 控制是否总是使用稀疏矩阵（默认False，保持原有行为）
        # 原来只有压力泊松方程使用稀疏矩阵，动量方程和标量方程不转换
        self._sparse_always = getattr(Region, 'sparse_always', False)

        if self.MatrixFormat == 'acnb':
            self._init_acnb_format(Region)
            if self._sparse_always:
                self._init_csr_format(Region)
        elif self.MatrixFormat == 'ldu':
            self._init_ldu_format(Region)
            if self._sparse_always:
                self._init_csr_format(Region)
        elif self.MatrixFormat == 'csr':
            self._init_csr_format(Region)
        elif self.MatrixFormat == 'coo':
            self._init_coo_format(Region)
        else:
            raise ValueError(f"Unsupported MatrixFormat: {self.MatrixFormat}")

    def _init_ldu_format(self, Region):
        # Initialize the LDU matrix structure
        self._upperAddr=Region.mesh.lduUpperAddr
        self._lowerAddr=Region.mesh.lduLowerAddr
        self._facesAsOwner = Region.mesh.facesAsOwner
        self._facesAsNeighbour = Region.mesh.facesAsNeighbour
        self.Diag  = np.zeros((Region.mesh.numberOfElements), dtype=np.float64)
        self.Upper = np.zeros((Region.mesh.numberOfInteriorFaces), dtype=np.float64)
        self.Lower = np.zeros((Region.mesh.numberOfInteriorFaces), dtype=np.float64)
        
    def _init_coo_format(self, Region):
        # Initialize the COO matrix structure
        self._row = Region.mesh.cooRow
        self._col = Region.mesh.cooCol
        self._coofaceToRowIndex = Region.mesh.coofaceToRowIndex
        self._coodiagPositions = Region.mesh.coodiagPositions
        self.coodata = np.zeros(len(self._row), dtype=np.float64)
        self._sparse_matrix_structure_needs_update = False

    def _init_csr_format(self, Region):
        # NumberOfElements=self.NumberOfElements
        # 组装矩阵结构部分（indices 和 indptr）
        self._indptr = Region.mesh.csrindptr
        self._indices = Region.mesh.csrindices
        self._csrfaceToRowIndex=Region.mesh.csrfaceToRowIndex
        # self._diag_positions = self._indptr[:-1]  # 每行起始位置就是对角位置
        self.csrdata = np.zeros(len(self._indices), dtype=np.float64)
        self._sparse_matrix_structure_needs_update =False

    def _init_acnb_format(self, Region):
        """Initialize the matrix structure for the 'acnb' format."""
        ## array of cell-centered contribution to the flux term. These are constants and constant diffusion coefficients and therefore act as 'coefficients' in the algebraic equations. See p. 229 Moukalled.
        self._theCConn = Region.mesh.elementNeighbours
        self.ac=np.zeros((self.NumberOfElements),dtype=np.float64)#矩阵对角线元素
        # 使用NumPy对象数组，允许每个元素的邻居数不一样
        self.anb = [np.zeros(len(neighbors), dtype=np.float64) for neighbors in self._theCConn]

    def data_sparse_matrix_update(self):
        """Assembles the original matrix A from method To the sparse matrix csr format."""
        if self._A_sparse_needs_update:
            if self.MatrixFormat=='acnb':
                self._assemble_acnb_to_csr()
            elif self.MatrixFormat=='coo':
                self._assemble_coo_to_coo()
            elif self.MatrixFormat=='ldu':
                self._assemble_ldu_to_csr()
            elif self.MatrixFormat=='csr':
                self._assemble_csr_to_csr()
            else:
                raise ValueError(f"Unknown method")

        self._A_sparse_needs_update = False
        return self._A_sparse


    def cfdZeroCoefficients(self):
        # ==========================================================================
        #  Routine Description:
        #    This function zeros the coefficients
        # --------------------------------------------------------------------------
        # array of cell-centered contribution to the flux term. These are constants and constant diffusion coefficients and therefore act as 'coefficients' in the algebraic equations. See p. 229 Moukalled.
        # array of the boundary condition contributions to the flux term.
        self.bc.fill(0)
        self.dphi.fill(0)
        self._A_sparse_needs_update = True
        if self.MatrixFormat == 'acnb':
            self.ac.fill(0)
            # reset the anb list of lists
            for iElement in range(self.NumberOfElements):
                self.anb[iElement].fill(0)
        elif self.MatrixFormat == 'ldu':
            self.Diag.fill(0)
            # reset the lower and upper lists
            self.Lower.fill(0)
            self.Upper.fill(0)
        elif self.MatrixFormat == 'csr':
            self.csrdata.fill(0)
        elif self.MatrixFormat == 'coo':
            self.coodata.fill(0)

    def _assemble_csr_to_csr(self):
        """
        Assemble the sparse matrix A from ac, anb, and cconn.
        """
        # Implementation for assembling acnb to csr
        if not hasattr(self, '_A_sparse'):
            from scipy.sparse import csr_matrix
            # # Create the sparse matrix in COO format
            self._A_sparse = csr_matrix((self.csrdata, self._indices, self._indptr), shape=(self.NumberOfElements, self.NumberOfElements))
        else:
            # Update existing data array
            self._A_sparse.data = self.csrdata

    def _assemble_coo_to_coo(self):
        """
        Assemble the sparse matrix A from ac, anb, and cconn.
        Args:
            ac (ndarray): Diagonal elements of the matrix A.
            anb (list of lists): Off-diagonal neighbor elements of A.
            cconn (list of lists): Connectivity (indices of neighbors) for each row.

        Returns:
            A_sparse (scipy.sparse.csr_matrix): The assembled sparse matrix in CSR format.
        """
        numberOfElements = self.NumberOfElements
        if not hasattr(self, '_A_sparse'):
            from scipy.sparse import coo_matrix
            # # Create the sparse matrix in COO format
            self._A_sparse = coo_matrix((self.coodata, (self._row, self._col)), shape=(numberOfElements, numberOfElements))
            # Convert to CSR format for efficient arithmetic and solving
            # self._A_sparse = A_coo.tocsr()
        else:
            # Update existing data array
            self._A_sparse.data = self.coodata


    def _assemble_acnb_to_csr(self):
        """
        使用 Numpy 数组将 ac, anb 和 cconn 组装成 CSR 格式的稀疏矩阵。

        参数：
            ac (ndarray): 矩阵 A 的对角元素。
            anb (list of lists): 矩阵 A 的非对角（邻接）元素。
            cconn (list of lists): 每一行的邻接（邻居的索引）。

        返回：
            data (ndarray): 矩阵的非零值。
            indices (ndarray): 非零值对应的列索引。
            indptr (ndarray): 每一行在 data 和 indices 中的起始位置索引。
        """
        # Assemble data array
        # 组装数据部分（按行顺序）
        for i in range(self.NumberOfElements):
            start = self._indptr[i]
            self.csrdata[start] = self.ac[i]
            self.csrdata[start + 1:self._indptr[i+1]] = self.anb[i]

        if not hasattr(self, '_A_sparse'):
            from scipy.sparse import csr_matrix
            self._A_sparse = csr_matrix((self.csrdata, self._indices, self._indptr), shape=(self.NumberOfElements, self.NumberOfElements))
        else:
            # Update existing data array
            self._A_sparse.data = self.csrdata



    def _assemble_ldu_to_csr(self):
        """将LDU格式转换为CSR格式（利用现有CSR结构）"""
        # 直接使用现有的CSR结构
        # self.csrdata.fill(0.0)  # 清零数据
        # Step 1: 填充对角元素（每行第一个元素就是对角）
        # diag_positions = self._indptr[:-1]  # 每行起始位置就是对角位置
        self.csrdata[self._indptr[:-1]] = self.Diag
        # Step 2: 利用csrfaceToRowIndex填充非对角元素
        # numberOfInteriorFaces = len(self.Upper)
        # face_positions = self._csrfaceToRowIndex
        # csrfaceToRowIndex[f, 0]: neighbor行中owner列的位置
        # csrfaceToRowIndex[f, 1]: owner行中neighbor列的位置
        # neighbor_to_owner_positions = self._csrfaceToRowIndex[:, 0]  # lower值的位置
        # owner_to_neighbor_positions = self._csrfaceToRowIndex[:, 1]  # upper值的位置
        # 填充upper和lower值
        self.csrdata[self._csrfaceToRowIndex[:, 0]] = self.Lower
        self.csrdata[self._csrfaceToRowIndex[:, 1]] = self.Upper

        # Step 3: 创建或更新scipy CSR矩阵
        if not hasattr(self, '_A_sparse'):
            from scipy.sparse import csr_matrix
            self._A_sparse = csr_matrix(
                (self.csrdata, self._indices, self._indptr),
                shape=(self.NumberOfElements, self.NumberOfElements)
            )
        else:
            # 更新现有矩阵的数据
            self._A_sparse.data = self.csrdata

    def cfdComputeResidualsArray(self):
        """计算残差数组
        
        Args:
            force_sparse: 强制使用稀疏矩阵乘法
        """
        Adphi = self.theCoefficients_Matrix_multiplication(self.dphi)
        rc = self.bc - Adphi
        return rc
    
    def theCoefficients_Matrix_multiplication(self, x):
        """矩阵向量乘法（格式无关）
        Args:
            x: 输入向量
        """
        # 如果设置了_sparse_always=True，使用稀疏矩阵
        # 1. 核心优化策略
        # - 设置 sparse_always = True 让所有方程都使用scipy.sparse矩阵乘法
        # - 保持原有ACNB格式用于组装，转换为CSR格式进行计算
        # - 利用scipy库的高度优化C/Fortran实现
        # 2. 关键代码改动
        # - Region.py: self.sparse_always = True
        # - Coefficients.py:: 自动检测并初始化CSR结构，统一使用 self._A_sparse @ x 进行矩阵乘法
        # 3. 性能提升机制
        # - scipy.sparse优势: 使用高度优化的C/Fortran代码替代Python循环
        # - 向量化计算: @ 操作符调用BLAS库进行优化计算
        # - 内存效率: CSR格式的紧凑存储和高效访问模式
        # 4. 架构优势
        # - 向后兼容: 保留了所有原有计算方法
        # - 灵活控制: 通过sparse_always参数可选择优化策略
        # - 代码简洁: 统一的矩阵乘法接口
        if self._sparse_always:
            if self._A_sparse_needs_update or not hasattr(self, '_A_sparse'):
                self.data_sparse_matrix_update()
            return self._A_sparse @ x
        
        # 否则使用原有的格式特定乘法
        if self.MatrixFormat == 'ldu':
            return self._ldu_multiply(x)
        elif self.MatrixFormat == 'acnb':
            return self._acnb_multiply(x)
        elif self.MatrixFormat == 'csr':
            return self._csr_multiply(x)
        elif self.MatrixFormat == 'coo':
            return self._coo_multiply(x)
        else:
            raise ValueError(f"Unsupported MatrixFormat: {self.MatrixFormat}")

    def _ldu_multiply(self, x):
        """LDU格式的高效矩阵乘法"""
        y = self.ac * x
        np.add.at(y, self._lowerAddr, self.Upper * x[self._upperAddr])
        np.add.at(y, self._upperAddr, self.Lower * x[self._lowerAddr])
        return y

    def _acnb_multiply(self, x):
        """ACNB格式的高效矩阵乘法"""
        y = self.ac * x
        for i in range(self.NumberOfElements):
            y[i] += np.sum(self.anb[i] * x[self._theCConn[i]])
        return y

    def _csr_multiply(self, x):
        """CSR格式的高效矩阵乘法"""
        y = np.zeros(self.NumberOfElements, dtype=np.float64)
        for i in range(self.NumberOfElements):
            start = self._indptr[i]
            end = self._indptr[i + 1]
            y[i] = np.dot(self.csrdata[start:end], x[self._indices[start:end]])
        return y
    
    def _coo_multiply(self, x):
        """COO格式的高效矩阵乘法"""
        y = np.zeros(self.NumberOfElements, dtype=np.float64)
        np.add.at(y, self._row, self.coodata * x[self._col])
        return y

    def verify_matrix_properties(self):
        from scipy.sparse.linalg import norm
        # 检查对称性：计算 Frobenius 范数
        symmetry_error = norm(self._A_sparse - self._A_sparse.T, ord='fro')
        if symmetry_error > 1e-5:
            raise ValueError(f"矩阵 A 不是对称的，对称性误差为 {symmetry_error}")

        # 检查正定性
        if np.any(self._A_sparse.diagonal() <= 0):
            raise ValueError("矩阵 A 不是正定的")

