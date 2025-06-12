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
            
        ## (list of lists) identical to polyMesh.elementNeighbours. Provides a list where each index represents an element in the domain. Each index has an associated list which contains the elements for which is shares a face (i.e. the neighbouring elements).
        self.theCConn = Region.mesh.elementNeighbours
        
        ## array containing the number of neighbouring elements for each element in the domain
        self.theCSize = np.array([int(len(neighbours)) for neighbours in self.theCConn],dtype=np.int32)  # 强制转换为整数

        theNumberOfElements=int(len(self.theCConn))

        self.NumberOfElements=theNumberOfElements
        
        ## array of cell-centered contribution to the flux term. These are constants and constant diffusion coefficients and therefore act as 'coefficients' in the algebraic equations. See p. 229 Moukalled.
        self.ac=np.zeros((theNumberOfElements),dtype=np.float64)
        
        ## see ac, however this is for the previous timestep? Check this later when you know more. 
        self.ac_old=np.zeros((theNumberOfElements),dtype=np.float64)
        
        ## array of the boundary condition contributions to the flux term.
        self.bc=np.zeros((theNumberOfElements),dtype=np.float64)

        # 使用NumPy对象数组，允许每个元素的邻居数不一样
        self.anb = [np.zeros(len(neighbors), dtype=np.float64) for neighbors in self.theCConn]
        # self.anb = np.empty(theNumberOfElements, dtype=object)
        # for iElement in range(theNumberOfElements):
        #     # easiest way to make a list of zeros of defined length ...
        #     self.anb[iElement] = np.zeros(int(self.theCSize[iElement]),dtype=float)
        
        self.dphi=np.zeros((theNumberOfElements),dtype=np.float64)
        self._A_sparse_needs_update = True
        # self.dc=np.zeros((theNumberOfElements),dtype=float)
        # self.rc=np.zeros((theNumberOfElements),dtype=float)

    def cfdZeroCoefficients(self):
        # ==========================================================================
        #  Routine Description:
        #    This function zeros the coefficients
        # --------------------------------------------------------------------------

        # array of cell-centered contribution to the flux term. These are constants and constant diffusion coefficients and therefore act as 'coefficients' in the algebraic equations. See p. 229 Moukalled.
        self.ac.fill(0)
        self.ac_old.fill(0)
        # array of the boundary condition contributions to the flux term.
        self.bc.fill(0)
        # reset the anb list of lists
        for iElement in range(self.NumberOfElements):
            self.anb[iElement].fill(0)
        self.dphi.fill(0)
        self._A_sparse_needs_update = True
        # self.dc.fill(0)
        # self.rc.fill(0)

    def assemble_sparse_matrix_coo(self):
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
        if not hasattr(self, '_coo_structure'):
            # row_indices = []
            # col_indices = []
            # # Add diagonal elements
            # for i in range(numberOfElements):
            #     row_indices.append(i)
            #     col_indices.append(i)
            # # Add off-diagonal elements
            # for i in range(numberOfElements):
            #     neighbors = self.theCConn[i]
            #     anb_values = self.anb[i]
            #     for j_, j in enumerate(neighbors):
            #         row_indices.append(i)
            #         col_indices.append(j)
            # self._coo_row_indices=np.array(row_indices, dtype=np.int32)
            # self._coo_col_indices=np.array(col_indices, dtype=np.int32)

            # Precompute the row and column indices
            # Diagonal indices
            diag_indices = np.arange(numberOfElements, dtype=np.int32)

            # Off-diagonal indices
            off_diag_row_indices = []
            off_diag_col_indices = []
            for i in range(numberOfElements):
                neighbors = self.theCConn[i]
                off_diag_row_indices.extend([i] * len(neighbors))
                off_diag_col_indices.extend(neighbors)

            # Combine indices
            self._coo_row_indices = np.concatenate([diag_indices, np.array(off_diag_row_indices, dtype=np.int32)])
            self._coo_col_indices = np.concatenate([diag_indices, np.array(off_diag_col_indices, dtype=np.int32)])
            self._coo_structure = True

        # data = []
        # # Add diagonal elements
        # for i in range(numberOfElements):
        #     data.append(self.ac[i])
        # # Add off-diagonal elements
        # for i in range(numberOfElements):
        #     neighbors = self.theCConn[i]
        #     anb_values = self.anb[i]
        #     for j_, j in enumerate(neighbors):
        #         data.append(anb_values[j_])
        # data=np.array(data, dtype=np.float64)
        # Assemble data array
        # Diagonal data
        diag_data = self.ac.astype(np.float64)
        # Off-diagonal data
        off_diag_data = np.concatenate(self.anb).astype(np.float64)
        # Combine data
        data = np.concatenate([diag_data, off_diag_data])

        if not hasattr(self, '_A_sparse'):
            from scipy.sparse import coo_matrix
            # Create the sparse matrix in COO format
            A_coo = coo_matrix((data, (self._coo_row_indices, self._coo_col_indices)), shape=(numberOfElements, numberOfElements))
            # Convert to CSR format for efficient arithmetic and solving
            self._A_sparse = A_coo.tocsr()
        else:
            # Update existing data array
            self._A_sparse.data[:numberOfElements] = diag_data
            self._A_sparse.data[numberOfElements:] = off_diag_data
        self._A_sparse_needs_update = False


    def assemble_sparse_matrix_csr(self):
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
        NumberOfElements=self.NumberOfElements
        if not hasattr(self, '_csr_structure'):
            # 组装矩阵结构部分（indices 和 indptr）
            # indices = []
            # indptr = [0]
            # for i in range(self.NumberOfElements):
            #     # 添加对角线索引
            #     indices.append(i)
            #     # 添加非对角线索引
            #     indices.extend(self.theCConn[i])
            #     indptr.append(len(indices))
            # self._indices = np.array(indices, dtype=np.int32)
            # self._indptr = np.array(indptr, dtype=np.int32)
            indptr = np.zeros(NumberOfElements + 1, dtype=np.int32)
            indices = []
            for i in range(NumberOfElements):
                neighbors = self.theCConn[i]
                # Diagonal and off-diagonal indices
                row_indices = [i] + neighbors
                indices.extend(row_indices)
                indptr[i + 1] = indptr[i] + len(row_indices)
            self._indices = np.array(indices, dtype=np.int32)
            self._indptr = indptr
            self._data = np.zeros(len(self._indices), dtype=np.float64)
            self._csr_structure = True
        
        # Assemble data array
        # 组装数据部分（按行顺序）
        for i in range(NumberOfElements):
            start = self._indptr[i]
            self._data[start] = self.ac[i]
            self._data[start + 1:start + 1 + len(self.anb[i])] = self.anb[i]


        if not hasattr(self, '_A_sparse'):
            from scipy.sparse import csr_matrix
            self._A_sparse = csr_matrix((self._data, self._indices, self._indptr), shape=(NumberOfElements, NumberOfElements))
        else:
            # Update existing data array
            self._A_sparse.data = self._data
        self._A_sparse_needs_update = False



    def assemble_sparse_matrix_lil(self):
        NumberOfElements = self.NumberOfElements
        from scipy.sparse import lil_matrix
        
        A = lil_matrix((NumberOfElements, NumberOfElements), dtype=np.float64)
        for i in range(NumberOfElements):
            A[i, i] = self.ac[i]
            for j, neighbor in enumerate(self.theCConn[i]):
                A[i, neighbor] = self.anb[i][j]
        self._A_sparse = A.tocsr()
        self._A_sparse_needs_update = False
        return self._A_sparse


    def cfdComputeResidualsArray(self):
        Adphi=self.theCoefficients_Matrix_multiplication(self.dphi)
        rc= self.bc-Adphi
        return rc
    
    def theCoefficients_Matrix_multiplication(self,d):
        # if hasattr(self, '_A_sparse') and not self._A_sparse_needs_update:
        #     Ad = self._A_sparse @ d
        # else:
        #     Ad = np.zeros_like(d)
        #     for iElement in range(self.NumberOfElements):
        #             Ad[iElement] = self.ac[iElement] * d[iElement]
        #             for iLocalNeighbour,neighbor in enumerate(self.theCConn[iElement]):
        #                 Ad[iElement] += self.anb[iElement][iLocalNeighbour] * d[neighbor]
        return self.theCoefficients_sparse_multiplication(d)

    def theCoefficients_sparse_multiplication(self,d):
        if not hasattr(self, '_A_sparse') or self._A_sparse_needs_update:
            self.assemble_sparse_matrix()
        return self._A_sparse @ d
    
    def assemble_sparse_matrix(self,method='csr'):
        if not hasattr(self, '_A_sparse') or self._A_sparse_needs_update:
            if method=='csr':
                self.assemble_sparse_matrix_csr()
            elif method=='coo':
                self.assemble_sparse_matrix_coo()
            else:
                raise ValueError(f"Unknown method {method}")
        return self._A_sparse
    
    

    

    def verify_matrix_properties(self):
        from scipy.sparse.linalg import norm
        # 检查对称性：计算 Frobenius 范数
        symmetry_error = norm(self._A_sparse - self._A_sparse.T, ord='fro')
        if symmetry_error > 1e-6:
            raise ValueError(f"矩阵 A 不是对称的，对称性误差为 {symmetry_error}")

        # 检查正定性
        if np.any(self._A_sparse.diagonal() <= 0):
            raise ValueError("矩阵 A 不是正定的")

