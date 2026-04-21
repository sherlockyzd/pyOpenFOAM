#!/usr/bin/env python3
"""
PETSc求解器 - 精简高效版本
"""
import cfdtool.Math as mth
import numpy as np
import cfdtool.IO as io

try:
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    print("警告: PETSc不可用")

_global_solver_cache = {}

class PETScSolver:
    """PETSc线性求解器 - 支持CPU/GPU
    PETSc对象包括：
  - Mat: 稀疏矩阵 (CPU: AIJ, GPU: AIJCUSPARSE)
  - Vec: 向量 (CPU: Seq, GPU: CUDA)
  - KSP: Krylov子空间求解器
  - PC: 预处理器
    """
    def __init__(self, use_gpu=False):
        if not PETSC_AVAILABLE:
            raise ImportError("PETSc未安装")

        self.use_gpu = use_gpu
        self.A_petsc = None
        self.b_petsc = None
        self.x_petsc = None
        self.ksp = None
        self.cached_size = None
        
        if use_gpu:
            self._check_gpu()
        
        print(f"PETSc求解器: {'GPU' if use_gpu else 'CPU'}")
    
    def _check_gpu(self):
        """检查GPU"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, timeout=3)
            if result.returncode != 0:
                print("GPU检测失败，使用CPU")
                self.use_gpu = False
        except:
            print("GPU检测失败，使用CPU")
            self.use_gpu = False
    
    def setup_matrix(self, values, indptr, indices, size):
        """设置矩阵"""
        if self.cached_size != size or self.A_petsc is None:
            if self.A_petsc:
                self.A_petsc.destroy()
            
            nnz = max(10, len(values) // size + 5)
            
            if self.use_gpu:
                self.A_petsc = PETSc.Mat().create()
                self.A_petsc.setSizes([size, size])
                self.A_petsc.setType(PETSc.Mat.Type.AIJCUSPARSE)
                self.A_petsc.setPreallocationNNZ(nnz)
            else:
                self.A_petsc = PETSc.Mat().createAIJ([size, size], nnz=nnz)
            
            self.A_petsc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            self.A_petsc.setUp()
            self.cached_size = size
            print(f"创建矩阵: {size}x{size}")
        else:
            print("矩阵已存在，跳过重新分配")
        
        # 更新矩阵值
        self.A_petsc.setValuesCSR(indptr, indices, values)
        # self.A_petsc.zeroEntries()
        # for i in range(len(indptr) - 1):
        #     start, end = indptr[i], indptr[i + 1]
        #     if end > start:
        #         self.A_petsc.setValues(i, indices[start:end], values[start:end])
        
        self.A_petsc.assemblyBegin()
        self.A_petsc.assemblyEnd()
    
    def setup_vectors(self, theCoefficients):
        """设置向量"""
        n = theCoefficients.NumberOfElements
        
        if self.b_petsc is None or self.b_petsc.getSize() != n:
            if self.b_petsc:
                self.b_petsc.destroy()
                self.x_petsc.destroy()
            
            if self.use_gpu:
                self.b_petsc = PETSc.Vec().create()
                self.b_petsc.setSizes(n)
                self.b_petsc.setType(PETSc.Vec.Type.CUDA)
                self.b_petsc.setUp()
                
                self.x_petsc = PETSc.Vec().create()
                self.x_petsc.setSizes(n)
                self.x_petsc.setType(PETSc.Vec.Type.CUDA)
                self.x_petsc.setUp()
            else:
                self.b_petsc = PETSc.Vec().createSeq(n)
                self.x_petsc = PETSc.Vec().createSeq(n)
        
        # 更新向量值
        b_data = getattr(theCoefficients, 'bc', np.zeros(n))
        x_data = getattr(theCoefficients, 'dphi', np.zeros(n))
        
        self.b_petsc.setArray(b_data)
        self.x_petsc.setArray(x_data)
    
    def solve(self, solver_type='gmres', preconditioner='ilu', rtol=1e-6, atol=1e-10, max_iter=1000):
        """求解"""
        if self.ksp is None:
            self.ksp = PETSc.KSP().create()
        
        self.ksp.setOperators(self.A_petsc)
        
        # 设置求解器类型
        if solver_type == 'gmres':
            self.ksp.setType(PETSc.KSP.Type.GMRES)
        elif solver_type == 'cg':
            self.ksp.setType(PETSc.KSP.Type.CG)
        elif solver_type == 'bicgstab':
            self.ksp.setType(PETSc.KSP.Type.BCGS)
        else:
            self.ksp.setType(PETSc.KSP.Type.GMRES)
        
        # 设置预处理器
        pc = self.ksp.getPC()
        if self.use_gpu and preconditioner == 'ilu':
            pc.setType(PETSc.PC.Type.JACOBI)  # GPU上ILU不稳定
        elif preconditioner == 'ilu':
            pc.setType(PETSc.PC.Type.ILU)
        elif preconditioner == 'jacobi':
            pc.setType(PETSc.PC.Type.JACOBI)
        elif preconditioner == 'lu':
            pc.setType(PETSc.PC.Type.LU)
        else:
            pc.setType(PETSc.PC.Type.ILU)
        
        # 设置容差
        self.ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_iter)
        self.ksp.setFromOptions()
        
        # 求解
        self.ksp.solve(self.b_petsc, self.x_petsc)
        
        # 检查收敛
        reason = self.ksp.getConvergedReason()
        iterations = self.ksp.getIterationNumber()
        residual = self.ksp.getResidualNorm()
        
        if reason > 0:
            print(f"求解成功: {iterations}次迭代, 残差={residual:.2e}")
            return True, iterations, residual
        else:
            print(f"求解失败: reason={reason}")
            return False, iterations, residual
    
    def get_solution(self):
        """获取解"""
        return self.x_petsc.getArray().copy()


def _get_csr_data(theCoefficients):
    """获取CSR数据"""
    if theCoefficients.MatrixFormat in ['csr', 'ldu', 'acnb']:
        return theCoefficients.csrdata, theCoefficients._indptr, theCoefficients._indices
    elif theCoefficients.MatrixFormat == 'coo':
        A_csr = theCoefficients._A_sparse.tocsr()
        return A_csr.data, A_csr.indptr, A_csr.indices
    else:
        raise ValueError(f"不支持的矩阵格式: {theCoefficients.MatrixFormat}")


def cfdSolvePETSc(theCoefficients, maxIter=1000, tolerance=1e-6, relTol=0.1, 
                  solver_type='gmres', preconditioner='ilu', use_gpu=False):
    """PETSc求解接口"""
    
    if not PETSC_AVAILABLE:
        io.cfdError("PETSc未安装，无法求解")

    theCoefficients.data_sparse_matrix_update()
    
    # 获取CSR数据
    values, indptr, indices = _get_csr_data(theCoefficients)
    
    # 计算初始残差
    residualsArray = theCoefficients.cfdComputeResidualsArray()
    initRes = mth.cfdResidual(residualsArray)
    
    if initRes < tolerance or maxIter == 0:
        return initRes, initRes
    
    # 获取求解器
    cache_key = f"{use_gpu}_{solver_type}_{preconditioner}"
    
    if cache_key not in _global_solver_cache:
        solver = PETScSolver(use_gpu=use_gpu)
        _global_solver_cache[cache_key] = solver
    else:
        solver = _global_solver_cache[cache_key]
    
    # 设置矩阵和向量
    matrix_size = theCoefficients.NumberOfElements
    solver.setup_matrix(values, indptr, indices, matrix_size)
    solver.setup_vectors(theCoefficients)
    
    # 求解
    rtol_adj = relTol * initRes if relTol * initRes > tolerance else tolerance
    success, iterations, residual = solver.solve(
        solver_type=solver_type,
        preconditioner=preconditioner,
        rtol=rtol_adj,
        atol=tolerance,
        max_iter=maxIter
    )
    
    if success:
        theCoefficients.dphi = solver.get_solution()
        return initRes, np.float64(residual)
    else:
        return initRes, initRes