#!/usr/bin/env python3
"""
PETSc求解器 - 单核版本
简化的PETSc线性求解器，专注于单核性能
"""
import cfdtool.Math as mth
import numpy as np
# import sys
# 检查PETSc可用性
try:
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    print("警告: PETSc不可用，将使用SciPy后备方案")

#TODO: 要启用GPU并行，需要：
#   配置选项方式（最简单）：
#   # 运行时通过环境变量
#   export PETSC_OPTIONS="-vec_type cuda -mat_type aijcusparse"
#   python pyFVMScript.py

#   代码修改方式：
#   # 在_create_matrix_structure()中改为GPU矩阵
#   self.A_petsc = PETSc.Mat().createAIJCUSPARSE(...)  # GPU稀疏矩阵
#   # 在create_vectors()中改为GPU向量
#   self.b_petsc = PETSc.Vec().createCUDA(...)         # GPU向量



class PETScSolver:
    """PETSc单核线性求解器
    PETSc对象包括：
  - Mat: 稀疏矩阵
  - Vec: 向量
  - KSP: Krylov子空间求解器
  - PC: 预处理器
    """
    def __init__(self):
        if not PETSC_AVAILABLE:
            raise ImportError("PETSc未安装，请运行: pip install petsc petsc4py")

        # PETSc初始化
        PETSc.Sys.pushErrorHandler("python")
        
        # 求解器配置
        self.ksp = None
        self.pc = None
        self.A_petsc = None
        self.b_petsc = None
        self.x_petsc = None
        
        # 矩阵结构缓存
        self.matrix_structure_cached = False
        self.cached_matrix_size = None
        self.cached_nnz = None
        
        # 配置缓存
        self.last_solver_config = None
        
        print("PETSc单核求解器初始化完成")
    
    
    def create_matrix_from_csr_data(self, values, indptr, indices, matrix_size):
        """从 CSR 数据创建PETSc矩阵"""
        # 检查是否需要重建矩阵结构
        if (not self.matrix_structure_cached or 
            self.cached_matrix_size != matrix_size):
            
            self._create_matrix_structure(values, indptr, matrix_size)
            self.matrix_structure_cached = True
            self.cached_matrix_size = matrix_size
            print(f"PETSc矩阵创建: {matrix_size}x{matrix_size}")
        
        # 更新矩阵数据
        self._update_matrix_data(values, indptr, indices)
    
    
    def _estimate_nnz(self, values, indptr):
        """估算每行非零元素数量 - 使用更保守的估算"""
        if len(indptr) <= 1:
            return 10  # 默认值
        
        # 计算每行的实际nnz
        row_nnz = []
        for i in range(len(indptr) - 1):
            row_nnz.append(indptr[i + 1] - indptr[i])
        
        # 使用最大值或平均值的1.5倍，取较大者
        avg_nnz = sum(row_nnz) // len(row_nnz)
        max_nnz = max(row_nnz)
        
        return max(max_nnz, int(avg_nnz * 1.5))
    
    def _create_matrix_structure(self, values, indptr, matrix_size):
        """创建PETSc矩阵结构"""
        actual_nnz = len(values)
        nnz_per_row = self._estimate_nnz(values, indptr)
        
        # 创建单核AIJ矩阵
        self.A_petsc = PETSc.Mat().createAIJ([matrix_size, matrix_size], nnz=nnz_per_row)
        
        # 允许新的非零元素分配，避免结构错误
        self.A_petsc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        
        self.A_petsc.setUp()
        
        print(f"PETSc矩阵结构创建: {matrix_size}x{matrix_size}, nnz={actual_nnz}")
    
    def _update_matrix_data(self, values, indptr, indices):
        """更新PETSc矩阵数据"""
        # 清空矩阵
        self.A_petsc.zeroEntries()
        
        # 使用CSR数据直接填充矩阵
        self._update_csr_matrix(values, indptr, indices)
        
        # 矩阵组装
        self.A_petsc.assemblyBegin()
        self.A_petsc.assemblyEnd()
    
    def _update_csr_matrix(self, values, row_ptr, col_indices):
        """使用CSR数据更新PETSc矩阵"""
        # 批量设置矩阵元素 - 逐行处理以提高缓存效率
        for i in range(len(row_ptr) - 1):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            
            if end > start:  # 该行有非零元素
                row_cols = col_indices[start:end]
                row_vals = values[start:end]
                # 批量设置一整行
                self.A_petsc.setValues(i, row_cols, row_vals)
    
    def create_vectors(self, theCoefficients):
        """创建PETSc向量"""
        n = theCoefficients.NumberOfElements
        
        # 创建单核向量
        self.b_petsc = PETSc.Vec().createSeq(n)
        self.x_petsc = PETSc.Vec().createSeq(n)
        
        # 设置右端向量
        if hasattr(theCoefficients, 'bc'):
            b_data = theCoefficients.bc
        else:
            b_data = np.zeros(n)
        
        self.b_petsc.setArray(b_data)
        
        # 设置初始解
        if hasattr(theCoefficients, 'dphi'):
            x_data = theCoefficients.dphi
        else:
            x_data = np.zeros(n)
        
        self.x_petsc.setArray(x_data)
        
        # 向量组装
        self.b_petsc.assemblyBegin()
        self.b_petsc.assemblyEnd()
        self.x_petsc.assemblyBegin()
        self.x_petsc.assemblyEnd()
        
        print("PETSc向量创建完成")
    
    def setup_solver(self, solver_type='gmres', preconditioner='ilu', **kwargs):
        """配置PETSc求解器"""
        current_config = f"{solver_type}_{preconditioner}_{kwargs.get('rtol', 1e-6)}"
        
        # 检查配置复用
        if (self.ksp is not None and 
            self.last_solver_config == current_config):
            # 只需要更新算子
            self.ksp.setOperators(self.A_petsc)
            print(f"复用求解器配置: {solver_type} + {preconditioner}")
            return
        
        # 创建新的求解器
        if self.ksp is None:
            self.ksp = PETSc.KSP().create()
        
        self.ksp.setOperators(self.A_petsc)
        
        # 设置求解器类型
        self._configure_solver_type(solver_type, **kwargs)
        
        # 设置预处理器
        self._configure_preconditioner(preconditioner)
        
        # 设置收敛参数
        self._configure_tolerances(**kwargs)
        
        # 性能优化设置
        self._configure_performance_options()
        
        # 从选项数据库设置其他参数
        self.ksp.setFromOptions()
        
        # 缓存配置
        self.last_solver_config = current_config
        
        print(f"PETSc求解器配置: {solver_type} + {preconditioner}")
    
    def _configure_solver_type(self, solver_type, **kwargs):
        """配置求解器类型"""
        if solver_type.lower() == 'gmres':
            self.ksp.setType(PETSc.KSP.Type.GMRES)
            self.ksp.setGMRESRestart(kwargs.get('restart', 30))
        elif solver_type.lower() == 'cg':
            self.ksp.setType(PETSc.KSP.Type.CG)
        elif solver_type.lower() == 'bicgstab':
            self.ksp.setType(PETSc.KSP.Type.BCGS)
        elif solver_type.lower() == 'richardson':
            self.ksp.setType(PETSc.KSP.Type.RICHARDSON)
        else:
            print(f"未知求解器类型 {solver_type}，使用GMRES")
            self.ksp.setType(PETSc.KSP.Type.GMRES)
    
    def _configure_preconditioner(self, preconditioner):
        """配置预处理器"""
        self.pc = self.ksp.getPC()
        
        if preconditioner.lower() == 'ilu':
            self.pc.setType(PETSc.PC.Type.ILU)
        elif preconditioner.lower() == 'jacobi':
            self.pc.setType(PETSc.PC.Type.JACOBI)
        elif preconditioner.lower() == 'sor':
            self.pc.setType(PETSc.PC.Type.SOR)
        elif preconditioner.lower() == 'lu':
            self.pc.setType(PETSc.PC.Type.LU)
        elif preconditioner.lower() == 'none':
            self.pc.setType(PETSc.PC.Type.NONE)
        else:
            print(f"未知预处理器 {preconditioner}，使用ILU")
            self.pc.setType(PETSc.PC.Type.ILU)
    
    def _configure_tolerances(self, **kwargs):
        """配置收敛参数"""
        rtol = min(0.1, max(1e-12, kwargs.get('rtol', 1e-6)))
        atol = max(1e-50, kwargs.get('atol', 1e-50))
        max_iter = kwargs.get('max_iter', 1000)
        
        self.ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_iter)
        
        print(f"收敛参数: rtol={rtol:.2e}, atol={atol:.2e}, max_iter={max_iter}")
    
    def _configure_performance_options(self):
        """配置性能优化选项"""
        try:
            # 启用预处理器复用
            if self.matrix_structure_cached:
                if hasattr(self.ksp, 'setReusePreconditioner'):
                    self.ksp.setReusePreconditioner(True)
                
                if hasattr(self.ksp, 'setInitialGuessNonzero'):
                    self.ksp.setInitialGuessNonzero(True)
        except Exception as e:
            print(f"性能优化配置跳过: {e}")
        
        try:
            # 设置矩阵优化选项
            if hasattr(self.A_petsc, 'setOption'):
                self.A_petsc.setOption(PETSc.Mat.Option.USE_HASH_TABLE, False)
        except Exception as e:
            print(f"矩阵优化选项配置跳过: {e}")
    
    def solve(self, solver_type='gmres', preconditioner='ilu'):
        """执行线性求解"""
        if self.ksp is None:
            raise RuntimeError("求解器未初始化，请先调用setup_solver()")
        
        try:
            # 执行求解
            self.ksp.solve(self.b_petsc, self.x_petsc)
            
            # 获取收敛信息
            reason = self.ksp.getConvergedReason()
            iterations = self.ksp.getIterationNumber()
            residual = self.ksp.getResidualNorm()
            
            if reason > 0:
                print(f"PETSc求解成功: {iterations}次迭代, 残差={residual:.2e}")
                success = True
            else:
                print(f"PETSc求解失败: reason={reason}, {iterations}次迭代, 残差={residual:.2e}")
                success = False
                
                # 尝试备用方案
                if preconditioner != 'lu':
                    print("尝试LU分解作为备用方案")
                    return self._fallback_solve()
            
            return success, iterations, np.float64(residual)
            
        except Exception as e:
            print(f"求解异常: {e}")
            return self._fallback_solve()
    
    def _fallback_solve(self):
        """备用求解方案"""
        try:
            # 使用直接求解器
            self.pc.setType(PETSc.PC.Type.LU)
            self.ksp.setType(PETSc.KSP.Type.PREONLY)  # 只使用预处理器
            self.ksp.setFromOptions()
            
            self.ksp.solve(self.b_petsc, self.x_petsc)
            
            reason = self.ksp.getConvergedReason()
            iterations = self.ksp.getIterationNumber()
            residual = self.ksp.getResidualNorm()
            
            if reason > 0:
                print(f"备用求解器成功: LU直接求解, 残差={residual:.2e}")
                return True, iterations, np.float64(residual)
            else:
                print("备用求解器也失败")
                return False, 0, np.float64(0.0)
                
        except Exception as e:
            print(f"备用求解器异常: {e}")
            return False, 0, np.float64(0.0)
    
    def get_solution(self):
        """获取解向量"""
        if self.x_petsc is None:
            raise RuntimeError("解向量不存在")
        
        return self.x_petsc.getArray().copy()
    
    def update_coefficients(self, theCoefficients):
        """将求解结果更新回系数结构"""
        solution = self.get_solution()
        theCoefficients.dphi = solution
        print("解向量已更新到coefficients.dphi")
    
    
    def cleanup(self):
        """清理PETSc对象"""
        if self.ksp:
            self.ksp.destroy()
        if self.A_petsc:
            self.A_petsc.destroy()
        if self.b_petsc:
            self.b_petsc.destroy()
        if self.x_petsc:
            self.x_petsc.destroy()
        
        # 重置缓存
        self.matrix_structure_cached = False


def _get_csr_data(theCoefficients):
    """统一获取CSR格式数据"""
    if theCoefficients.MatrixFormat in ['csr', 'ldu', 'acnb']:
        return (theCoefficients.csrdata, 
               theCoefficients._indptr, 
               theCoefficients._indices)
    elif theCoefficients.MatrixFormat == 'coo':
        A_csr = theCoefficients._A_sparse.tocsr()
        return A_csr.data, A_csr.indptr, A_csr.indices
    else:
        raise ValueError(f"PETScSolver只支持CSR/COO格式，当前格式: {theCoefficients.MatrixFormat}")


def cfdSolvePETSc(theCoefficients, maxIter=1000, tolerance=1e-6, relTol=0.1, 
                  solver_type='gmres', preconditioner='ilu'):
    """
    PETSc求解器的高层接口 - 单核CSR版本
    
    Args:
        theCoefficients: 系数对象 (支持CSR/COO格式)
            CSR格式要求: csrdata, _indptr, _indices
            COO格式要求: _A_sparse (scipy.sparse矩阵)
            通用要求: bc (右端向量), dphi (解向量，可选)
        maxIter: 最大迭代次数
        tolerance: 绝对收敛容限
        relTol: 相对收敛容限  
        solver_type: 求解器类型 ('gmres', 'cg', 'bicgstab')
        preconditioner: 预处理器 ('ilu', 'jacobi', 'sor', 'lu')
    
    Returns:
        (initRes, finalRes): 初始和最终残差
    """
    theCoefficients.data_sparse_matrix_update()
    
    # 获取CSR格式数据
    values, indptr, indices = _get_csr_data(theCoefficients)
    
    # 计算初始残差
    residualsArray = theCoefficients.cfdComputeResidualsArray()
    initRes = mth.cfdResidual(residualsArray)
    
    if initRes < tolerance or maxIter == 0:
        return initRes, initRes
    
    # 创建单核PETSc求解器
    petsc_solver = PETScSolver()
    
    # 创建矩阵和向量
    matrix_size = theCoefficients.NumberOfElements
    petsc_solver.create_matrix_from_csr_data(values, indptr, indices, matrix_size)
    petsc_solver.create_vectors(theCoefficients)
    
    # 配置求解器
    petsc_solver.setup_solver(
        solver_type=solver_type,
        preconditioner=preconditioner,
        rtol=relTol * initRes if relTol * initRes > tolerance else tolerance,
        atol=tolerance,
        max_iter=maxIter
    )
    
    # 求解
    success, iterations, final_residual = petsc_solver.solve(solver_type, preconditioner)
    
    if success:
        # 更新解到原系数结构
        petsc_solver.update_coefficients(theCoefficients)
        finalRes = np.float64(final_residual)
    else:
        finalRes = initRes
        print("警告: PETSc求解未收敛，保持原解")
    
    # 清理
    petsc_solver.cleanup()
    
    return initRes, finalRes
