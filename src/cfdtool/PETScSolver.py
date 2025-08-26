#!/usr/bin/env python3
"""
PETSc求解器集成模块
优先实现PETSc分布式线性求解，为后续GPU+MPI组装做准备
"""
import cfdtool.Math as mth
import numpy as np
from mpi4py import MPI
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


# PETSc求解器对象池 - 避免重复创建销毁
_petsc_solver_pool = {}

class PETScSolver:
    """PETSc分布式线性求解器 - 优化版本"""
    def __init__(self, comm=None):
        if not PETSC_AVAILABLE:
            raise ImportError("PETSc未安装，请运行: pip install petsc petsc4py")
          
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
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
        self.cached_indptr = None
        self.cached_indices = None
        self.cached_size = None
        
        # 性能优化配置
        self.preconditioner_cached = False
        self.last_solver_config = None
        
        if self.rank == 0:
            print(f"PETSc求解器初始化完成，MPI进程数: {self.size}")
    
    @classmethod
    def get_cached_solver(cls, equation_name, matrix_size, comm=None):
        """获取缓存的求解器实例，避免重复创建"""
        key = f"{equation_name}_{matrix_size}_{comm.Get_rank() if comm else 0}"
        
        if key not in _petsc_solver_pool:
            _petsc_solver_pool[key] = cls(comm=comm)
            if comm is None or comm.Get_rank() == 0:
                print(f"创建新的PETSc求解器缓存: {equation_name}")
        
        return _petsc_solver_pool[key]
    
    def create_matrix_from_coefficients(self, theCoefficients):
        """从现有系数结构创建PETSc分布式矩阵"""
        n_local = theCoefficients.NumberOfElements
        n_global = n_local  # 先简化为单进程情况
        
        
        if not self.matrix_structure_cached:
            # 首次创建 - 缓存矩阵结构
            self._cache_matrix_structure(theCoefficients, n_local, n_global)
            self._create_matrix_structure()
            self.matrix_structure_cached = True
            
            if self.rank == 0:
                print(f"PETSc矩阵结构创建并缓存: {n_global}x{n_global}")
        
        # 更新稀疏矩阵数据
        self._update_matrix_data(theCoefficients)
    
    def _estimate_nnz(self, theCoefficients):
        """估算每行非零元素数量"""
        if theCoefficients.MatrixFormat == 'acnb':
            # ac对角 + anb非对角
            avg_neighbors = np.mean([len(neighbors) for neighbors in theCoefficients._theCConn])
            return int(avg_neighbors + 1)  # +1 for diagonal
        elif theCoefficients.MatrixFormat == 'ldu':
            # 估算LDU格式的非零元素
            return len(theCoefficients.Upper) // theCoefficients.NumberOfElements + 1
        else:
            return 7  # 默认估算值
    
    
    def create_vectors(self, theCoefficients):
        """创建PETSc向量"""
        n_local = theCoefficients.NumberOfElements
        n_global = n_local
        
        # 创建右端向量
        self.b_petsc = PETSc.Vec().createMPI(
            size=(n_local, n_global), 
            comm=self.comm
        )
        
        # 创建解向量
        self.x_petsc = self.b_petsc.duplicate()
        
        # 设置初始值
        rstart, rend = self.b_petsc.getOwnershipRange()
        
        # 设置RHS
        b_local = theCoefficients.bc[rstart:rend] if hasattr(theCoefficients, 'bc') else np.zeros(rend-rstart)
        self.b_petsc.setArray(b_local)
        
        # 设置初始解
        x_local = theCoefficients.dphi[rstart:rend] if hasattr(theCoefficients, 'dphi') else np.zeros(rend-rstart)
        self.x_petsc.setArray(x_local)
        
        self.b_petsc.assemblyBegin()
        self.b_petsc.assemblyEnd()
        self.x_petsc.assemblyBegin() 
        self.x_petsc.assemblyEnd()
    
    def setup_solver(self, solver_type='gmres', preconditioner='gamg', **kwargs):
        """配置PETSc求解器 - 带智能缓存复用"""
        current_config = f"{solver_type}_{preconditioner}_{kwargs.get('rtol', 1e-6)}"
        
        # 检查是否可以复用现有配置
        if (self.ksp is not None and 
            self.last_solver_config == current_config and 
            self.preconditioner_cached):
            
            # 只需要更新算子，复用求解器配置
            self.ksp.setOperators(self.A_petsc)
            if self.rank == 0:
                print(f"复用PETSc求解器配置: {solver_type} + {preconditioner}")
            return
        
        # 创建或重新配置求解器
        if self.ksp is None:
            self.ksp = PETSc.KSP().create(comm=self.comm)
        
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
        self.preconditioner_cached = True
        
        if self.rank == 0:
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
        else:
            self.ksp.setType(PETSc.KSP.Type.GMRES)
    
    def _configure_preconditioner(self, preconditioner):
        """配置预处理器"""
        self.pc = self.ksp.getPC()
        if preconditioner.lower() == 'gamg':
            self.pc.setType(PETSc.PC.Type.GAMG)
            self.pc.setGAMGType(PETSc.PC.GAMGType.AGG)
        elif preconditioner.lower() == 'ilu':
            self.pc.setType(PETSc.PC.Type.ILU)
        elif preconditioner.lower() == 'jacobi':
            self.pc.setType(PETSc.PC.Type.JACOBI)
        elif preconditioner.lower() == 'none':
            self.pc.setType(PETSc.PC.Type.NONE)
        else:
            self.pc.setType(PETSc.PC.Type.GAMG)
    
    def _configure_tolerances(self, **kwargs):
        """配置收敛参数"""
        rtol = min(0.1, max(1e-12, kwargs.get('rtol', 1e-6)))
        atol = max(1e-50, kwargs.get('atol', 1e-50))
        max_iter = kwargs.get('max_iter', 1000)
        
        self.ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_iter)
        
        if self.rank == 0:
            print(f"求解器容差: rtol={rtol:.2e}, atol={atol:.2e}, max_iter={max_iter}")
    
    def _configure_performance_options(self):
        """配置性能优化选项 - 兼容不同PETSc版本"""
        try:
            # 启用预处理器复用以提升性能（如果支持）
            if self.matrix_structure_cached:
                if hasattr(self.ksp, 'setReusePreconditioner'):
                    self.ksp.setReusePreconditioner(True)
                
                if hasattr(self.ksp, 'setInitialGuessNonzero'):
                    self.ksp.setInitialGuessNonzero(True)  # 使用非零初始猜测
        except Exception as e:
            if self.rank == 0:
                print(f"预处理器复用配置跳过: {e}")
        
        try:
            # 设置矩阵优化选项
            if hasattr(self.A_petsc, 'setOption'):
                self.A_petsc.setOption(PETSc.Mat.Option.USE_HASH_TABLE, False)
                if self.size > 1:
                    self.A_petsc.setOption(PETSc.Mat.Option.IGNORE_OFF_PROC_ENTRIES, True)
        except Exception as e:
            if self.rank == 0:
                print(f"矩阵优化选项配置跳过: {e}")
    
    def solve(self, solver_type='gmres', preconditioner='gamg'):
        """执行线性求解，带备用方案"""
        if self.ksp is None:
            raise RuntimeError("求解器未初始化，请先调用setup_solver()")
        
        try:
            # 主要求解
            self.ksp.solve(self.b_petsc, self.x_petsc)
            
            # 获取收敛信息
            reason = self.ksp.getConvergedReason()
            iterations = self.ksp.getIterationNumber()
            residual = self.ksp.getResidualNorm()
            
            if self.rank == 0:
                if reason > 0:
                    print(f"PETSc求解成功: {iterations}次迭代, 残差={residual:.2e}")
                else:
                    print(f"PETSc求解失败: reason={reason}, {iterations}次迭代")
            
            # 如果失败，尝试备用方案
            if reason <= 0:
                success, iterations, residual = self._handle_solver_failure(reason, solver_type, preconditioner)
                return success, iterations, residual
            
            return True, iterations, residual
            
        except Exception as e:
            if self.rank == 0:
                print(f"求解异常: {e}")
            # 尝试备用方案
            return self._handle_solver_failure(-999, solver_type, preconditioner)
    
    def get_solution(self):
        """获取解向量"""
        if self.x_petsc is None:
            raise RuntimeError("解向量不存在")
        
        # 获取本地部分
        rstart, rend = self.x_petsc.getOwnershipRange()
        x_local = self.x_petsc.getArray()
        
        return x_local.copy()
    
    def update_coefficients(self, theCoefficients):
        """将PETSc解更新回系数结构"""
        x_local = self.get_solution()
        rstart, rend = self.x_petsc.getOwnershipRange()
        
        # 更新dphi
        theCoefficients.dphi[rstart:rend] = x_local
        
        if self.rank == 0:
            print("解向量已更新到coefficients.dphi")
    
    
    def _cache_matrix_structure(self, theCoefficients, n_local, n_global):
        """缓存稀疏矩阵结构信息"""
        if theCoefficients.MatrixFormat in ['csr', 'acnb', 'ldu']:
            self.cached_indptr = theCoefficients._indptr.copy()
            self.cached_indices = theCoefficients._indices.copy()
        elif theCoefficients.MatrixFormat == 'coo':
            self.cached_coo_row = theCoefficients._row.copy()
            self.cached_coo_col = theCoefficients._col.copy()
        else:
            raise ValueError(f"PETScSolver只支持CSR/COO稀疏格式，当前格式: {theCoefficients.MatrixFormat}")
        
        self.cached_size = (n_local, n_global)
    
    def _create_matrix_structure(self):
        """创建PETSc稀疏矩阵结构"""
        n_local, n_global = self.cached_size
        
        # 基于缓存的稀疏结构信息精确估算nnz
        if self.cached_indptr is not None:
            # CSR格式：精确计算每行非零元素数
            nnz_per_row = []
            for i in range(len(self.cached_indptr) - 1):
                nnz_per_row.append(self.cached_indptr[i+1] - self.cached_indptr[i])
            nnz = max(nnz_per_row) if nnz_per_row else 7
        else:
            # COO格式：估算平均非零元素数
            nnz = max(1, int(len(self.cached_coo_row) / n_local))
        
        # 创建PETSc矩阵
        self.A_petsc = PETSc.Mat().createAIJ(
            size=((n_local, n_global), (n_local, n_global)),
            nnz=nnz,
            comm=self.comm
        )
        
        # 允许动态分配新的非零元素（预防措施）
        self.A_petsc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    
    def _update_matrix_data(self, theCoefficients):
        """简化的矩阵数据更新 - 避免复杂优化"""
        rstart, rend = self.A_petsc.getOwnershipRange()
        
        # 清零矩阵
        self.A_petsc.zeroEntries()
        
        # 直接使用原始的简单设置方式
        if theCoefficients.MatrixFormat in ['csr', 'ldu', 'acnb']:
            data = theCoefficients.csrdata
            indptr = theCoefficients._indptr
            indices = theCoefficients._indices
            
            for i_local in range(rend - rstart):
                i_global = rstart + i_local
                if i_local < len(indptr) - 1:
                    start_idx = indptr[i_local]
                    end_idx = indptr[i_local + 1]
                    
                    if start_idx < end_idx and end_idx <= len(data):
                        cols = indices[start_idx:end_idx]
                        vals = data[start_idx:end_idx]
                        self.A_petsc.setValues([i_global], cols, vals,
                                             addv=PETSc.InsertMode.INSERT_VALUES)
        
        elif theCoefficients.MatrixFormat == 'coo':
            A_csr = theCoefficients._A_sparse.tocsr()
            
            for i_local in range(rend - rstart):
                i_global = rstart + i_local
                if i_local < len(A_csr.indptr) - 1:
                    start_idx = A_csr.indptr[i_local]
                    end_idx = A_csr.indptr[i_local + 1]
                    
                    if start_idx < end_idx:
                        cols = A_csr.indices[start_idx:end_idx]
                        vals = A_csr.data[start_idx:end_idx]
                        self.A_petsc.setValues([i_global], cols, vals,
                                             addv=PETSc.InsertMode.INSERT_VALUES)
        
        # 完成矩阵组装
        self.A_petsc.assemblyBegin()
        self.A_petsc.assemblyEnd()
    
    def _get_csr_data(self, theCoefficients):
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
    
    def _batch_set_csr_data(self, data, indptr, indices, rstart, rend):
        """批量设置CSR数据到PETSc矩阵 - 带安全检查"""
        # 安全检查
        n_rows = rend - rstart
        if len(indptr) < n_rows + 1:
            raise ValueError(f"indptr长度不足: 需要{n_rows + 1}, 实际{len(indptr)}")
        
        for i_local in range(n_rows):
            i_global = rstart + i_local
            
            # 确保不越界
            if i_local >= len(indptr) - 1:
                if self.rank == 0:
                    print(f"警告: indptr索引越界，跳过行 {i_global}")
                continue
                
            start_idx = indptr[i_local]
            end_idx = indptr[i_local + 1]
            
            # 检查索引范围
            if start_idx < 0 or end_idx < 0 or start_idx > len(data) or end_idx > len(data):
                if self.rank == 0:
                    print(f"警告: 数据索引越界 行{i_global}: [{start_idx}:{end_idx}], 数据长度{len(data)}")
                continue
                
            if start_idx < end_idx:
                cols = indices[start_idx:end_idx]
                vals = data[start_idx:end_idx]
                
                # 验证列索引
                if len(cols) > 0 and (cols.min() < 0 or cols.max() >= self.cached_size[1]):
                    if self.rank == 0:
                        print(f"警告: 列索引越界 行{i_global}: 列范围[{cols.min()}, {cols.max()}]")
                    continue
                
                try:
                    self.A_petsc.setValues([i_global], cols, vals,
                                         addv=PETSc.InsertMode.INSERT_VALUES)
                except Exception as e:
                    if self.rank == 0:
                        print(f"设置矩阵值失败 行{i_global}: {e}")
                    continue
    
    
    
    def _validate_matrix(self):
        """验证矩阵质量，避免病态矩阵"""
        try:
            # 检查矩阵是否有空行
            row_sums = self.A_petsc.getDiagonal()
            zero_diag_count = 0
            
            # 获取对角元素
            diag_array = row_sums.getArray()
            for i, diag_val in enumerate(diag_array):
                if abs(diag_val) < 1e-14:
                    zero_diag_count += 1
            
            if zero_diag_count > 0 and self.rank == 0:
                print(f"警告: 检测到{zero_diag_count}个接近零的对角元素，可能导致求解问题")
            
        except Exception as e:
            if self.rank == 0:
                print(f"矩阵验证失败: {e}")
    
    def _handle_solver_failure(self, reason, solver_type, preconditioner):
        """处理求解器失败，尝试备用方案"""
        if self.rank == 0:
            print(f"求解器失败 (reason={reason})，尝试备用方案...")
        
        # 对于GAMG失败，尝试ILU
        if preconditioner.lower() == 'gamg':
            if self.rank == 0:
                print("GAMG预处理器失败，切换到ILU预处理器")
            
            # 重新配置求解器
            self.pc.setType(PETSc.PC.Type.ILU)
            self.ksp.setFromOptions()
            
            # 再次尝试求解
            try:
                self.ksp.solve(self.b_petsc, self.x_petsc)
                reason = self.ksp.getConvergedReason()
                iterations = self.ksp.getIterationNumber()
                residual = self.ksp.getResidualNorm()
                
                if reason > 0:
                    if self.rank == 0:
                        print(f"备用求解器成功: {iterations}次迭代, 残差={residual:.2e}")
                    return True, iterations, residual
            except Exception as e:
                if self.rank == 0:
                    print(f"备用求解器也失败: {e}")
        
        return False, 0, 0.0
    
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
        
        # 清理缓存
        self.matrix_structure_cached = False
        self.cached_indptr = None
        self.cached_indices = None


def cfdSolvePETSc(theCoefficients, maxIter=1000, tolerance=1e-6, relTol=0.1, 
                  solver_type='gmres', preconditioner='gamg', comm=None):
    """
    PETSc求解器的高层接口，兼容现有代码
    
    Args:
        theCoefficients: 系数对象
        maxIter: 最大迭代次数
        tolerance: 绝对收敛容限
        relTol: 相对收敛容限  
        solver_type: 求解器类型 ('gmres', 'cg', 'bicgstab')
        preconditioner: 预处理器 ('gamg', 'ilu', 'jacobi')
        comm: MPI通信子
    
    Returns:
        (initRes, finalRes): 初始和最终残差
    """
    theCoefficients.data_sparse_matrix_update()
    # 计算初始残差
    residualsArray = theCoefficients.cfdComputeResidualsArray()
    initRes = mth.cfdResidual(residualsArray)
    
    if initRes < tolerance or maxIter == 0:
        return initRes, initRes
    
    # 暂时禁用求解器缓存，避免潜在的内存问题
    petsc_solver = PETScSolver(comm=comm)
    
    # 转换矩阵格式
    petsc_solver.create_matrix_from_coefficients(theCoefficients)
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
        finalRes = final_residual
    else:
        finalRes = initRes
        print("警告: PETSc求解未收敛，保持原解")
    
    # 清理
    petsc_solver.cleanup()
    
    return initRes, finalRes
