#!/usr/bin/env python3
"""
PETSc求解器测试脚本
测试PETSc求解器的集成和性能
"""

import os
import sys
import time
import numpy as np

# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the path to the pyFVM directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
print(f"src 路径: {src_path}")
sys.path.insert(0, src_path)

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    MPI_AVAILABLE = False
    rank = 0
    size = 1
    print("MPI不可用，使用单进程测试")

from pyFVM import Coefficients as coefficients
from pyFVM.PETScSolver import PETScSolver, PETScLinearSolver, cfdSolvePETSc

class TestRegion:
    """模拟Region对象用于测试"""
    def __init__(self, n_elements=1000):
        self.n_elements = n_elements
        
        # 创建简单的网格邻接关系
        self.mesh = self._create_test_mesh(n_elements)
        
        # 创建系数对象
        self.coefficients = coefficients.Coefficients(self)
        
        # 设置矩阵格式为CSR以便PETSc使用
        self.coefficients.MatrixFormat = 'csr'
        self.coefficients._init_csr_format(self)
        
        # 生成测试问题
        self._setup_test_problem()
    
    def _create_test_mesh(self, n):
        """创建1D链式网格用于测试"""
        class TestMesh:
            def __init__(self, n):
                self.numberOfElements = n
                # 创建1D链式邻接关系
                self.elementNeighbours = []
                for i in range(n):
                    neighbors = []
                    if i > 0:
                        neighbors.append(i-1)
                    if i < n-1:
                        neighbors.append(i+1)
                    self.elementNeighbours.append(neighbors)
        
        return TestMesh(n)
    
    def _setup_test_problem(self):
        """设置测试的线性系统"""
        n = self.n_elements
        
        # 创建简单的三对角系统 (1D热传导)
        # -u_{i-1} + 2*u_i - u_{i+1} = f_i
        
        # 对角元素
        self.coefficients.ac = 2.0 * np.ones(n)
        
        # 非对角元素 (邻居系数)
        self.coefficients.anb = []
        for i in range(n):
            neighbors = self.mesh.elementNeighbours[i]
            anb_i = -1.0 * np.ones(len(neighbors))
            self.coefficients.anb.append(anb_i)
        
        # 右端项 (正弦函数)
        x = np.linspace(0, 1, n)
        self.coefficients.bc = np.sin(np.pi * x) * np.pi**2
        
        # 初始解猜测
        self.coefficients.dphi = np.zeros(n)
        
        # 更新CSR格式
        self.coefficients._update_csr_data()
        
        if rank == 0:
            print(f"测试问题设置完成: {n}元素三对角系统")

def test_petsc_solver_basic():
    """基础PETSc求解器测试"""
    if rank == 0:
        print("\n=== PETSc求解器基础测试 ===")
    
    # 检查PETSc可用性
    if not PETScLinearSolver.available():
        if rank == 0:
            print("PETSc不可用，跳过测试")
        return False
    
    # 创建测试问题
    test_region = TestRegion(n_elements=500)
    
    # 测试cfdSolvePETSc函数
    try:
        start_time = time.time()
        
        init_res, final_res = cfdSolvePETSc(
            test_region.coefficients,
            maxIter=1000,
            tolerance=1e-8,
            relTol=1e-6,
            solver_type='gmres',
            preconditioner='gamg'
        )
        
        petsc_time = time.time() - start_time
        
        if rank == 0:
            print(f"PETSc求解完成:")
            print(f"  初始残差: {init_res:.2e}")
            print(f"  最终残差: {final_res:.2e}")
            print(f"  求解时间: {petsc_time:.4f}s")
            print(f"  收敛比: {init_res/final_res:.1e}")
        
        return True
        
    except Exception as e:
        if rank == 0:
            print(f"PETSc求解测试失败: {e}")
        return False

def test_solver_comparison():
    """对比不同求解器性能"""
    if rank == 0:
        print("\n=== 求解器性能对比 ===")
    
    problem_sizes = [100, 500, 1000] if MPI_AVAILABLE else [100, 500]
    
    for n in problem_sizes:
        if rank == 0:
            print(f"\n问题规模: {n}元素")
        
        # 创建测试问题
        test_region = TestRegion(n_elements=n)
        
        results = {}
        
        # 测试SciPy求解器 (参考)
        if rank == 0:  # 只在主进程测试SciPy
            try:
                from scipy.sparse.linalg import spsolve
                from scipy.sparse import csr_matrix
                
                # 构造SciPy CSR矩阵
                A_csr = test_region.coefficients.data_sparse_matrix_update()
                b = test_region.coefficients.bc.copy()
                
                start_time = time.time()
                x_scipy = spsolve(A_csr, b)
                scipy_time = time.time() - start_time
                
                residual_scipy = np.linalg.norm(A_csr @ x_scipy - b)
                results['SciPy'] = {
                    'time': scipy_time,
                    'residual': residual_scipy,
                    'iterations': 'Direct'
                }
                
            except Exception as e:
                results['SciPy'] = {'error': str(e)}
        
        # 测试PETSc求解器
        if PETScLinearSolver.available():
            try:
                # 重置解向量
                test_region.coefficients.dphi = np.zeros(n)
                
                start_time = time.time()
                init_res, final_res = cfdSolvePETSc(
                    test_region.coefficients,
                    maxIter=1000,
                    tolerance=1e-10,
                    relTol=1e-8,
                    solver_type='gmres',
                    preconditioner='gamg'
                )
                petsc_time = time.time() - start_time
                
                results['PETSc'] = {
                    'time': petsc_time,
                    'residual': final_res,
                    'iterations': 'Unknown',
                    'init_residual': init_res
                }
                
            except Exception as e:
                results['PETSc'] = {'error': str(e)}
        
        # 打印结果对比
        if rank == 0:
            for solver_name, result in results.items():
                if 'error' in result:
                    print(f"  {solver_name:10s}: 失败 - {result['error']}")
                else:
                    time_str = f"{result['time']:.4f}s"
                    res_str = f"{result['residual']:.2e}"
                    iter_str = str(result['iterations'])
                    print(f"  {solver_name:10s}: {time_str:>10s}, 残差={res_str}, 迭代={iter_str}")

def test_different_solvers():
    """测试不同的PETSc求解器组合"""
    if rank == 0:
        print("\n=== 不同PETSc求解器组合测试 ===")
    
    if not PETScLinearSolver.available():
        if rank == 0:
            print("PETSc不可用，跳过测试")
        return
    
    # 测试不同求解器组合
    solver_configs = [
        ('gmres', 'gamg'),
        ('gmres', 'ilu'),
        ('cg', 'gamg'),
        ('bicgstab', 'ilu'),
        ('gmres', 'jacobi')
    ]
    
    test_region = TestRegion(n_elements=300)
    
    for solver_type, preconditioner in solver_configs:
        try:
            # 重置解向量
            test_region.coefficients.dphi = np.zeros(test_region.n_elements)
            
            start_time = time.time()
            init_res, final_res = cfdSolvePETSc(
                test_region.coefficients,
                maxIter=500,
                tolerance=1e-8,
                relTol=1e-6,
                solver_type=solver_type,
                preconditioner=preconditioner
            )
            solve_time = time.time() - start_time
            
            if rank == 0:
                config_name = f"{solver_type:>8s}+{preconditioner:>6s}"
                print(f"  {config_name}: {solve_time:.4f}s, 残差={final_res:.2e}, 收敛={init_res/final_res:.1e}")
                
        except Exception as e:
            if rank == 0:
                config_name = f"{solver_type:>8s}+{preconditioner:>6s}"
                print(f"  {config_name}: 失败 - {str(e)[:50]}")

def test_mpi_scalability():
    """测试MPI可扩展性"""
    if rank == 0:
        print(f"\n=== MPI可扩展性测试 (进程数={size}) ===")
    
    if not MPI_AVAILABLE or not PETScLinearSolver.available():
        if rank == 0:
            print("MPI或PETSc不可用，跳过测试")
        return
    
    # 测试不同规模问题的MPI扩展性
    problem_sizes = [1000, 2000, 5000]
    
    for n in problem_sizes:
        if rank == 0:
            print(f"\n问题规模: {n}元素, {size}进程")
        
        try:
            test_region = TestRegion(n_elements=n)
            
            # 同步所有进程
            comm.Barrier() if MPI_AVAILABLE else None
            start_time = time.time()
            
            init_res, final_res = cfdSolvePETSc(
                test_region.coefficients,
                maxIter=1000,
                tolerance=1e-8,
                relTol=1e-6,
                solver_type='gmres',
                preconditioner='gamg',
                comm=comm if MPI_AVAILABLE else None
            )
            
            comm.Barrier() if MPI_AVAILABLE else None
            total_time = time.time() - start_time
            
            if rank == 0:
                efficiency = n / (total_time * size) if total_time > 0 else 0
                print(f"  总时间: {total_time:.4f}s")
                print(f"  效率: {efficiency:.0f} 元素/(s·进程)")
                print(f"  残差: {init_res:.2e} -> {final_res:.2e}")
                
        except Exception as e:
            if rank == 0:
                print(f"  测试失败: {e}")

def main():
    """主测试函数"""
    if rank == 0:
        print("PETSc求解器集成测试")
        print("=" * 50)
        
        if MPI_AVAILABLE:
            print(f"MPI环境: {size}个进程")
        else:
            print("单进程环境")
    
    # 运行各项测试
    success = True
    
    # 基础功能测试
    if not test_petsc_solver_basic():
        success = False
    
    # 性能对比测试
    try:
        test_solver_comparison()
    except Exception as e:
        if rank == 0:
            print(f"性能对比测试出错: {e}")
        success = False
    
    # 不同求解器组合测试
    try:
        test_different_solvers()
    except Exception as e:
        if rank == 0:
            print(f"求解器组合测试出错: {e}")
    
    # MPI扩展性测试
    if MPI_AVAILABLE and size > 1:
        try:
            test_mpi_scalability()
        except Exception as e:
            if rank == 0:
                print(f"MPI扩展性测试出错: {e}")
    
    if rank == 0:
        print("\n" + "=" * 50)
        if success:
            print("✅ PETSc求解器集成测试通过")
            print("\n下一步:")
            print("1. 在实际CFD案例中测试PETSc求解器")
            print("2. 调优PETSc求解器参数")
            print("3. 实施GPU+MPI矩阵组装优化")
        else:
            print("❌ 部分测试失败，请检查PETSc安装和配置")
        print("=" * 50)

if __name__ == "__main__":
    main()
        """
        直接求解CSR格式的线性系统
        
        Args:
            A_csr: scipy.sparse CSR矩阵
            b: 右端向量
            x0: 初始解
            **kwargs: 求解器参数
            
        Returns:
            x: 解向量
            info: 求解信息
        """
        if not PETSC_AVAILABLE:
            from scipy.sparse.linalg import spsolve
            return spsolve(A_csr, b), {'converged': True}
        
        comm = kwargs.get('comm', MPI.COMM_WORLD)
        
        # 创建PETSc矩阵
        A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape, comm=comm)
        A_petsc.setUp()
        
        # 填充数据
        A_petsc.setValuesCSR(A_csr.indptr, A_csr.indices, A_csr.data)
        A_petsc.assemblyBegin()
        A_petsc.assemblyEnd()
        
        # 创建向量
        b_petsc = PETSc.Vec().createMPI(size=len(b), comm=comm)
        x_petsc = b_petsc.duplicate()
        
        b_petsc.setArray(b)
        if x0 is not None:
            x_petsc.setArray(x0)
        
        # 创建求解器
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A_petsc)
        ksp.setType(kwargs.get('solver_type', 'gmres'))
        
        pc = ksp.getPC()
        pc.setType(kwargs.get('preconditioner', 'gamg'))
        
        ksp.setTolerances(rtol=kwargs.get('rtol', 1e-6), 
                         max_it=kwargs.get('max_iter', 1000))
        
        # 求解
        ksp.solve(b_petsc, x_petsc)
        
        # 获取结果
        x = x_petsc.getArray().copy()
        converged = ksp.getConvergedReason() > 0
        iterations = ksp.getIterationNumber()
        
        # 清理
        ksp.destroy()
        A_petsc.destroy()
        b_petsc.destroy()
        x_petsc.destroy()
        
        return x, {'converged': converged, 'iterations': iterations}


# if __name__ == "__main__":
#     # 简单测试
#     if PETSC_AVAILABLE:
#         print("PETSc求解器模块测试通过")
        
#         # 创建测试矩阵
#         n = 100
#         from scipy.sparse import random
#         A_test = random(n, n, density=0.1) 
#         A_test = A_test + A_test.T + 2*np.eye(n)  # 确保正定
#         b_test = np.random.rand(n)
        
#         # 测试求解
#         x, info = PETScLinearSolver.solve_linear_system(A_test.tocsr(), b_test)
        
#         if info['converged']:
#             residual = np.linalg.norm(A_test @ x - b_test)
#             print(f"测试求解成功，残差: {residual:.2e}")
#         else:
#             print("测试求解失败")
#     else:
#         print("PETSc不可用，请安装: pip install petsc petsc4py")