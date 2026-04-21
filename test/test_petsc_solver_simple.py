#!/usr/bin/env python3
"""
简化的PETSc求解器测试脚本
"""

import os
import sys
import time
import numpy as np

# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
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

# 测试PETSc模块导入
try:
    from pyFVM.PETScSolver import PETScSolver, PETScLinearSolver, cfdSolvePETSc
    PETSC_SOLVER_AVAILABLE = True
    print("✅ PETSc求解器模块导入成功")
except ImportError as e:
    PETSC_SOLVER_AVAILABLE = False
    print(f"❌ PETSc求解器模块导入失败: {e}")

def test_basic_import():
    """测试基本导入"""
    if rank == 0:
        print("\n=== 基本导入测试 ===")
    
    success = True
    
    # 测试PETSc可用性
    if PETSC_SOLVER_AVAILABLE:
        try:
            available = PETScLinearSolver.available()
            if rank == 0:
                print(f"PETSc可用性: {available}")
        except Exception as e:
            if rank == 0:
                print(f"PETSc可用性检查失败: {e}")
            success = False
    else:
        success = False
    
    return success

def test_simple_solve():
    """测试简单矩阵求解"""
    if rank == 0:
        print("\n=== 简单求解测试 ===")
    
    if not PETSC_SOLVER_AVAILABLE:
        if rank == 0:
            print("跳过：PETSc不可用")
        return False
    
    try:
        # 创建简单的3x3测试矩阵
        from scipy.sparse import csr_matrix
        
        # 简单的三对角矩阵
        data = np.array([2, -1, 2, -1, 2])
        row = np.array([0, 0, 1, 1, 2])
        col = np.array([0, 1, 0, 2, 2])
        A = csr_matrix((data, (row, col)), shape=(3, 3))
        b = np.array([1.0, 2.0, 3.0])
        
        if rank == 0:
            print("测试矩阵:")
            print(A.toarray())
            print(f"右端向量: {b}")
        
        # 使用PETSc求解
        start_time = time.time()
        x, info = PETScLinearSolver.solve_linear_system(A, b)
        solve_time = time.time() - start_time
        
        if rank == 0:
            print(f"求解结果: {x}")
            print(f"收敛信息: {info}")
            print(f"求解时间: {solve_time:.4f}s")
            
            # 验证结果
            residual = np.linalg.norm(A @ x - b)
            print(f"残差: {residual:.2e}")
            
            if residual < 1e-10:
                print("✅ 简单求解测试通过")
                return True
            else:
                print("❌ 简单求解测试失败：残差过大")
                return False
                
    except Exception as e:
        if rank == 0:
            print(f"❌ 简单求解测试出错: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    if rank == 0:
        print("PETSc求解器简化测试")
        print("=" * 40)
        
        if MPI_AVAILABLE:
            print(f"MPI环境: {size}个进程")
        else:
            print("单进程环境")
    
    success = True
    
    # 基本导入测试
    if not test_basic_import():
        success = False
    
    # 简单求解测试
    if not test_simple_solve():
        success = False
    
    if rank == 0:
        print("\n" + "=" * 40)
        if success:
            print("✅ PETSc求解器基本测试通过")
            print("\n使用建议:")
            print("1. 修改fvSolution文件中的solver为PETSc")
            print("2. 运行实际CFD案例测试")
        else:
            print("❌ 测试失败，请检查:")
            print("1. PETSc安装: pip install petsc petsc4py")
            print("2. MPI安装: pip install mpi4py") 
        print("=" * 40)

if __name__ == "__main__":
    main()