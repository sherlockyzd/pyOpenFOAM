#!/usr/bin/env python3
"""
AMGX诊断脚本 - 分析段错误根本原因
"""
import os
import sys
import numpy as np

def diagnose_amgx():
    print("=" * 60)
    print("🔍 AMGX 深度诊断")
    print("=" * 60)
    
    # 1. 环境检查
    print("\n1. 🌍 环境检查:")
    env_vars = ['PETSC_DIR', 'PETSC_ARCH', 'AMGX_DIR', 'LD_LIBRARY_PATH', 'CUDA_HOME']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"   {var}: {value}")
    
    # 2. 库版本检查
    print("\n2. 📚 库版本检查:")
    try:
        from petsc4py import PETSc
        print(f"   PETSc版本: {PETSc.Sys.getVersion()}")
        print(f"   PETSc架构: {PETSc.Sys.getArch()}")
        
        # 检查编译配置
        configure_opts = PETSc.Sys.getVersionInfo()
        print(f"   PETSc配置信息: {len(configure_opts)} 个选项")
        
        # 检查CUDA支持
        cuda_support = PETSc.Sys.hasExternalPackage('cuda')
        print(f"   CUDA支持: {cuda_support}")
        
        # 检查AMGX支持
        amgx_support = PETSc.Sys.hasExternalPackage('amgx')
        print(f"   AMGX支持: {amgx_support}")
        
    except Exception as e:
        print(f"   ❌ PETSc检查失败: {e}")
    
    # 3. CUDA检查
    print("\n3. 🎯 CUDA环境检查:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   GPU设备: {result.stdout.strip()}")
        else:
            print("   ❌ nvidia-smi 失败")
            
        # CUDA版本
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"   NVCC版本: {line.strip()}")
                    break
    except Exception as e:
        print(f"   ⚠️ CUDA检查异常: {e}")
    
    # 4. 最小AMGX测试
    print("\n4. 🧪 最小AMGX测试:")
    try:
        from petsc4py import PETSc
        
        # 创建最小矩阵 (2x2)
        A = PETSc.Mat().createAIJ([2, 2], nnz=2)
        A.setValues(0, [0, 1], [2.0, -1.0])
        A.setValues(1, [0, 1], [-1.0, 2.0])
        A.assemble()
        print("   ✅ 创建测试矩阵成功")
        
        # 创建向量
        b = PETSc.Vec().createSeq(2)
        x = PETSc.Vec().createSeq(2)
        b.setValues([0, 1], [1.0, 1.0])
        b.assemble()
        print("   ✅ 创建测试向量成功")
        
        # 尝试创建AMGX预处理器
        ksp = PETSc.KSP().create()
        ksp.setOperators(A, A)
        pc = ksp.getPC()
        
        print("   🎯 尝试设置AMGX预处理器...")
        pc.setType('amgx')
        print("   ✅ AMGX预处理器设置成功")
        
        # 最关键的测试：setFromOptions
        print("   🎯 尝试setFromOptions...")
        ksp.setFromOptions()
        print("   ✅ setFromOptions成功")
        
        # 尝试求解
        print("   🎯 尝试最小求解...")
        ksp.solve(b, x)
        print("   🎉 AMGX最小求解成功!")
        
        # 清理
        A.destroy()
        b.destroy() 
        x.destroy()
        ksp.destroy()
        
    except Exception as e:
        print(f"   ❌ AMGX测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 诊断结论
    print("\n5. 📋 诊断结论:")
    print("   基于以上测试结果，可能的解决方案:")
    print("   a) 重新编译PETSc，确保CUDA和AMGX版本匹配")
    print("   b) 检查AMGX编译时的CUDA架构设置")
    print("   c) 使用不同的矩阵格式 (MPIAIJ vs SEQAIJ)")
    print("   d) 调整OpenMP/CUDA线程配置")
    print("   e) 尝试AMGX的不同配置参数")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    diagnose_amgx()