#!/usr/bin/env python3
"""
CUSPARSE vs AMGX兼容性测试
分析为什么CUSPARSE矩阵不能与AMGX配合使用
"""
import os
import sys
import numpy as np

def test_cusparse_amgx_compatibility():
    print("=" * 70)
    print("🔬 CUSPARSE vs AMGX 兼容性深度分析")
    print("=" * 70)
    
    try:
        from petsc4py import PETSc
        
        # 创建测试矩阵数据 (400x400稀疏矩阵 - 模拟CFD矩阵)
        n = 400
        print(f"\n1. 创建 {n}x{n} 测试矩阵...")
        
        # 测试1：CUSPARSE矩阵 + AMGX
        print("\n🎯 测试1: CUSPARSE矩阵 + AMGX预处理器")
        try:
            # 创建CUSPARSE矩阵
            A_cusparse = PETSc.Mat().create()
            A_cusparse.setSizes([n, n])
            A_cusparse.setType(PETSc.Mat.Type.AIJCUSPARSE)
            A_cusparse.setPreallocationNNZ(5)  # 每行平均5个非零元素
            A_cusparse.setUp()
            
            # 填充矩阵数据 (简单的三对角矩阵)
            for i in range(n):
                if i > 0:
                    A_cusparse.setValues(i, [i-1], [-1.0])
                A_cusparse.setValues(i, [i], [2.0])
                if i < n-1:
                    A_cusparse.setValues(i, [i+1], [-1.0])
            
            A_cusparse.assemble()
            print("   ✅ CUSPARSE矩阵创建成功")
            
            # 创建CUDA向量
            b_cuda = PETSc.Vec().create()
            b_cuda.setSizes(n)
            b_cuda.setType(PETSc.Vec.Type.CUDA)
            b_cuda.setUp()
            b_cuda.setValues(range(n), [1.0] * n)
            b_cuda.assemble()
            
            x_cuda = b_cuda.duplicate()
            print("   ✅ CUDA向量创建成功")
            
            # 创建求解器并尝试使用AMGX
            ksp = PETSc.KSP().create()
            ksp.setOperators(A_cusparse, A_cusparse)
            ksp.setType(PETSc.KSP.Type.GMRES)
            
            pc = ksp.getPC()
            print("   🎯 尝试设置AMGX预处理器...")
            pc.setType('amgx')
            print("   ✅ AMGX预处理器设置成功")
            
            print("   🎯 尝试setFromOptions...")
            ksp.setFromOptions()
            print("   ✅ setFromOptions成功")
            
            print("   🎯 尝试求解...")
            ksp.solve(b_cuda, x_cuda)
            print("   🎉 CUSPARSE + AMGX 求解成功!")
            
            # 检查结果
            x_cuda.assemble()
            norm = x_cuda.norm()
            print(f"   解的范数: {norm:.6f}")
            
            # 清理
            A_cusparse.destroy()
            b_cuda.destroy()
            x_cuda.destroy()
            ksp.destroy()
            
        except Exception as e:
            print(f"   ❌ CUSPARSE + AMGX 失败: {e}")
            import traceback
            print("   详细错误信息:")
            traceback.print_exc()
        
        print("\n" + "─" * 70)
        
        # 测试2：SeqAIJ矩阵 + AMGX
        print("🎯 测试2: SeqAIJ矩阵 + AMGX预处理器")
        try:
            # 创建SeqAIJ矩阵
            A_seqaij = PETSc.Mat().create()
            A_seqaij.setSizes([n, n])
            A_seqaij.setType(PETSc.Mat.Type.SEQAIJ)
            A_seqaij.setPreallocationNNZ(5)
            A_seqaij.setUp()
            
            # 填充相同的矩阵数据
            for i in range(n):
                if i > 0:
                    A_seqaij.setValues(i, [i-1], [-1.0])
                A_seqaij.setValues(i, [i], [2.0])
                if i < n-1:
                    A_seqaij.setValues(i, [i+1], [-1.0])
            
            A_seqaij.assemble()
            print("   ✅ SeqAIJ矩阵创建成功")
            
            # 创建CPU向量
            b_seq = PETSc.Vec().createSeq(n)
            x_seq = PETSc.Vec().createSeq(n)
            b_seq.setValues(range(n), [1.0] * n)
            b_seq.assemble()
            print("   ✅ CPU向量创建成功")
            
            # 创建求解器并使用AMGX
            ksp2 = PETSc.KSP().create()
            ksp2.setOperators(A_seqaij, A_seqaij)
            ksp2.setType(PETSc.KSP.Type.GMRES)
            
            pc2 = ksp2.getPC()
            print("   🎯 尝试设置AMGX预处理器...")
            pc2.setType('amgx')
            print("   ✅ AMGX预处理器设置成功")
            
            print("   🎯 尝试setFromOptions...")
            ksp2.setFromOptions()
            print("   ✅ setFromOptions成功")
            
            print("   🎯 尝试求解...")
            ksp2.solve(b_seq, x_seq)
            print("   🎉 SeqAIJ + AMGX 求解成功!")
            
            # 检查结果
            x_seq.assemble()
            norm2 = x_seq.norm()
            print(f"   解的范数: {norm2:.6f}")
            
            # 清理
            A_seqaij.destroy()
            b_seq.destroy()
            x_seq.destroy()
            ksp2.destroy()
            
        except Exception as e:
            print(f"   ❌ SeqAIJ + AMGX 失败: {e}")
            import traceback
            print("   详细错误信息:")
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("📋 分析结论:")
        print("1. 矩阵格式兼容性问题")
        print("2. GPU内存管理冲突")
        print("3. CUDA上下文管理问题") 
        print("4. 数据传输和同步问题")
        
    except Exception as e:
        print(f"❌ 测试环境初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cusparse_amgx_compatibility()