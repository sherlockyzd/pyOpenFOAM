#!/usr/bin/env python3
"""
诊断CFD矩阵数据的特殊性质
找出为什么相同的CUSPARSE+AMGX在CFD应用中失败
"""
import os
import sys
import numpy as np

sys.path.append('/mnt/f/Desktop/pyFVM-master/pyOpenFOAM/src')

def diagnose_cfd_matrix():
    print("=" * 70)
    print("🔬 CFD矩阵数据诊断")
    print("=" * 70)
    
    try:
        # 导入CFD相关模块 
        import pyFVM.Region as Rg
        from petsc4py import PETSc
        
        # 模拟CFD矩阵生成过程
        print("\n1. 🏗️ 生成CFD矩阵...")
        # 这里需要实际的CFD矩阵生成代码
        # 但目前我们无法直接访问，所以用模拟数据
        
        n = 400
        # 创建更接近CFD特性的矩阵
        print(f"   创建 {n}x{n} CFD风格矩阵...")
        
        # 模拟CFD矩阵的特殊性质
        # 1. 非均匀稀疏模式 (边界密集，内部稀疏)
        # 2. 大的条件数
        # 3. 接近奇异
        
        A = PETSc.Mat().create()
        A.setSizes([n, n])
        A.setType(PETSc.Mat.Type.AIJCUSPARSE)
        A.setPreallocationNNZ(7)  # CFD通常每行5-7个非零元素
        A.setUp()
        
        # 构造CFD风格的系数矩阵 (模拟扩散-对流方程)
        print("   填充CFD风格矩阵数据...")
        boundary_factor = 1000.0  # 边界条件的强制系数
        
        for i in range(n):
            row_sum = 0.0
            values = []
            cols = []
            
            # 边界节点处理
            is_boundary = (i < 20 or i >= n-20)  # 前后20个节点作为边界
            
            if is_boundary:
                # Dirichlet边界条件: 大对角元
                A.setValues(i, [i], [boundary_factor])
            else:
                # 内部节点: 5点模板 (2D CFD)
                # 中心差分 + 对流项的上风格式
                center_coeff = 4.0
                neighbor_coeff = -1.0
                convection_coeff = 0.1  # 对流项
                
                # 左邻居
                if i-1 >= 0 and not (i-1 < 20):
                    cols.append(i-1)
                    values.append(neighbor_coeff - convection_coeff)
                
                # 右邻居  
                if i+1 < n and not (i+1 >= n-20):
                    cols.append(i+1)
                    values.append(neighbor_coeff + convection_coeff)
                
                # 上下邻居 (模拟2D网格)
                grid_width = int(np.sqrt(n))
                up = i - grid_width
                down = i + grid_width
                
                if up >= 0 and not (up < 20):
                    cols.append(up)
                    values.append(neighbor_coeff)
                
                if down < n and not (down >= n-20):
                    cols.append(down) 
                    values.append(neighbor_coeff)
                
                # 对角元 = 负的所有邻居系数之和
                diagonal = -sum(values)
                cols.append(i)
                values.append(diagonal)
                
                # 批量设置
                A.setValues(i, cols, values)
        
        A.assemble()
        print("   ✅ CFD矩阵构造完成")
        
        # 分析矩阵性质
        print("\n2. 📊 矩阵性质分析:")
        info = A.getInfo()
        print(f"   非零元素数: {int(info['nz_used'])}")
        print(f"   内存分配: {info['memory']:.2f} bytes")
        print(f"   填充率: {info['fill_ratio_given']:.4f}")
        
        # 创建对应向量
        print("\n3. 🎯 向量创建与数据填充:")
        b = PETSc.Vec().create()
        b.setSizes(n)
        b.setType(PETSc.Vec.Type.CUDA)
        b.setUp()
        
        x = b.duplicate()
        
        # 填充特殊的RHS向量 (模拟CFD源项)
        b_array = np.zeros(n)
        for i in range(n):
            if i < 20 or i >= n-20:  # 边界
                b_array[i] = 300.0  # 边界温度
            else:  # 内部
                b_array[i] = 0.0   # 无源项
        
        b.setValues(range(n), b_array)
        b.assemble()
        print("   ✅ CUDA向量创建完成")
        
        # 测试AMGX求解
        print("\n4. 🧪 AMGX求解测试:")
        ksp = PETSc.KSP().create()
        ksp.setOperators(A, A)
        ksp.setType(PETSc.KSP.Type.GMRES)
        
        pc = ksp.getPC()
        pc.setType('amgx')
        
        ksp.setTolerances(rtol=1e-6, atol=1e-10, max_it=1000)
        ksp.setFromOptions()
        
        print("   🎯 尝试AMGX求解CFD风格矩阵...")
        try:
            ksp.solve(b, x)
            
            # 检查收敛
            x.assemble()
            residual_norm = ksp.getResidualNorm()
            iterations = ksp.getIterationNumber()
            
            print(f"   🎉 AMGX求解成功!")
            print(f"   迭代次数: {iterations}")
            print(f"   残差范数: {residual_norm:.2e}")
            
            # 检查解的范数
            solution_norm = x.norm()
            print(f"   解的范数: {solution_norm:.6f}")
            
        except Exception as e:
            print(f"   ❌ AMGX求解失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理
        A.destroy()
        b.destroy()
        x.destroy()
        ksp.destroy()
        
        print("\n" + "=" * 70)
        print("📋 结论: 问题可能在于:")
        print("1. CFD矩阵的病态条件数")
        print("2. 边界条件的数值处理方式") 
        print("3. 向量数据的特殊访问模式")
        print("4. pyOpenFOAM中的内存管理冲突")
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_cfd_matrix()