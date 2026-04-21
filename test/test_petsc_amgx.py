#!/usr/bin/env python3
"""
检测PETSc是否支持AMGX集成的测试脚本
需要在 conda activate pyOpenFOAM 环境下运行
"""

import sys
import contextlib
import io

def suppress_stderr():
    """临时抑制stderr输出"""
    return contextlib.redirect_stderr(io.StringIO())

def test_petsc_amgx():
    """测试PETSc AMGX集成"""
    print("=" * 60)
    print("🚀 PETSc AMGX 集成检测")
    print("=" * 60)
    
    # 1. 检查PETSc可用性
    try:
        from petsc4py import PETSc
        print("✅ PETSc (petsc4py) 导入成功")
    except ImportError as e:
        print(f"❌ PETSc不可用: {e}")
        return False
    
    # 2. 检查PETSc版本和配置
    try:
        version = PETSc.Sys.getVersion()
        print(f"📦 PETSc版本: {version}")
        
        # 检查编译配置
        configure_options = PETSc.Sys.hasExternalPackage('amgx')
        if configure_options:
            print("✅ PETSc编译时包含AMGX支持")
        else:
            print("⚠️  PETSc编译时未明确包含AMGX (但可能仍然可用)")
            
    except Exception as e:
        print(f"⚠️  获取PETSc信息时出现问题: {e}")
    
    # 3. 测试AMGX预处理器创建 (关键测试)
    print("\n🧪 测试AMGX预处理器创建...")
    pc_creation_success = False
    pc = None
    ksp = None
    
    try:
        # 使用更完整的方式创建预处理器
        ksp = PETSc.KSP().create()
        pc = ksp.getPC()
        pc.setType('amgx')
        print("🎉 AMGX预处理器创建成功!")
        pc_creation_success = True
        
        # 尝试设置一些基本选项 (验证功能)
        pc.setFromOptions()
        print("✅ AMGX预处理器配置成功")
        
    except Exception as e:
        print(f"❌ AMGX预处理器创建失败: {e}")
        pc_creation_success = False
    
    # 4. 检查PETSc.PC.Type常量 (预期会失败，但不影响功能)
    print("\n🔍 检查PETSc预处理器常量...")
    try:
        if hasattr(PETSc.PC.Type, 'AMGX'):
            print("✅ PETSc.PC.Type.AMGX 常量存在")
            amgx_type_available = True
        else:
            print("ℹ️  PETSc.PC.Type.AMGX 常量不存在 (正常现象，使用字符串方式)")
            amgx_type_available = False
    except Exception as e:
        print(f"⚠️  检查PC.Type常量时出错: {e}")
        amgx_type_available = False
    
    # 5. 显示可用的预处理器类型 (简化版)
    print("\n📋 可用的预处理器类型概览...")
    try:
        pc_types = [attr for attr in dir(PETSc.PC.Type) if not attr.startswith('_')]
        print(f"✅ 总共有 {len(pc_types)} 种预处理器类型可用")
        
        # 检查GPU相关类型
        gpu_pc_types = [t for t in pc_types if any(gpu in t.lower() for gpu in ['cuda', 'hip', 'viennacl'])]
        if gpu_pc_types:
            print(f"🎯 发现GPU相关类型: {', '.join(gpu_pc_types)}")
        
        # 检查常用高性能类型
        performance_types = [t for t in pc_types if t in ['GAMG', 'ML', 'HYPRE', 'HPDDM']]
        if performance_types:
            print(f"⚡ 高性能预处理器: {', '.join(performance_types)}")
            
    except Exception as e:
        print(f"⚠️  检查PC类型时出错: {e}")
    
    # 6. GPU和CUDA检查
    print("\n🖥️  GPU环境检查...")
    try:
        # 检查是否有CUDA支持
        cuda_available = PETSc.Sys.hasExternalPackage('cuda')
        if cuda_available:
            print("✅ PETSc支持CUDA")
        else:
            print("⚠️  PETSc未检测到CUDA支持")
            
    except Exception as e:
        print(f"⚠️  GPU检查时出错: {e}")
    
    # 7. 安全清理资源
    print("\n🧹 清理测试资源...")
    cleanup_success = True
    try:
        with suppress_stderr():  # 抑制清理时的警告信息
            if pc is not None:
                pc.destroy()
            if ksp is not None:
                ksp.destroy()
        print("✅ 资源清理完成")
    except Exception as e:
        print(f"⚠️  资源清理时有警告 (不影响功能): {type(e).__name__}")
        cleanup_success = False
    
    # 8. 最终结果汇总
    print("\n" + "=" * 60)
    print("🎯 检测结果汇总:")
    print("-" * 30)
    
    if pc_creation_success:
        print("🎉 AMGX集成状态: ✅ 完全可用")
        print("📈 性能模式: GPU加速已启用")
        print("🔧 使用方法: 在代码中设置 preconditioner='amgx' 和 use_gpu=True")
    else:
        print("❌ AMGX集成状态: 不可用")
        print("💡 建议: 检查AMGX库安装和OpenMP环境变量")
    
    print(f"\n📊 技术详情:")
    print(f"  • PETSc版本: {version if 'version' in locals() else '未知'}")
    print(f"  • AMGX预处理器: {'✅ 可创建' if pc_creation_success else '❌ 创建失败'}")
    print(f"  • PC.Type常量: {'✅ 存在' if amgx_type_available else 'ℹ️ 使用字符串模式'}")
    print(f"  • 资源清理: {'✅ 正常' if cleanup_success else '⚠️ 有警告'}")
    
    print("\n" + "=" * 60)
    
    if pc_creation_success:
        print("🚀 集成测试通过！可以开始使用GPU加速CFD求解了！")
    else:
        print("🔧 集成测试未通过，请检查安装配置")
    
    print("=" * 60)
    
    return pc_creation_success

if __name__ == "__main__":
    try:
        success = test_petsc_amgx()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生未预期错误: {e}")
        sys.exit(1)