import numpy as np
# from scipy.sparse.linalg import cg, spilu, LinearOperator
# from scipy.sparse import csr_matrix
# from pyamg import smoothed_aggregation_solver
import os
import sys

# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the path to the pyFVM directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
print(f"src 路径: {src_path}")  # 打印路径，确保正确
sys.path.insert(0, src_path)

from pyFVM import Coefficients as coefficients
from cfdtool import Solve as slv

# 定义一个简单的 Mesh 类
class Mesh:
    def __init__(self, elementNeighbours):
        self.elementNeighbours = elementNeighbours

# 定义一个简单的 Region 类
class Region:
    def __init__(self, mesh):
        self.mesh = mesh

# 测试函数
def test_pcg_solver_large_system(N=1000):
    """
    测试PCG求解器在较大规模系统上的性能和准确性。

    Args:
        N (int, optional): 系统的规模（元素数量）。默认为1000。
    """
    # 构造1D网格的邻接关系
    theCConn = []
    for i in range(N):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < N - 1:
            neighbors.append(i + 1)
        theCConn.append(neighbors)

    # 设置系数
    ac = np.full(N, 2.0, dtype=np.float64)  # 对角线元素
    anb = []
    for i in range(N):
        neighbors = theCConn[i]
        coeffs = np.full(len(neighbors), -1.0, dtype=np.float64)  # 非对角线元素
        anb.append(coeffs)
    bc = np.ones(N, dtype=np.float64)+np.random.rand(N)  # 右端项，可以根据需要调整，加上一个随机偏移量

    # 创建一个假的 Region 对象，包含 mesh.elementNeighbours
    mesh = Mesh(elementNeighbours=theCConn)
    region = Region(mesh=mesh)

    # 实例化 Coefficients 对象
    coeffs = coefficients.Coefficients(region)

    # 设置系数
    coeffs.ac = ac
    coeffs.anb = anb
    coeffs.bc = bc

    # 组装矩阵
    coeffs.assemble_sparse_matrix(method='csr')

    # 验证矩阵属性
    try:
        coeffs.verify_matrix_properties()
        print("矩阵通过对称性和正定性验证。")
    except ValueError as e:
        print(f"矩阵验证失败: {e}")
        return

    # # 定义PCG求解器
    # def cfdSolvePCG(theCoefficients, maxIter, tolerance, relTol, preconditioner='ILU'):
    #     from scipy.sparse.linalg import spilu, LinearOperator  # 确保导入

    #     A_sparse = theCoefficients.assemble_sparse_matrix()
    #     theCoefficients.verify_matrix_properties()

    #     r = theCoefficients.cfdComputeResidualsArray()
    #     initRes = np.linalg.norm(r)

    #     if initRes < tolerance or maxIter == 0:
    #         return initRes, initRes

    #     dphi = np.copy(theCoefficients.dphi)  # 初始猜测
    #     bc = theCoefficients.bc

    #     # 设置预处理器
    #     if preconditioner == 'ILU':
    #         try:
    #             ilu = spilu(A_sparse)
    #             M_x = lambda x: ilu.solve(x)
    #             M = LinearOperator(shape=A_sparse.shape, matvec=M_x)
    #         except Exception as e:
    #             raise RuntimeError(f"ILU 分解失败: {e}")
    #     elif preconditioner == 'DIC':
    #         # diag_A = A_sparse.diagonal().copy()
    #         # diag_A[diag_A <= 0] = 1e-10
    #         # M_diag = 1.0 / np.sqrt(diag_A)
    #         # M = LinearOperator(shape=A_sparse.shape, matvec=lambda x: M_diag * x)
    #         # 使用pyamg的SMG作为预处理器示例
    #         ml = smoothed_aggregation_solver(A_sparse)
    #         M = ml.aspreconditioner()
    #     elif preconditioner == 'None':
    #         M = None  # 不使用预处理器
    #     else:
    #         raise ValueError(f"Unknown preconditioner: {preconditioner}")

    #     # 调用共轭梯度求解器
    #     dphi, info = cg(A_sparse, bc, x0=dphi, tol=tolerance, maxiter=maxIter, M=M)

    #     if info == 0:
    #         print(f"求解成功收敛 (预处理器: {preconditioner})")
    #     elif info > 0:
    #         print(f"在 {info} 次迭代后达到设定的收敛容限 (预处理器: {preconditioner})")
    #     else:
    #         print(f"求解未能收敛 (预处理器: {preconditioner})")

    #     finalRes = np.linalg.norm(bc - A_sparse @ dphi)
    #     theCoefficients.dphi = dphi
    #     return initRes, finalRes

    # 计算精确解以进行比较（仅在适当规模时）
    if N <= 1000:
        A_dense = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            A_dense[i, i] = 2.0
            if i > 0:
                A_dense[i, i - 1] = -1.0
            if i < N - 1:
                A_dense[i, i + 1] = -1.0
        b_dense = bc
        try:
            x_exact = np.linalg.solve(A_dense, b_dense)
            # print("精确解 x_exact:", x_exact)

            # 运行PCG求解器，使用ILU预处理器
            print("\n使用ILU预处理器进行求解：")
            maxIter = 1000
            tolerance = 1e-6
            relTol = 1e-6
            preconditioner = 'ILU'
            initRes_ilu, finalRes_ilu = slv.cfdSolvePCG(coeffs, maxIter, tolerance, relTol, preconditioner)
            print("初始残差:", initRes_ilu)
            print("最终残差:", finalRes_ilu)
            # print("求解结果 dphi (ILU):", coeffs.dphi)
            # 比较数值解和精确解
            error_ilu = np.linalg.norm(coeffs.dphi - x_exact)
            # print("\n精确解 x_exact:", x_exact)
            print("解的误差 (ILU):", error_ilu)
            if error_ilu < 1e-4:
                print("ILU预处理器测试通过")
            else:
                print("ILU预处理器测试失败")

            # 重置dphi和bc以进行下一次求解
            coeffs.dphi = np.zeros(N, dtype=np.float64)
            # coeffs.bc = bc  # 保持bc不变

            # 运行PCG求解器，使用DIC预处理器
            print("\n使用DIC预处理器进行求解：")
            preconditioner = 'DIC'

            initRes_dic, finalRes_dic = slv.cfdSolvePCG(coeffs, maxIter, tolerance, relTol, preconditioner)

            print("初始残差:", initRes_dic)
            print("最终残差:", finalRes_dic)
            # print("求解结果 dphi (DIC):", coeffs.dphi)
            # 比较数值解和精确解
            error_dic = np.linalg.norm(coeffs.dphi - x_exact)
            # print("\n精确解 x_exact:", x_exact)
            print("解的误差 (DIC):", error_dic)
            if error_dic < 1e-4:
                print("DIC预处理器测试通过")
            else:
                print("DIC预处理器测试失败")


            # 重置dphi和bc以进行下一次求解
            coeffs.dphi = np.zeros(N, dtype=np.float64)
            # coeffs.bc = bc  # 保持bc不变
            # 运行PCG求解器，不使用预处理器
            print("\n不使用预处理器进行求解：")
            preconditioner = 'None'
            initRes_none, finalRes_none = slv.cfdSolvePCG(coeffs, maxIter, tolerance, relTol, preconditioner)
            print("初始残差:", initRes_none)
            print("最终残差:", finalRes_none)
            # print("求解结果 dphi (None):", coeffs.dphi)
            # 比较数值解和精确解
            error_none = np.linalg.norm(coeffs.dphi - x_exact)
            # print("\n精确解 x_exact:", x_exact)
            print("解的误差 (None):", error_none)
            if error_none < 1e-4:
                print("无预处理器测试通过")
            else:
                print("无预处理器测试失败")

        except np.linalg.LinAlgError:
            print("精确解计算失败：矩阵可能太大或病态。")
        pass

# 运行测试函数
if __name__ == "__main__":
    # 设定系统规模为1000
    test_pcg_solver_large_system(N=1000)
