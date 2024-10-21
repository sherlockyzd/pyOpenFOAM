import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
import os
import sys

# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the path to the pyFVM directory
src_path = os.path.abspath(os.path.join(current_dir,'..', 'src'))
print(f"src 路径: {src_path}")  # 打印路径，确保正确
sys.path.insert(0, src_path)

from pyFVM import Coefficients as coefficients

# 定义一个简单的 Mesh 类
class Mesh:
    def __init__(self, elementNeighbours):
        self.elementNeighbours = elementNeighbours

# 定义一个简单的 Region 类
class Region:
    def __init__(self, mesh):
        self.mesh = mesh

# 定义 Coefficients 类（已在前面给出）

# 测试函数
def test_pcg_solver():
    # 定义一个简单的 3 元素 1D 问题
    theCConn = [[1], [0, 2], [1]]  # 邻接关系

    NumberOfElements = 3

    ac = np.array([2, 2, 2], dtype=np.float32)
    anb = [
        np.array([-1], dtype=np.float32),
        np.array([-1, -1], dtype=np.float32),
        np.array([-1], dtype=np.float32)
    ]
    bc = np.array([1, 2, 3], dtype=np.float32)

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

    # 运行 PCG 求解器
    maxIter = 1000
    tolerance = 1e-6
    relTol = 1e-6
    preconditioner = 'None'  # 对于小问题，不使用预处理器

    def cfdSolvePCG(theCoefficients, maxIter, tolerance, relTol, preconditioner='ILU'):
        A_sparse = theCoefficients.assemble_sparse_matrix()
        theCoefficients.verify_matrix_properties()

        r = theCoefficients.cfdComputeResidualsArray()
        initRes = np.linalg.norm(r)

        if initRes < tolerance or maxIter == 0:
            return initRes, initRes

        dphi = np.copy(theCoefficients.dphi)  # 初始猜测
        bc = theCoefficients.bc

        # 设置预处理器
        if preconditioner == 'ILU':
            try:
                ilu = spilu(A_sparse)
                M_x = lambda x: ilu.solve(x)
                M = LinearOperator(shape=A_sparse.shape, matvec=M_x)
            except Exception as e:
                raise RuntimeError(f"ILU 分解失败: {e}")
        elif preconditioner == 'DIC':
            diag_A = A_sparse.diagonal().copy()
            diag_A[diag_A <= 0] = 1e-10
            M_diag = 1.0 / np.sqrt(diag_A)
            M = LinearOperator(shape=A_sparse.shape, matvec=lambda x: M_diag * x)
        elif preconditioner == 'None':
            M = None  # 不使用预处理器
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner}")

        # 调用共轭梯度求解器
        dphi, info = cg(A_sparse, bc, x0=dphi, tol=tolerance, maxiter=maxIter, M=M)

        if info == 0:
            print("求解成功收敛")
        elif info > 0:
            print(f"在 {info} 次迭代后达到设定的收敛容限")
        else:
            print("求解未能收敛")

        finalRes = np.linalg.norm(bc - A_sparse @ dphi)
        theCoefficients.dphi = dphi
        return initRes, finalRes



    initRes, finalRes = cfdSolvePCG(coeffs, maxIter, tolerance, relTol, preconditioner)

    print("初始残差:", initRes)
    print("最终残差:", finalRes)
    print("求解结果 dphi:", coeffs.dphi)

    # 计算精确解以进行比较
    A_dense = np.array([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ], dtype=np.float32)
    b_dense = bc
    x_exact = np.linalg.solve(A_dense, b_dense)
    print("精确解 x_exact:", x_exact)

    # 比较数值解和精确解
    error = np.linalg.norm(coeffs.dphi - x_exact)
    print("解的误差:", error)
    if error < 1e-6:
        print("测试通过")
    else:
        print("测试失败")
        
    pass

# 运行测试函数
test_pcg_solver()
