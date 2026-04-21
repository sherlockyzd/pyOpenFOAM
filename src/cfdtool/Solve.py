# import os
import cfdtool.IO as io
import cfdtool.Math as mth
import numpy as np

def cfdSolveEquation(Region,theEquationName, iComponent):
    # foamDict = cfdGetFoamDict;
    solver    = Region.dictionaries.fvSolution['solvers'][theEquationName]['solver']
    maxIter   = Region.dictionaries.fvSolution['solvers'][theEquationName]['maxIter']
    tolerance = Region.dictionaries.fvSolution['solvers'][theEquationName]['tolerance']
    relTol    = Region.dictionaries.fvSolution['solvers'][theEquationName]['relTol']

    if solver in ['PCG','PETSc','GAMG']:
        Region_csr_format(Region)

    if solver=='GAMG':
        # GAMG: 使用 PETSc 的代数多重网格求解器
        from cfdtool.PETScSolver import cfdSolvePETSc, PETSC_AVAILABLE
        if PETSC_AVAILABLE:
            petsc_solver_type = Region.dictionaries.fvSolution['solvers'][theEquationName].get('petsc_solver', 'cg')
            petsc_preconditioner = Region.dictionaries.fvSolution['solvers'][theEquationName].get('preconditioner', 'gamg')
            use_gpu = Region.dictionaries.fvSolution['solvers'][theEquationName].get('use_gpu', False)
            [initRes, finalRes] = cfdSolvePETSc(Region.coefficients, maxIter=maxIter, tolerance=tolerance, relTol=relTol, solver_type=petsc_solver_type, preconditioner=petsc_preconditioner, use_gpu=use_gpu)
            device_info = "GPU" if use_gpu else "CPU"
            # print(f"GAMG/PETSc求解完成({device_info}): {petsc_solver_type} + {petsc_preconditioner}")  # 高频，已静默
        else:
            io.cfdError("GAMG求解需要PETSc支持，请安装 petsc4py 或改用其他求解器")
    elif solver=='smoothSolver':
        smoother = Region.dictionaries.fvSolution['solvers'][theEquationName]['smoother']
        [initRes, finalRes] = cfdSolveAlgebraicSystem(1,theEquationName,Region.coefficients,smoother,maxIter,tolerance,relTol)
    elif solver == 'PCG':
        preconditioner = Region.dictionaries.fvSolution['solvers'][theEquationName]['preconditioner']
        [initRes, finalRes] = cfdSolvePCG(Region.coefficients, maxIter, tolerance, relTol, preconditioner)
    elif solver == 'PETSc':
        # 尝试导入PETSc求解器
        from cfdtool.PETScSolver import cfdSolvePETSc,PETSC_AVAILABLE
        if PETSC_AVAILABLE:
            petsc_solver_type = Region.dictionaries.fvSolution['solvers'][theEquationName].get('petsc_solver', 'gmres')
            petsc_preconditioner = Region.dictionaries.fvSolution['solvers'][theEquationName].get('preconditioner', 'gamg')
            # 检查是否启用GPU加速
            use_gpu = Region.dictionaries.fvSolution['solvers'][theEquationName].get('use_gpu', False)
            # use_gpu = True
            [initRes, finalRes] = cfdSolvePETSc(Region.coefficients,maxIter=maxIter,tolerance=tolerance, relTol=relTol,solver_type=petsc_solver_type,preconditioner=petsc_preconditioner,use_gpu=use_gpu)
            device_info = "GPU" if use_gpu else "CPU"
            # print(f"PETSc求解完成({device_info}): {petsc_solver_type} + {petsc_preconditioner}")  # 高频，已静默
        else:
            io.cfdError("PETSc不可用，回退到PCG求解器")
            # preconditioner = Region.dictionaries.fvSolution['solvers'][theEquationName].get('preconditioner', 'ILU') 
            # [initRes, finalRes] = cfdSolvePCG(Region.coefficients, maxIter, tolerance, relTol, preconditioner)
    else:
        # print('  s not defined', solver)
        io.cfdError(solver+' solver has not beeen defined!!!')


    io.cfdPrintResidualsHeader(theEquationName,tolerance,maxIter,initRes,finalRes)
    #    Store linear solver residuals
    if iComponent != -1:
        Region.assembledPhi[theEquationName].theEquation.initResidual[iComponent]=initRes
        Region.assembledPhi[theEquationName].theEquation.finalResidual[iComponent]=finalRes
    else:
        Region.assembledPhi[theEquationName].theEquation.initResidual=initRes
        Region.assembledPhi[theEquationName].theEquation.finalResidual=finalRes
    
def Region_csr_format(Region):
    if Region.coefficients._sparse_matrix_structure_needs_update:
        if Region.MatrixFormat == 'coo':
            Region.mesh._init_coo_format()
            Region.coefficients._init_coo_format(Region)
        elif Region.MatrixFormat in ['csr','ldu','acnb']:
            Region.mesh._init_csr_format()
            Region.coefficients._init_csr_format(Region)
        else:
            io.cfdError("不支持的矩阵格式")

'''
-----------------------------------------------------------
smooth Solver
-----------------------------------------------------------
'''
def cfdSolveAlgebraicSystem(gridLevel,theEquationName,theCoefficients,smoother = 'DILU',maxIter = 20,tolerance = 1e-6,relTol = 0.1,*args):
    """
    Solve the algebraic system for computational fluid dynamics.
    Args:
        gridLevel (int): The grid level.
        theEquationName (str): The name of the equation.
        theCoefficients (object): The coefficients object.
        smoother (str, optional): The smoother type. Defaults to 'DILU'.
        maxIter (int, optional): The maximum number of iterations. Defaults to 20.
        tolerance (float, optional): The tolerance for convergence. Defaults to 1e-6.
        relTol (float, optional): The relative tolerance for convergence. Defaults to 0.1.
        iComponent (int, optional): The component index. Defaults to -1.
    Returns:
        tuple: A tuple containing the initial residual and final residual.
    """
    # 获取矩阵格式
    matrix_format = theCoefficients.MatrixFormat

    # Compute initial residual
    residualsArray = theCoefficients.cfdComputeResidualsArray()
    initialResidual = mth.cfdResidual(residualsArray)
    finalResidual = initialResidual
    if maxIter == 0:
        return initialResidual, finalResidual

    # 根据矩阵格式选择求解方法
    if matrix_format == 'acnb':
        return _solve_acnb_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual)
    elif matrix_format == 'ldu':
        return _solve_ldu_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual)
    elif matrix_format in ['csr', 'coo']:
        return _solve_sparse_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual)
    else:
        raise ValueError(f"Unsupported matrix format: {matrix_format}")


def _solve_acnb_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual):
    """ACNB格式的求解

    当 sparse_always=True（即 CSR 稀疏矩阵可用）时，使用 scipy 稀疏求解器替代
    纯 Python 逐单元迭代，大幅提升性能。否则回退到原始 ACNB Python 循环。
    """
    # 优先使用 scipy 稀疏求解（如果 CSR 矩阵可用）
    if getattr(theCoefficients, '_sparse_always', False):
        return _solve_sparse_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual)

    ac = theCoefficients.ac
    anb = theCoefficients.anb
    bc = theCoefficients.bc
    cconn = theCoefficients._theCConn
    dphi = theCoefficients.dphi
    # theNumberOfElements = theCoefficients.NumberOfElements
    # rc=bc-theCoefficients_Matrix_multiplication(ac,anb,cconn,dphi)
    # Compute initial residual
    residualsArray = theCoefficients.cfdComputeResidualsArray()
    # initialResidual = mth.cfdResidual(residualsArray)
    finalResidual = initialResidual

    if smoother=='DILU':
        # 使用原始的ILU分解
        #    Factorize Ax=b (Apply incomplete upper lower decomposition)
        dc = cfdFactorizeILU(ac,anb,cconn)
        #    Solve system
        for iter in range(maxIter):
            dphi = cfdSolveILU(ac,anb,residualsArray,dc,cconn,dphi)
            theCoefficients.dphi = dphi
            #    Check if termination criterion satisfied
            residualsArray = theCoefficients.cfdComputeResidualsArray()
            finalResidual = mth.cfdResidual(residualsArray)
            if (finalResidual<relTol*initialResidual) and (finalResidual<tolerance):
                break

    elif smoother in ['SOR', 'GaussSeidel', 'symGaussSeidel']:
        # Solve system 使用原始的SOR/GS方法
        for iter in range(maxIter):
            dphi = cfdSolveSOR(ac,anb,bc,cconn,dphi)
            theCoefficients.dphi = dphi
            #    Check if termination criterion satisfied
            residualsArray = theCoefficients.cfdComputeResidualsArray()
            finalResidual = mth.cfdResidual(residualsArray)
            if (finalResidual<relTol*initialResidual) and (finalResidual<tolerance):
                break
    #    Store
    theCoefficients.dphi = dphi
    return initialResidual,finalResidual
    # pass

def _solve_ldu_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual):
    """LDU格式的求解

    当 sparse_always=True（即 CSR 稀疏矩阵可用）时，使用 scipy 稀疏求解器替代
    纯 Python 逐单元迭代，大幅提升性能。否则回退到原始 LDU Python 循环。
    """
    # 优先使用 scipy 稀疏求解（如果 CSR 矩阵可用）
    if getattr(theCoefficients, '_sparse_always', False):
        return _solve_sparse_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual)

    diag = theCoefficients.Diag  # LDU格式中对角就是ac
    upper = theCoefficients.Upper
    lower = theCoefficients.Lower
    bc = theCoefficients.bc
    dphi = theCoefficients.dphi

    if smoother == 'DILU':
        # LDU格式的优化DILU
        dc = _factorize_ldu_dilu(diag, upper, lower, theCoefficients)
        for iter in range(maxIter):
            dphi = _solve_ldu_dilu(diag, upper, lower, bc, dc, dphi,theCoefficients)
            theCoefficients.dphi = dphi

            residualsArray = theCoefficients.cfdComputeResidualsArray()
            finalResidual = mth.cfdResidual(residualsArray)
            if (finalResidual < relTol * initialResidual) and (finalResidual <tolerance):
                break

        pass  # LDU格式DILU求解完成（已静默）

    elif smoother in ['SOR', 'GaussSeidel', 'symGaussSeidel']:
        # LDU格式的优化GS/SOR
        for iter in range(maxIter):
            dphi = _solve_ldu_sor(diag, upper, lower, bc, dphi, theCoefficients)
            theCoefficients.dphi = dphi

            residualsArray = theCoefficients.cfdComputeResidualsArray()
            finalResidual = mth.cfdResidual(residualsArray)
            if (finalResidual < relTol * initialResidual) and (finalResidual <tolerance):
                break

        # LDU格式SOR求解完成（已静默）

    theCoefficients.dphi = dphi
    return initialResidual, finalResidual

def _solve_sparse_format(theCoefficients, smoother, maxIter, tolerance, relTol, initialResidual):
    """CSR/COO格式的scipy高效求解

    使用 scipy 的 Krylov 子空间方法（BICGSTAB/gmres）配合 ILU 预处理，
    一次调用完成求解，充分利用 Krylov 子空间正交化加速收敛。
    """
    from scipy.sparse.linalg import spilu, LinearOperator, bicgstab, gmres, cg

    # 获取scipy稀疏矩阵（CSR用于Krylov求解，CSC用于spilu预处理）
    A_sparse = theCoefficients.data_sparse_matrix_update()
    A_csr = A_sparse.tocsr() if A_sparse.format != 'csr' else A_sparse
    A_csc = A_sparse.tocsc() if A_sparse.format != 'csc' else A_sparse
    bc = theCoefficients.bc
    dphi = theCoefficients.dphi

    # 构建 ILU 预处理器（使用CSC格式）
    try:
        ilu = spilu(A_csc, drop_tol=1e-4, fill_factor=10)
        M = LinearOperator(shape=A_csr.shape, matvec=lambda x: ilu.solve(x))
    except Exception:
        M = None

    # 选择求解器：DILU 用 bicgstab（动量方程非对称），SOR 用 gmres
    if smoother == 'DILU':
        solver_func = bicgstab
    else:
        solver_func = gmres

    # 容差转换：OpenFOAM tolerance 是绝对容差 ||r||_2 < tolerance
    # scipy 的 tol 是相对容差 ||r||_2 / ||b||_2 < tol
    # 转换：scipy_tol = tolerance / ||b||_2
    # 同时考虑 relTol：scipy_tol = max(relTol, tolerance/||b||_2)
    norm_b = np.linalg.norm(bc)
    if norm_b > 0:
        abs_tol_as_rel = tolerance / norm_b
    else:
        abs_tol_as_rel = tolerance  # b 全零时退化为绝对容差
    scipy_tol = max(relTol, abs_tol_as_rel) if relTol > 0 else abs_tol_as_rel

    # scipy 1.14+ 改用 rtol 代替 tol，兼容两种接口
    try:
        dphi, info = solver_func(A_csr, bc, x0=dphi, rtol=scipy_tol,
                                  maxiter=maxIter, M=M)
    except TypeError:
        # fallback for older scipy versions (参数名为 tol)
        dphi, info = solver_func(A_csr, bc, x0=dphi, tol=scipy_tol,
                                  maxiter=maxIter, M=M)
    # 预处理失败时无预处理重试
    if info != 0 and info > 0:
        try:
            dphi2, info2 = solver_func(A_csr, bc, x0=dphi, rtol=scipy_tol,
                                        maxiter=maxIter)
            if info2 == 0 or info2 < 0:
                dphi, info = dphi2, info2
        except TypeError:
            try:
                dphi2, info2 = solver_func(A_csr, bc, x0=dphi, tol=scipy_tol,
                                            maxiter=maxIter)
                if info2 == 0 or info2 < 0:
                    dphi, info = dphi2, info2
            except Exception:
                pass

    # 更新解
    theCoefficients.dphi = dphi
    # 计算最终残差
    residualsArray = theCoefficients.cfdComputeResidualsArray()
    finalResidual = mth.cfdResidual(residualsArray)

    return initialResidual, finalResidual

def _factorize_ldu_dilu(diag, upper, lower, theCoefficients):
    """
    生成 DILU 的对角近似 D（返回其倒数 dc = 1/D）。
    对每个 owner-neighbour 面，仅用一次“较低编号行 → 较高编号行”的更新；
    若编号相反则对 owner 行做对称更新，避免丢一半面贡献。
    """
    dc = diag.copy()
    # owners = theCoefficients._lowerAddr
    neighbours = theCoefficients._upperAddr
    facesAsOwner = theCoefficients._facesAsOwner

    nC  = len(diag)
    # nIf = len(upper)

    tiny = 1e-30  # 防止除零
    for i in range(nC):                         # 按行递增
        Di = dc[i]
        if Di <= tiny: 
            Di = dc[i] = tiny
        for f in facesAsOwner[i]:
            j = neighbours[f]                   # j > i（owner<neigh）
            dc[j] -= (lower[f] * upper[f]) / Di

    invD = 1.0 / dc
    return invD

def _solve_ldu_dilu(diag, upper, lower, bc, invD, dphi, theCoefficients):
    """
    应用 M^{-1} r，其中 M ≈ L D U（DILU）。
    标准顺序：y = L^{-1} r； y = D^{-1} y； y = U^{-1} y； dphi += y
    L[j,i] = lower[f] / D[i]，U[i,j] = upper[f] / D[j]，D^{-1} = dc。
    """
    # owners = theCoefficients._lowerAddr
    neighbours = theCoefficients._upperAddr
    facesAsOwner = theCoefficients._facesAsOwner
    # nFaces = len(upper)
    nC  = len(diag)

    # 1) residual r = bc - A dphi
    # 建议直接调你已有的残差函数：theCoefficients.cfdComputeResidualsArray()
    r = theCoefficients.cfdComputeResidualsArray().copy()

    # 2) Forward sweep on L (按 i 递增)
    for i in range(nC):
        ri = r[i]
        invDi = invD[i]
        for f in facesAsOwner[i]:
            j = neighbours[f]
            r[j] -= lower[f] * invDi * ri

    # 3) Backward sweep on U (按 i 递减)
    for i in range(nC-1, -1, -1):
        for f in facesAsOwner[i]:
            j = neighbours[f]
            r[i] -= upper[f] * invD[j] * r[j]

    # 4) Diagonal scaling & update
    dphi += invD * r
    return dphi

def _solve_ldu_sor(diag, upper, lower, bc, dphi, theCoefficients):
    """LDU格式的高效SOR求解 - 使用预计算的facesAsOwner和facesAsNeighbor"""
    numberOfElements = len(diag)
    neighbors = theCoefficients._upperAddr
    owners = theCoefficients._lowerAddr
    facesAsOwner = theCoefficients._facesAsOwner
    facesAsNeighbor = theCoefficients._facesAsNeighbour

    # 前向扫描
    for i in range(numberOfElements):
        sigma = bc[i]
        # 作为owner的面（连接到高编号邻居）
        for face in facesAsOwner[i]:
            j = neighbors[face]
            sigma -= upper[face] * dphi[j]
        # 作为neighbor的面（连接到低编号邻居）
        for face in facesAsNeighbor[i]:
            sigma -= lower[face] * dphi[owners[face]]
        dphi[i] = sigma / diag[i]

    # 后向扫描
    for i in range(numberOfElements-1, -1, -1):
        sigma = bc[i]
        # 作为owner的面（连接到高编号邻居）
        for face in facesAsOwner[i]:
            j = neighbors[face]
            sigma -= upper[face] * dphi[j]
        # 作为neighbor的面（连接到低编号邻居）
        for face in facesAsNeighbor[i]:
            sigma -= lower[face] * dphi[owners[face]]
        dphi[i] = sigma / diag[i]

    return dphi

def cfdFactorizeILU(ac, anb, cconn):
    """
    Incomplete Lower Upper (ILU) factorization.
    Args:
        ac (ndarray): Diagonal elements of the matrix A.
        anb (list of lists of floats): Off-diagonal elements (neighbors) of the matrix A.
        bc (ndarray): Right-hand side (source term).
        cconn (list of lists of ints): Connectivity of cells (indices of neighbors).
    
    Returns:
        dc (ndarray): Diagonal elements after ILU factorization.
    """
    numberOfElements = len(ac)

    # Initialize dc and rc with zero
    # Step 1: Copy ac into dc
    dc = np.copy(ac)

    # Step 2: Perform ILU factorization
    for i1 in range(numberOfElements):
        dc[i1] = 1.0 / dc[i1]  # Store the inverse of the diagonal element

        for j1_, jj1 in enumerate(cconn[i1]):
            if jj1 > i1:
                # Find the index where jj1 connects back to i1 in cconn[jj1]
                try:
                    i1_index = cconn[jj1].index(i1)
                except ValueError:
                    print(f'The index for i1 in jj1 was not found for element {i1}')
                    continue

                # Update the diagonal element of dc at jj1
                dc[jj1] -= anb[jj1][i1_index] * dc[i1] * anb[i1][j1_]

    return dc

def cfdSolveILU(ac, anb, r, dc, cconn, dphi):
    """
    Solve Incomplete Lower Upper (ILU) system.
    Args:
        ac (ndarray): Diagonal elements of the matrix A.
        anb (list of lists of floats): Off-diagonal elements (neighbors) of the matrix A.
        bc (ndarray): Right-hand side (source term).
        dc (ndarray): Inverse diagonal elements from ILU factorization.
        rc (ndarray): Residual correction array.
        cconn (list of lists of ints): Connectivity of cells (indices of neighbors).
        dphi (ndarray): Current solution vector.
    
    Returns:
        dphi (ndarray): Updated solution vector after ILU solve.
    """
    numberOfElements = len(ac)
    # Initialize Residuals array
    rc=np.copy(r)

    # Forward Substitution
    for i1 in range(numberOfElements):
        mat1 = dc[i1] * rc[i1]
        i1NbList = cconn[i1]
        for j1_, j1 in enumerate(i1NbList):
            if j1 > i1 and j1 < numberOfElements:
                try:
                    i1_index = cconn[j1].index(i1)
                except ValueError:
                    print('ILU Solver Error: The index for i in element j is not found')
                    continue

                mat2 = anb[j1][i1_index] * mat1
                rc[j1] -= mat2

    # Backward Substitution
    for i1 in range(numberOfElements - 1, -1, -1):
        if i1 < numberOfElements - 1:
            for j1_, j in enumerate(cconn[i1]):
                if j > i1:
                    rc[i1] -= anb[i1][j1_] * rc[j]

        mat1 = dc[i1] * rc[i1]
        rc[i1] = mat1
        dphi[i1] += mat1

    return dphi

def cfdSolveSOR(ac,anb,bc,cconn,dphi):
#   ==========================================================================
#    Routine Description:
#      This functions solves linear system Ax = b using Guass-Seidel method   
#   --------------------------------------------------------------------------
    numberOfElements = len(ac)
    for iElement in range(numberOfElements):
        local_dphi = bc[iElement]
        for iLocalNeighbour,neighbor in enumerate(cconn[iElement]):
            local_dphi -= anb[iElement][iLocalNeighbour]*dphi[neighbor]
        dphi[iElement]  = local_dphi/ac[iElement]

    for iElement in range(numberOfElements-1,-1,-1):  #逆序
        local_dphi = bc[iElement]
        for iLocalNeighbour,neighbor in enumerate(cconn[iElement]):
            local_dphi -= anb[iElement][iLocalNeighbour]*dphi[neighbor]
        dphi[iElement] = local_dphi/ac[iElement]
    
    return dphi

def cfdSetupPreconditioner(A_sparse, preconditioner='ILU'):
    from scipy.sparse.linalg import spilu,LinearOperator
    # Setup preconditioner
    if preconditioner == 'ILU':
        # Compute incomplete LU factorization
        try:
            ilu = spilu(A_sparse)
            return LinearOperator(shape=A_sparse.shape, matvec=lambda x: ilu.solve(x))
        except Exception as e:
            raise RuntimeError(f"ILU 分解失败: {e}. 检查矩阵是否满足对称正定条件")
    elif preconditioner == 'Jacobi':
        # Compute Jacobi preconditioner
        # Ensure diagonal entries are positive
        diag_A = A_sparse.diagonal()
        if np.any(diag_A == 0):
            raise RuntimeError("Jacobi 需要非零对角")
        inv_diag = 1.0 / diag_A
        # Create a linear operator for the preconditioner
        return LinearOperator(shape=A_sparse.shape, matvec=lambda x: inv_diag*x)
    elif preconditioner == 'DIC':
        # DIC: 对称矩阵 → 使用 ILU(0) 作为近似
        # scipy spilu 等价于不完全 LU，对于 SPD 矩阵退化为不完全 Cholesky
        try:
            A_csc = A_sparse.tocsc() if A_sparse.format != 'csc' else A_sparse
            ilu = spilu(A_csc, drop_tol=1e-4, fill_factor=10)
            return LinearOperator(shape=A_sparse.shape, matvec=lambda x: ilu.solve(x))
        except Exception as e:
            import warnings
            warnings.warn(f"DIC(ILU) 分解失败: {e}，回退到 Jacobi")
            diag_A = A_sparse.diagonal()
            inv_diag = 1.0 / np.where(diag_A != 0, diag_A, 1e-30)
            return LinearOperator(shape=A_sparse.shape, matvec=lambda x: inv_diag * x)

    elif preconditioner == 'DILU':
        # DILU: 非对称矩阵（如动量方程）→ 使用 ILU
        try:
            ilu = spilu(A_sparse, drop_tol=1e-4, fill_factor=10)
            return LinearOperator(shape=A_sparse.shape, matvec=lambda x: ilu.solve(x))
        except Exception as e:
            import warnings
            warnings.warn(f"DILU(ILU) 分解失败: {e}，回退到 Jacobi")
            diag_A = A_sparse.diagonal()
            inv_diag = 1.0 / np.where(diag_A != 0, diag_A, 1e-30)
            return LinearOperator(shape=A_sparse.shape, matvec=lambda x: inv_diag * x)

    # elif preconditioner == 'DIC':
    #     # # Compute DIC preconditioner
    #     diag_A = A_sparse.diagonal()
    #     # Ensure diagonal entries are positive
    #     if np.any(diag_A <= 0):
    #         raise RuntimeError("DILU预处理器要求矩阵对角元素为正")
    #     M_diag = 1.0 / np.sqrt(diag_A)
    #     # Create a linear operator for the preconditioner
    #     def M_x(x):
    #         return M_diag * x
    #     M = LinearOperator(shape=A_sparse.shape, matvec=M_x)
        # from pyamg import smoothed_aggregation_solver
        # # 使用pyamg的SMG作为预处理器示例
        # ml = smoothed_aggregation_solver(A_sparse)
        # M = ml.aspreconditioner()
    elif preconditioner == 'None':
        return None  # No preconditioning
    else:
        raise ValueError(f"Unknown preconditioner: {preconditioner}")
    
'''
-----------------------------------------------------------
PCG Solver
-----------------------------------------------------------
'''
# 更新PCG求解器以支持多种预处理器
def cfdSolvePCG(theCoefficients, maxIter, tolerance, relTol,preconditioner='ILU'):
    # A_sparse=theCoefficients.assemble_sparse_matrix()
    A_sparse=theCoefficients.data_sparse_matrix_update()
    # 注释掉 verify_matrix_properties：每次都算 Frobenius 范数太贵
    # theCoefficients.verify_matrix_properties()

    r = theCoefficients.cfdComputeResidualsArray()
    initRes = mth.cfdResidual(r)
    if initRes < tolerance or maxIter == 0:
        return initRes, initRes
    
    dphi = np.copy(theCoefficients.dphi)  # Initial guess
    bc = theCoefficients.bc

    # 使用 spilu 预处理器
    M = cfdSetupPreconditioner(A_sparse, preconditioner)

    # 容差转换：OpenFOAM tolerance 是绝对容差 ||r||_2 < tolerance
    # scipy 的 rtol 是相对容差 ||r||_2 / ||b||_2 < rtol
    norm_b = np.linalg.norm(bc)
    if norm_b > 0:
        abs_tol_as_rel = tolerance / norm_b
    else:
        abs_tol_as_rel = tolerance
    scipy_rtol = max(relTol, abs_tol_as_rel) if relTol > 0 else abs_tol_as_rel

    from scipy.sparse.linalg import cg, gmres
    try:
        dphi, info = cg(A_sparse, bc, x0=dphi, rtol=scipy_rtol,
                        maxiter=maxIter, M=M)
    except TypeError:
        # fallback for older scipy versions (tol 旧版也是相对容差)
        dphi, info = cg(A_sparse, bc, x0=dphi, tol=scipy_rtol,
                        maxiter=maxIter, M=M)
    # cg 不收敛时 fallback 到 gmres（适用于非对称矩阵）
    if info < 0:
        try:
            dphi, info = gmres(A_sparse, bc, x0=dphi, rtol=scipy_rtol,
                              maxiter=maxIter, M=M)
        except Exception:
            pass
    if info == 0:
        pass  # 静默成功
    elif info > 0:
        print(f"在 {info} 次迭代后达到设定的收敛容限")
    else:
        print(f"求解未能收敛，info = {info}. 请检查矩阵的正定性或调整预处理器")
    finalRes = mth.cfdResidual(bc - A_sparse @ dphi)

    # 将最终解更新到 theCoefficients.dphi 中
    theCoefficients.dphi = dphi
    return initRes, finalRes

def preconditioner_solve(r,theCoefficients,preconditioner):
    #Y. Ye, H. Guo, B. Wang, P. Wang, D. Chen and F. Li, "Coupled Incomplete Cholesky and Jacobi Preconditioned Conjugate Gradient on the New Generation of Sunway Many-Core Architecture," in IEEE Transactions on Computers, vol. 72, no. 11, pp. 3326-3339, 1 Nov. 2023, https://doi.org/10.1109/TC.2023.3296884.
    ac = theCoefficients.ac
    anb = theCoefficients.anb
    cconn = theCoefficients.theCConn
    if preconditioner == 'DIC':
        dc = cfdFactorizeDIC(ac, anb, cconn)
        z  = r*dc*dc
        # z = cfdSolveDIC(ac, anb, r, dc, cconn, np.zeros_like(r))
    elif preconditioner == 'ILU':
        dc = cfdFactorizeILU(ac, anb, cconn)
        z = cfdSolveILU(ac, anb, r, dc, cconn, np.zeros_like(r))
    elif preconditioner == 'Jacobi':
        z = jacobiPreconditioner(theCoefficients, r)
    else:
        raise ValueError(f"未知的预处理器: {preconditioner}")
    return z

# 更新PCG求解器以支持多种预处理器
# def cfdSolvePCG1(theCoefficients, maxIter, tolerance, relTol,preconditioner='ILU'):
#     ac = theCoefficients.ac
#     anb = theCoefficients.anb
#     cconn = theCoefficients.theCConn
#     dphi = np.copy(theCoefficients.dphi)  # Initial guess
#     # Compute initial residual
#     r = theCoefficients.cfdComputeResidualsArray()
#     initRes = mth.cfdResidual(r)
#     if initRes < tolerance or maxIter == 0:
#         return initRes, initRes

#     # Apply the preconditioner to the initial residual
#     z=preconditioner_solve(r,theCoefficients,preconditioner)  # M_inv should be set in theCoefficients
#     d = np.copy(z)
#     rz_old = np.dot(r, z)

#     for iter in range(maxIter):
#         # Compute Ad = A * d
#         Ad=theCoefficients.theCoefficients_Matrix_multiplication(d)

#         # Calculate step size alpha
#         alpha = rz_old / (np.dot(d, Ad)+1e-20)

#         # Update solution
#         dphi += alpha * d

#         # Calculate new residual
#         r -= alpha * Ad

#         # Check for convergence
#         finalRes = mth.cfdResidual(r)
#         if finalRes < max(relTol * initRes, tolerance):
#             break
#         # Apply the preconditioner
#         z=preconditioner_solve(r,theCoefficients,preconditioner)

#         # Calculate beta
#         rz_new = np.dot(r, z)
#         beta = rz_new / rz_old

#         # Update search direction
#         d = z + beta * d

#         # Update rz_old for next iteration
#         rz_old = rz_new

#     # 将最终解更新到 theCoefficients.dphi 中
#     theCoefficients.dphi = dphi
#     return initRes, finalRes

def cfdSolvePCG0(theCoefficients, maxIter, tolerance, relTol, preconditioner='ILU'):
    # ac = theCoefficients.ac
    # anb = theCoefficients.anb
    # cconn = theCoefficients.theCConn
    dphi = theCoefficients.dphi
    # theNumberOfElements = theCoefficients.NumberOfElements
    # 初始残差计算
    r = theCoefficients.cfdComputeResidualsArray()
    initRes = mth.cfdResidual(r)
    # 选择预处理器并初始化 z
    #Y. Ye, H. Guo, B. Wang, P. Wang, D. Chen and F. Li, "Coupled Incomplete Cholesky and Jacobi Preconditioned Conjugate Gradient on the New Generation of Sunway Many-Core Architecture," in IEEE Transactions on Computers, vol. 72, no. 11, pp. 3326-3339, 1 Nov. 2023, https://doi.org/10.1109/TC.2023.3296884.
    z=preconditioner_solve(r,theCoefficients,preconditioner)
    p = np.copy(z)
    rz_old = np.dot(r, z)
    finalRes = initRes
    if maxIter == 0:
        return initRes, finalRes

    for k in range(maxIter):
        # 矩阵向量乘积 A * p
        Ap = theCoefficients.theCoefficients_Matrix_multiplication(p)
        alpha = rz_old / (np.dot(p, Ap)+1e-20)
        dphi = dphi + alpha * p
        r = r - alpha * Ap

        finalRes = mth.cfdResidual(r)
        if finalRes < max(relTol * initRes, tolerance):
            break
        # 预处理应用
        z=preconditioner_solve(r,theCoefficients,preconditioner)
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    # 将最终解更新到 theCoefficients.dphi 中
    theCoefficients.dphi = dphi
    return initRes, finalRes



def jacobiPreconditioner(theCoefficients, r, max_iter=10, tol=1e-6):
    """
    使用ac, anb和cconn数据结构进行Jacobi预处理。
    Args:
        ac (ndarray): 对角元素数组。
        anb (list of lists): 非对角元素，列表的列表形式。
        cconn (list of lists): 每个节点的连接关系，存储相邻节点的索引。
        r (ndarray): 残差向量。
        max_iter (int): Jacobi迭代的最大次数。
        tol (float): 收敛准则的容差。
    
    Returns:
        z (ndarray): 经过Jacobi预处理后的向量。
    """
    # 提取对角元素的倒数
    D_inv = 1.0 / theCoefficients.ac
    
    # 初始计算 z_0 = D^-1 * r
    z = D_inv * r
    
    # Jacobi 迭代更新
    for _ in range(max_iter):
        # 计算 Az_k
        Az =theCoefficients.theCoefficients_Matrix_multiplication(z)
        
        # 更新 z_k+1
        z_new = z + D_inv * (r - Az)
        
        # 检查收敛条件
        if np.linalg.norm(z_new - z) < tol:
            break
        
        z = z_new
    
    return z


# DIC预处理器的实现
def cfdFactorizeDIC(ac, anb, cconn):
    """
    对角不完全 Cholesky (DIC) 因子化。

    参数：
        ac (ndarray): 矩阵 A 的对角元素。
        anb (list of lists of floats): 矩阵 A 的非对角（邻接）元素。
        cconn (list of lists of ints): 单元的连通性（邻居的索引）。

    返回：
        dc (ndarray): 调整后对角元素平方根的倒数。
    """
    numberOfElements = len(ac)
    dc = np.zeros(numberOfElements)

    for i in range(numberOfElements):
        sum_terms = 0.0
        for j_, neighbor in enumerate(cconn[i]):
            if neighbor < i:
                # 使用先前计算的 dc[neighbor]
                sum_terms += (anb[i][j_] * dc[neighbor]) ** 2
        diag_element = ac[i] - sum_terms
        if diag_element <= 0:
            diag_element = 1e-10  # 防止对角元素为零或负值
        dc[i] = 1.0 / np.sqrt(diag_element)

    return dc


def cfdSolveDIC(ac, anb, r, dc, cconn, dphi):
    """
    对残差向量应用 DIC 预处理器。

    参数：
        ac (ndarray): 矩阵 A 的对角元素。
        anb (list of lists of floats): 矩阵 A 的非对角（邻接）元素。
        r (ndarray): 残差向量。
        dc (ndarray): 调整后对角元素平方根的倒数。
        cconn (list of lists of ints): 单元的连通性（邻居的索引）。
        dphi (ndarray): 待更新的解向量。

    返回：
        dphi (ndarray): 应用 DIC 后更新的解向量。
    """
    numberOfElements = len(ac)
    rc = np.copy(r)

    # 前向替换
    for i in range(numberOfElements):
        sum_terms = 0.0
        for j_, j in enumerate(cconn[i]):
            if j < i:
                sum_terms += anb[i][j_] * rc[j]
        rc[i] = (rc[i] - sum_terms) * dc[i]

    # 后向替换
    for i in range(numberOfElements -1, -1, -1):
        rc[i] *= dc[i]
        sum_terms = 0.0
        for j_, j in enumerate(cconn[i]):
            if j > i:
                sum_terms += anb[i][j_] * rc[j]
        rc[i] -= sum_terms
        dphi[i] += rc[i]

    return dphi



def cfdSolveILU_orig(theCoefficients,dc,rc):
#   ==========================================================================
#    Routine Description:
#      Solve Incomplete Lower Upper system
#   --------------------------------------------------------------------------
#    ILU Iterate
    # ac = theCoefficients.ac
    anb = theCoefficients.anb
    cconn = theCoefficients.theCConn
    bc = theCoefficients.bc
    numberOfElements=theCoefficients.NumberOfElements
    dphi = np.copy(theCoefficients.dphi)
    #  Update Residuals array
    rc = bc - theCoefficients.theCoefficients_Matrix_multiplication(dphi)
    # for iElement in range(numberOfElements):
    #     conn = cconn[iElement]
    #     res = -ac[iElement]*dphi[iElement] + bc[iElement]
    #     theNumberOfNeighbours = len(conn)

    #     for iLocalNeighbour in range(theNumberOfNeighbours):
    #         #    Get the neighbour cell index
    #         j    = conn[iLocalNeighbour]
    #         res -=  anb[iElement][iLocalNeighbour]*dphi[j]
    #     rc[iElement]= res

    #    Forward Substitution
    for i1 in range(numberOfElements):
        mat1 = dc[i1]*rc[i1]
        i1NNb = len(cconn[i1])
        i1NbList = cconn[i1]
        #    Loop over neighbours of i
        j1_ = 0
        while j1_+1 <= i1NNb:
            j1_ +=  1
            j1   = i1NbList[j1_]
            #    For all neighbour j > i do
            if j1 > i1 and j1<=numberOfElements:
                j1NbList = cconn[j1]
                j1NNB = len(j1NbList)
                i1_= 0
                k = 0
                #    Get A[j][i]
                while i1_+1<=j1NNB  and k != i1 :
                    i1_ += 1
                    k    = j1NbList[i1_]
                #    Compute rc
                if k == i1:
                    mat2    =  anb[j1][i1_]*mat1
                    rc[j1] -= mat2
                else:
                    print('ILU Solver Error The index for i  in element j  is not found \n')
    #    Backward substitution
    for i1 in range(numberOfElements-1, -1, -1):
        #    Compute rc
        if i1<numberOfElements:
            i1NBList = cconn[i1]
            i1NNb = len(i1NBList)
            j1_ = 0
            #    Loop over neighbours of i
            while j1_+1 <= i1NNb:
                j1_ += 1
                j    = i1NBList[j1_]
                if j>i1:
                    rc[i1] -= anb[i1][j1_]*rc[j]
        #    Compute product D[i]*R[i]
        mat1 = dc[i1]*rc[i1]
        rc[i1] = mat1
        #    Update dphi
        dphi[i1] +=  mat1

    return dphi


