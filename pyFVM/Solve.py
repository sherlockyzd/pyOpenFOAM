# import os
import pyFVM.IO as io
import numpy as np

def cfdSolveEquation(Region,theEquationName, iComponent):
    # foamDict = cfdGetFoamDict;
    solver    = Region.dictionaries.fvSolution['solvers'][theEquationName]['solver']
    maxIter   = Region.dictionaries.fvSolution['solvers'][theEquationName]['maxIter']
    tolerance = Region.dictionaries.fvSolution['solvers'][theEquationName]['tolerance']
    relTol    = Region.dictionaries.fvSolution['solvers'][theEquationName]['relTol']

    if solver=='GAMG':
        # % Get GAMG settings
        preconditioner = Region.dictionaries.fvSolution['solvers'].get(theEquationName, {}).get('preconditioner', None)
        nPreSweeps     = Region.dictionaries.fvSolution['solvers'].get(theEquationName, {}).get('nPreSweeps', None)
        nPostSweeps    = Region.dictionaries.fvSolution['solvers'].get(theEquationName, {}).get('nPostSweeps', None)
        nFinestSweeps  = Region.dictionaries.fvSolution['solvers'].get(theEquationName, {}).get('nFinestSweeps', None)
        [initRes, finalRes] = cfdApplyAMG(preconditioner,maxIter,tolerance,relTol,nPreSweeps,nPostSweeps,nFinestSweeps)
    elif solver=='smoothSolver':
        smoother = Region.dictionaries.fvSolution['solvers'][theEquationName]['smoother']
        [initRes, finalRes] = cfdSolveAlgebraicSystem(1,theEquationName,Region.coefficients,smoother,maxIter,tolerance,relTol)
    elif solver == 'PCG':
        preconditioner = Region.dictionaries.fvSolution['solvers'][theEquationName]['preconditioner']
        [initRes, finalRes] = cfdSolvePCG(Region.coefficients, maxIter, tolerance, relTol, preconditioner)
    else:
        # print('%s not defined', solver)
        io.cfdError(solver+' solver has not beeen defined!!!')
    io.cfdPrintResidualsHeader(theEquationName,tolerance,maxIter,initRes,finalRes)
    # % Store linear solver residuals
    if iComponent != -1:
        Region.assembledPhi[theEquationName].theEquation.initResidual[iComponent]=initRes
        Region.assembledPhi[theEquationName].theEquation.finalResidual[iComponent]=finalRes
    else:
        Region.assembledPhi[theEquationName].theEquation.initResidual=initRes
        Region.assembledPhi[theEquationName].theEquation.finalResidual=finalRes



'''
-----------------------------------------------------------
PCG Solver
-----------------------------------------------------------
'''
# 更新PCG求解器以支持多种预处理器
def cfdSolvePCG(theCoefficients, maxIter, tolerance, relTol, preconditioner='ILU'):
    ac = theCoefficients.ac
    anb = theCoefficients.anb
    bc = theCoefficients.bc
    cconn = theCoefficients.theCConn
    dphi = theCoefficients.dphi
    theNumberOfElements = theCoefficients.NumberOfElements

    # 初始残差计算
    r = cfdComputeResidualsArray(theCoefficients)
    initRes = sum(abs(r)) / theNumberOfElements

    # 选择预处理器
    # preconditioner=choose_preconditioner(ac, anb, cconn)
    # 选择预处理器并初始化 z
    #Y. Ye, H. Guo, B. Wang, P. Wang, D. Chen and F. Li, "Coupled Incomplete Cholesky and Jacobi Preconditioned Conjugate Gradient on the New Generation of Sunway Many-Core Architecture," in IEEE Transactions on Computers, vol. 72, no. 11, pp. 3326-3339, 1 Nov. 2023, https://doi.org/10.1109/TC.2023.3296884.
    if preconditioner == 'DIC':
        dc = cfdFactorizeDIC(ac, anb, cconn)
        z = cfdSolveDIC(ac, anb, r, dc, cconn, np.zeros_like(r))
    elif preconditioner == 'ILU':
        dc = cfdFactorizeILU(ac, anb, cconn)
        z = cfdSolveILU(ac, anb, r, dc, cconn, np.zeros_like(r))
    elif preconditioner == 'Jacobi':
        z = jacobiPreconditioner(ac, anb, cconn, r)
    else:
        raise ValueError(f"未知的预处理器: {preconditioner}")
    # z=r/ac


    p = np.copy(z)
    rz_old = np.dot(r, z)
    finalRes = initRes

    if maxIter == 0:
        return initRes, finalRes

    for k in range(maxIter):
        # 矩阵向量乘积 A * p
        Ap = np.zeros(theNumberOfElements)
        for i in range(theNumberOfElements):
            Ap[i] = ac[i] * p[i]
            for j in range(len(cconn[i])):
                Ap[i] += anb[i][j] * p[cconn[i][j]]

        alpha = rz_old / (np.dot(p, Ap)+1e-20)
        dphi = dphi + alpha * p
        r = r - alpha * Ap

        finalRes = sum(abs(r)) / theNumberOfElements
        if finalRes < max(relTol * initRes, tolerance):
            break

        # 预处理应用
        if k % 5 == 0:
            if preconditioner == 'DIC':
                z = r * dc  # 使用选定预处理器的残差
                z = cfdSolveDIC(ac, anb, r, dc, cconn, z)
            elif preconditioner == 'ILU':
                z = r * dc  # 使用选定预处理器的残差
                z = cfdSolveILU(ac, anb, r, dc, cconn, z)
            elif preconditioner == 'Jacobi':
                z = jacobiPreconditioner(ac, anb, cconn, r)
            else:
                raise ValueError(f"未知的预处理器: {preconditioner}")


        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    # 将最终解更新到 theCoefficients.dphi 中
    theCoefficients.dphi = dphi
    return initRes, finalRes



def jacobiPreconditioner(ac, anb, cconn, r, max_iter=10, tol=1e-6):
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
    D_inv = 1.0 / ac
    
    # 初始计算 z_0 = D^-1 * r
    z = D_inv * r
    
    # Jacobi 迭代更新
    for _ in range(max_iter):
        # 计算 Az_k
        Az = np.zeros_like(r)
        for i in range(len(ac)):
            Az[i] = ac[i] * z[i]
            for j, neighbor in enumerate(cconn[i]):
                Az[i] += anb[i][j] * z[neighbor]
        
        # 更新 z_k+1
        z_new = z + D_inv * (r - Az)
        
        # 检查收敛条件
        if np.linalg.norm(z_new - z) < tol:
            break
        
        z = z_new
    
    return z

# 示例用法：
# ac 是对角元素数组
# anb 是非对角元素列表的列表
# cconn 是连接关系的列表的列表
# r 是残差向量
# z = jacobiPreconditioner(ac, anb, cconn, r)

# DIC预处理器的实现
def cfdFactorizeDIC(ac, anb, cconn):
    """
    Diagonal Incomplete Cholesky (DIC) factorization storing inverse of diagonal elements.
    Args:
        ac (ndarray): Diagonal elements of the matrix A.
        anb (ndarray): Off-diagonal elements (neighbors) of the matrix A.
        cconn (list of lists): Connectivity of cells (indices of neighbors).
    
    Returns:
        dc (ndarray): Inverse of the DIC factorized diagonal.
    """
    numberOfElements = len(ac)
    dc = np.zeros(numberOfElements)

    for i in range(numberOfElements):
        sum_terms = 0.0
        for j, neighbor in enumerate(cconn[i]):
            if neighbor < i:  # Only consider lower triangular part
                # 注意此处的修正项计算已经调整为使用倒数
                sum_terms += (anb[i][j] ** 2) * dc[neighbor]
        
        diag_element = ac[i] - sum_terms
        if diag_element <= 0:
            diag_element = 1e-10
            # raise ValueError(f"DIC factorization failed at element {i}, non-positive diagonal element.")

        dc[i] = 1.0 / diag_element  # 存储的是对角元素的倒数

    return dc

def cfdSolveDIC(ac, anb, bc, dc, cconn, dphi):
    """
    Solve system using Diagonal Incomplete Cholesky (DIC) factorization.
    Args:
        ac (ndarray): Diagonal elements of the matrix A.
        anb (list of lists of floats): Off-diagonal elements (neighbors) of the matrix A.
        bc (ndarray): Right-hand side (source term).
        dc (ndarray): Factorized diagonal elements from DIC factorization.
        cconn (list of lists of ints): Connectivity of cells (indices of neighbors).
        dphi (ndarray): Current solution vector.
    
    Returns:
        dphi (ndarray): Updated solution vector after DIC solve.
    """
    numberOfElements = len(ac)
    rc = np.zeros_like(dphi)

    # Step 1: Update Residuals array
    for iElement in range(numberOfElements):
        conn = cconn[iElement]
        res = -ac[iElement] * dphi[iElement] + bc[iElement]

        for iLocalNeighbour, j in enumerate(conn):
            res -= anb[iElement][iLocalNeighbour] * dphi[j]

        rc[iElement] = res

    # Step 2: Forward Substitution
    for i1 in range(numberOfElements):
        mat1 = rc[i1] / dc[i1]
        i1NbList = cconn[i1]

        for j1_, j1 in enumerate(i1NbList):
            if j1 > i1:
                try:
                    i1_index = cconn[j1].index(i1)
                except ValueError:
                    print('DIC Solver Error: The index for i1 in element j is not found')
                    continue

                rc[j1] -= anb[j1][i1_index] * mat1

    # Step 3: Backward Substitution
    for i1 in range(numberOfElements - 1, -1, -1):
        rc[i1] /= dc[i1]
        dphi[i1] += rc[i1]

        for j1_, j in enumerate(cconn[i1]):
            if j > i1:
                rc[j] -= anb[i1][j1_] * rc[i1]

    return dphi


# def cfdFactorizeILU_orig(ac,anb,bc,cconn):
# # %==========================================================================
# # % Routine Description:
# # %   Incomplete Lower Upper Factorization
# # %--------------------------------------------------------------------------
#     numberOfElements = len(ac)
#     dc = np.zeros_like(ac)
#     rc = np.zeros_like(ac)
#     for i1 in range(numberOfElements):
#         dc[i1] = ac[i1]

#     for i1 in range(numberOfElements):
#         dc[i1] = 1.0/dc[i1]# 这里将 dc 中的对角元素取倒数
#         rc[i1] = bc[i1]
#         i1NbList = cconn[i1]
#         i1NNb = len(i1NbList)
#         if i1 != numberOfElements-1:
#             # % loop over neighbours of iElement
#             j1_ = 1
#             while j1_<=i1NNb:
#                 jj1 = i1NbList(j1_)
#                 # % for all neighbour j > i do
#                 if jj1>i1 and jj1<=numberOfElements:
#                     j1NbList = cconn[jj1]
#                     j1NNb = len(j1NbList)
#                     i1_= 0
#                     k1 = -1
#                     # % find _i index to get A[j][_i]
#                     while i1_<=j1NNb and k1 != i1:
#                         i1_ = i1_ + 1
#                         k1 = j1NbList(i1_)
#                     # % Compute A[j][i]*D[i]*A[i][j]
#                     if k1 == i1:
#                         dc[jj1] -= anb[jj1][i1_]*dc[i1]*anb[i1][j1_]
#                     else:
#                         print('the index for i in j is not found')
#                 j1_ = j1_ + 1
#     return dc,rc


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
        rc (ndarray): Residual correction array initialized to bc.
    """
    numberOfElements = len(ac)

    # Initialize dc and rc with zeros
    dc = np.zeros_like(ac)
    # rc = np.copy(bc)  # Initialize rc with bc values

    # Step 1: Copy ac into dc
    dc[:] = ac[:]

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

# def cfdFactorizeDIC(ac, anb, cconn):
#     """
#     Diagonal Incomplete Cholesky (DIC) factorization.
#     Args:
#         ac (ndarray): Diagonal elements of the matrix A.
#         anb (ndarray): Off-diagonal elements (neighbors) of the matrix A.
#         cconn (list of lists): Connectivity of cells (indices of neighbors).
    
#     Returns:
#         dc (ndarray): DIC factorized diagonal.
#     """
#     numberOfElements = len(ac)
#     dc = np.zeros(numberOfElements)

#     for i in range(numberOfElements):
#         sum_terms = 0.0
#         for j, neighbor in enumerate(cconn[i]):
#             if neighbor < i:  # Only consider lower triangular part
#                 sum_terms += (anb[i][j] ** 2) / dc[neighbor]
        
#         dc[i] = ac[i] - sum_terms
#         if dc[i] <= 0:
#             raise ValueError(f"DIC factorization failed at element {i}, non-positive diagonal element.")
#         dc[i] = np.sqrt(dc[i])
#     return dc

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

    ac = theCoefficients.ac
    anb = theCoefficients.anb
    bc = theCoefficients.bc
    cconn = theCoefficients.theCConn
    dphi = theCoefficients.dphi
    theNumberOfElements = theCoefficients.NumberOfElements

    # % Compute initial residual
    residualsArray = cfdComputeResidualsArray(theCoefficients)
    # residualsArray = np.zeros(theNumberOfElements)
    # for iElement in range(theNumberOfElements):   
    #     residualsArray[iElement] = bc[iElement] - ac[iElement]*dphi[iElement]
    #     for nNeighbour in range(len(cconn[iElement])):
    #         iNeighbour = cconn[iElement][nNeighbour]
    #         residualsArray[iElement] -= anb[iElement][nNeighbour]*dphi[iNeighbour]
    initialResidual = sum(abs(residualsArray))/theNumberOfElements
    finalResidual = initialResidual

    if maxIter==0:
        return
    
    if smoother=='DILU':
        # % Factorize Ax=b (Apply incomplete upper lower decomposition)
        dc = cfdFactorizeILU(ac,anb,cconn)
        # % Solve system
        for iter in range(maxIter):
            dphi = cfdSolveILU(ac,anb,bc,dc,cconn,dphi)
            theCoefficients.dphi = dphi
            # % Check if termination criterion satisfied
            residualsArray = cfdComputeResidualsArray(theCoefficients)
            finalResidual = sum(abs(residualsArray))/theNumberOfElements
            # if iComponent != -1:
            #     io.cfdPrintInteration(theEquationName,iter,iComponent)
            # else:
            #     io.cfdPrintInteration(theEquationName,iter)
            if (finalResidual<relTol*initialResidual) and (finalResidual<tolerance):
                break

    elif smoother=='SOR' or smoother=='GaussSeidel'  or smoother=='symGaussSeidel':
        # % Solve system
        for iter in range(maxIter):
            dphi = cfdSolveSOR(ac,anb,bc,cconn,dphi)
            theCoefficients.dphi = dphi
            # % Check if termination criterion satisfied
            residualsArray = cfdComputeResidualsArray(theCoefficients)
            finalResidual = sum(abs(residualsArray))/theNumberOfElements
            # if iComponent != -1:
            #     io.cfdPrintInteration(theEquationName,iter,iComponent)
            # else:
            #     io.cfdPrintInteration(theEquationName,iter)
            if (finalResidual<relTol*initialResidual) and (finalResidual<tolerance):
                break
    # % Store
    theCoefficients.dphi = dphi
    return initialResidual,finalResidual
    # pass

def cfdComputeResidualsArray(theCoefficients):
    # ac = theCoefficients.ac
    # anb = theCoefficients.anb
    # bc = theCoefficients.bc
    # cconn = theCoefficients.theCConn
    # dphi = theCoefficients.dphi
    theNumberOfElements = theCoefficients.NumberOfElements
    residualsArray = np.zeros(theNumberOfElements)
    for iElement in range(theNumberOfElements):   
        residualsArray[iElement] = theCoefficients.bc[iElement] - theCoefficients.ac[iElement]*theCoefficients.dphi[iElement]
        for nNeighbour in range(len(theCoefficients.theCConn[iElement])):
            iNeighbour = theCoefficients.theCConn[iElement][nNeighbour]
            residualsArray[iElement] -= theCoefficients.anb[iElement][nNeighbour]*theCoefficients.dphi[iNeighbour]
    return residualsArray



def cfdSolveILU_orig(ac,anb,bc,dc,rc,cconn,dphi):
# %==========================================================================
# % Routine Description:
# %   Solve Incomplete Lower Upper system
# %--------------------------------------------------------------------------
# % ILU Iterate
    numberOfElements = len(ac)
    # % Update Residuals array
    for iElement in range(numberOfElements):
        conn = cconn[iElement]
        res = -ac[iElement]*dphi[iElement] + bc[iElement]
        theNumberOfNeighbours = len(conn)

        for iLocalNeighbour in range(theNumberOfNeighbours):
            # % Get the neighbour cell index
            j    = conn[iLocalNeighbour]
            res -=  anb[iElement][iLocalNeighbour]*dphi[j]
        rc[iElement]= res

    # % Forward Substitution
    for i1 in range(numberOfElements):
        mat1 = dc[i1]*rc[i1]
        i1NNb = len(cconn[i1])
        i1NbList = cconn[i1]
        # % Loop over neighbours of i
        j1_ = 0
        while j1_+1 <= i1NNb:
            j1_ +=  1
            j1   = i1NbList[j1_]
            # % For all neighbour j > i do
            if j1 > i1 and j1<=numberOfElements:
                j1NbList = cconn[j1]
                j1NNB = len(j1NbList)
                i1_= 0
                k = 0
                # % Get A[j][i]
                while i1_+1<=j1NNB  and k != i1 :
                    i1_ += 1
                    k    = j1NbList[i1_]
                # % Compute rc
                if k == i1:
                    mat2    =  anb[j1][i1_]*mat1
                    rc[j1] -= mat2
                else:
                    print('ILU Solver Error The index for i  in element j  is not found \n')
    # % Backward substitution
    for i1 in range(numberOfElements-1, -1, -1):
        # % Compute rc
        if i1<numberOfElements:
            i1NBList = cconn[i1]
            i1NNb = len(i1NBList)
            j1_ = 0
            # % Loop over neighbours of i
            while j1_+1 <= i1NNb:
                j1_ += 1
                j    = i1NBList[j1_]
                if j>i1:
                    rc[i1] -= anb[i1][j1_]*rc[j]
        # % Compute product D[i]*R[i]
        mat1 = dc[i1]*rc[i1]
        rc[i1] = mat1
        # % Update dphi
        dphi[i1] +=  mat1

    return dphi

import numpy as np

def cfdSolveILU(ac, anb, bc, dc, cconn, dphi):
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
    rc = np.zeros_like(dphi)

    # Update Residuals array
    for iElement in range(numberOfElements):
        conn = cconn[iElement]
        res = -ac[iElement] * dphi[iElement] + bc[iElement]

        for iLocalNeighbour, j in enumerate(conn):
            res -= anb[iElement][iLocalNeighbour] * dphi[j]

        rc[iElement] = res

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
# %==========================================================================
# % Routine Description:
# %   This functions solves linear system Ax = b using Guass-Seidel method   
# %--------------------------------------------------------------------------
    numberOfElements = len(ac)
    for iElement in range(numberOfElements):
        local_dphi = bc[iElement]
        for iLocalNeighbour in range(len(cconn[iElement])):
            iNeighbour  = cconn[iElement][iLocalNeighbour]
            local_dphi -= anb[iElement][iLocalNeighbour]*dphi[iNeighbour]
        dphi[iElement]  = local_dphi/ac[iElement]

    for iElement in range(numberOfElements-1,-1,-1):  #逆序
        local_dphi = bc[iElement]
        for iLocalNeighbour in range(len(cconn[iElement])):
            iNeighbour = cconn[iElement][iLocalNeighbour]
            local_dphi = local_dphi - anb[iElement][iLocalNeighbour]*dphi[iNeighbour]
        dphi[iElement] = local_dphi/ac[iElement]
    
    return dphi




'''
-----------------------------------------------------------
AMG Solver
-----------------------------------------------------------
'''
def cfdApplyAMG(preconditioner='DILU', maxIter=20, tolerance=1e-6, relTol=0.1, 
                nPreSweeps=0, nPostSweeps=2, nFinestSweeps=2):
    """
    这段代码是一个基于代数多重网格（AMG）方法的求解框架。AMG 是一种用于求解大型稀疏线性系统的高效方法，特别适用于计算流体动力学（CFD）等领域。
    cfdApplyAMG 是入口函数，控制整个求解流程。
    cfdAgglomerate 和 cfdAgglomerateLevel 用于从细网格逐层构建粗网格。
    cfdApplyVCycle 是 AMG 求解的核心，应用 V-Cycle 方法进行多层次求解。
    cfdRestrict 和 cfdProlongate 实现残差的限制和解的延拓，分别用于转移残差到粗网格和将解从粗网格延拓到细网格。
    cfdSetupCoefficients 和 cfdAssembleAgglomeratedLHS 负责初始化和计算各个层级的系数矩阵。
    通过这些函数的协作，AMG 可以有效地加速稀疏线性系统的求解过程。
    Solve using Geometric-Algebraic multi-grid solver.
    Parameters:
    preconditioner - 预处理器类型
    max_iter - 最大迭代次数
    tolerance - 绝对容差
    rel_tol - 相对容差
    n_pre_sweeps - 每次多重网格循环前的预平滑次数
    n_post_sweeps - 每次多重网格循环后的后平滑次数
    n_finest_sweeps - 在最精细网格上的迭代次数

    Returns:
    initial_residual - 初始残差
    final_residual - 最终残差

    功能: 
    这是整个 AMG 求解流程的入口函数。它控制着整个代数多重网格求解过程的各个阶段，包括初始化、残差计算、循环控制和收敛判断。
    参数:
    preconditioner: 选择的预处理器类型，默认是 DILU。
    maxIter: 最大迭代次数。
    tolerance: 收敛的绝对容差。
    relTol: 收敛的相对容差。
    nPreSweeps: 预平滑的迭代次数。
    nPostSweeps: 后平滑的迭代次数。
    nFinestSweeps: 在最细网格上的迭代次数。
    过程:
    多重网格层级构建: 使用 cfdAgglomerate 构建从细网格到粗网格的层级结构。
    初始残差计算: 计算初始残差，用于后续的收敛判断。
    V-Cycle: 通过 V-Cycle 方法来进行多重网格求解。在每个循环中，通过多次迭代平滑误差，直到满足收敛条件或达到最大迭代次数。
    """

    # 默认设置
    cycleType = 'V-Cycle'
    maxCoarseLevels = 10

    # 构建粗网格
    maxLevels = cfdAgglomerate(maxCoarseLevels)

    # 计算初始残差
    theCoefficients = cfdGetCoefficients()
    residualsArray = cfdComputeResidualsArray(theCoefficients)
    initialResidual = sum(abs(residualsArray))
    finalResidual = initialResidual

    # 多重网格循环
    if maxLevels <= 3:
        for _ in range(maxIter):
            finalResidual = cfdApplyVCycle(1, preconditioner, maxLevels, nPreSweeps, nPostSweeps, relTol, nFinestSweeps)
            if finalResidual < max(relTol * initialResidual, tolerance):
                break
        return initialResidual, finalResidual

    if cycleType == 'V-Cycle':
        for _ in range(maxIter):
            finalResidual = cfdApplyVCycle(1, preconditioner, maxLevels, nPreSweeps, nPostSweeps, relTol, nFinestSweeps)
            if finalResidual < max(relTol * initialResidual, tolerance):
                break

    return initialResidual, finalResidual



def cfdAgglomerate(maxCoarseLevels):
    """
    Build algebraic multigrid hierarchy.
    功能: 
        构建代数多重网格的层次结构，从细网格逐步聚合到粗网格。
    过程:
        逐级聚合: 从第一个层级开始，逐层聚合，调用 cfdAgglomerateLevel 生成更粗的网格。
        构建粗网格方程: 使用 cfdAssembleAgglomeratedLHS 组装每个粗网格层级的线性方程系统。
        判断停止条件: 如果聚合后的粗网格父节点数量少于一个阈值，则停止聚合。
    """

    minNumberOfParents = 5
    iLevel = 1

    while iLevel <= maxCoarseLevels:
        iLevel += 1
        theNumberOfParents = cfdAgglomerateLevel(iLevel)
        cfdAssembleAgglomeratedLHS(iLevel)
        if theNumberOfParents <= minNumberOfParents:
            break

    return iLevel


def cfdApplyVCycle(gridLevel, preconditioner, maxLevels, nPreSweeps, nPostSweeps, relTol, nFinestSweeps):
    """
    Apply V-Cycle.
    功能: 
        实现 AMG 求解中的 V-Cycle，一个用于加速收敛的多重网格方法。
    过程:
        预平滑阶段: 在当前网格层级上应用预平滑方法（如 Gauss-Seidel），减少高频误差。
        限制阶段: 将残差限制到更粗的网格层级。
        粗网格求解: 在最粗网格上求解修正方程。
        延拓阶段: 将粗网格上的解延拓到更细的网格上，并进行后平滑，减少低频误差。
    """

    # 限制阶段
    while gridLevel < maxLevels:
        # 预平滑
        cfdSolveAlgebraicSystem(gridLevel, preconditioner, nPreSweeps)

        # 限制残差
        cfdRestrict(gridLevel)

        # 更新层级
        gridLevel += 1

    # 在最粗网格上平滑
    cfdSolveAlgebraicSystem(gridLevel, preconditioner, nPostSweeps)

    # 延拓阶段
    while gridLevel > 1:
        if gridLevel == 2:
            # 将修正延拓到更细的解
            cfdProlongate(gridLevel)

            # 最精细层级的后平滑
            cfdSolveAlgebraicSystem(gridLevel - 1, preconditioner, nFinestSweeps)
        else:
            # 将修正延拓到更细的解
            cfdProlongate(gridLevel)

            # 后平滑
            cfdSolveAlgebraicSystem(gridLevel - 1, preconditioner, nPostSweeps)

        gridLevel -= 1

    # 计算最终残差
    theCoefficients = cfdGetCoefficients()
    residualsArray = cfdComputeResidualsArray(theCoefficients)
    finalResidual = sum(abs(residualsArray))

    return finalResidual


def cfdRestrict(gridLevel):
    """
    Restrict residuals from gridLevel to gridLevel+1.
    功能: 
        将当前网格层级的残差限制到更粗的网格层级，准备在粗网格上进行求解。
    过程:
        获取当前层级残差: 通过 cfdComputeResidualsArray 计算当前层级的残差。
        更新粗网格 RHS: 使用 cfdUpdateRHS 将细网格的残差传递给粗网格的 RHS。
    """

    theCoefficients = cfdGetCoefficients(gridLevel)
    residual = cfdComputeResidualsArray(theCoefficients)

    cfdUpdateRHS(gridLevel + 1, residual)

def cfdProlongate(gridLevel):
    """
    Prolongate to finer level.
    功能: 
        将粗网格上的解延拓到细网格，修正细网格的解。
    过程:
        延拓修正: 调用 cfdCorrectFinerLevelSolution，将粗网格层级的解修正延拓到细网格层级。
    """
    cfdCorrectFinerLevelSolution(gridLevel)

def cfdCorrectFinerLevelSolution(gridLevel):
    """
    Prolongate correction to finer level.
    功能:
        将粗网格解的修正传递到细网格，修正细网格的解。
    过程:
        获取当前层级修正: 从粗网格层级获取修正向量 dphi。
        更新细网格解: 根据粗网格的修正，更新细网格的解。
        存储更新后的解: 将更新后的 dphi 存储到细网格层级。
    """

    theCoefficients = cfdGetCoefficients(gridLevel)
    DPHI = theCoefficients['dphi']

    theFinerLevelCoefficients = cfdGetCoefficients(gridLevel - 1)
    dphi = theFinerLevelCoefficients['dphi']
    theParents = theFinerLevelCoefficients['parents']
    theNumberOfFineElements = theFinerLevelCoefficients['numberOfElements']

    for iFineElement in range(theNumberOfFineElements):
        iParent = theParents[iFineElement]
        dphi[iFineElement] += DPHI[iParent]

    # Store corrected correction
    theFinerLevelCoefficients['dphi'] = dphi
    cfdSetCoefficients(theFinerLevelCoefficients, gridLevel - 1)


def cfdGetCoefficients(iLevel=1):
    """
    This function gets the coefficients from the data base.
    获取指定层级的系数矩阵: 从 Region['coefficients'] 中提取指定层级的系数。
    """
    global Region
    return Region['coefficients'][iLevel]

def cfdAgglomerateLevel(iLevel):
    """
    Agglomerate level to construct coarser algebraic level.
    功能: 在给定层级上构建更粗的网格层级，创建父子关系。
    过程:
        Step 1 聚合: 在细网格中找到未分配的元素，将它们聚合成父元素。
        最后一步聚合: 处理孤立的元素，将它们聚合到父元素中。
        更新连接性: 为粗网格层级创建连接性和尺寸信息。
        设置系数: 使用 cfdSetupCoefficients 为粗网格层级设置初始系数。
    """

    # 获取信息
    theCoefficients = cfdGetCoefficients(iLevel - 1)
    theNumberOfFineElements = len(theCoefficients['ac'])

    anb = theCoefficients['anb']
    cconn = theCoefficients['cconn']
    csize = theCoefficients['csize']

    parents = np.zeros(theNumberOfFineElements, dtype=int)
    maxAnb = np.zeros(theNumberOfFineElements)

    for iElement in range(theNumberOfFineElements):
        maxAnb[iElement] = max([-val for val in anb[iElement]])

    iParent = 1

    # Step 1 Agglomeration
    for iSeed in range(theNumberOfFineElements):
        if parents[iSeed] == 0:
            parents[iSeed] = iParent
            children = [iSeed]
            for iNB_local in range(csize[iSeed]):
                iNB = cconn[iSeed][iNB_local]
                if parents[iNB] == 0:
                    if (-anb[iSeed][iNB_local] / maxAnb[iSeed]) > 0.5:
                        parents[iNB] = iParent
                        children.append(iNB)

            theNumberOfChildren = len(children)
            children2 = []
            for iChild_local in range(1, theNumberOfChildren):
                iChild = children[iChild_local]
                for iChildNB_local in range(csize[iChild]):
                    iChildNB = cconn[iChild][iChildNB_local]
                    if parents[iChildNB] == 0:
                        if (-anb[iChild][iChildNB_local] / maxAnb[iChild]) > 0.5:
                            parents[iChildNB] = iParent
                            children2.append(iChildNB)

            theNumberOfChildren += len(children2)
            if theNumberOfChildren == 1:
                parents[iSeed] = 0
            else:
                iParent += 1

    # 最后一步 Agglomeration
    for iOrphan in range(theNumberOfFineElements):
        if parents[iOrphan] == 0:
            strength = 0
            for iNB_local in range(csize[iOrphan]):
                iNB = cconn[iOrphan][iNB_local]
                if parents[iNB] != 0:
                    if strength < -anb[iOrphan][iNB_local] / maxAnb[iNB]:
                        strength = -anb[iOrphan][iNB_local] / maxAnb[iNB]
                        parents[iOrphan] = parents[iNB]

            if parents[iOrphan] == 0:
                parents[iOrphan] = iParent
                iParent += 1
                print('the orphan could not find a parent')

    theNumberOfParents = iParent - 1
    theCoefficients['parents'] = parents
    cfdSetCoefficients(theCoefficients, iLevel - 1)

    # setup connectivity and csize
    theParentCConn = {i: [] for i in range(theNumberOfParents)}
    for iElement in range(theNumberOfFineElements):
        for iNB_local in range(csize[iElement]):
            iNB = cconn[iElement][iNB_local]
            if parents[iElement] != parents[iNB]:
                if parents[iNB] not in theParentCConn[parents[iElement]]:
                    theParentCConn[parents[iElement]].append(parents[iNB])

    theParentCSize = {i: len(theParentCConn[i]) for i in range(theNumberOfParents)}

    # Setup coefficients for coarser level
    theParentCoefficients = cfdSetupCoefficients(theParentCConn, theParentCSize)
    cfdSetCoefficients(theParentCoefficients, iLevel)

    return theNumberOfParents

def cfdSetupCoefficients(theCConn=None, theCSize=None):
    """
    This function sets up the coefficients.
    功能: 
        为当前层级设置和初始化各个系数矩阵。
    过程:
        默认设置: 如果没有提供 theCConn 和 theCSize，从网格中获取邻接元素索引并计算邻接元素数量。
        初始化系数: 初始化所有系数列表，包括 ac, anb, bc, dphi 等。
        返回系数字典: 将所有初始化后的系数存储在一个字典中，并返回。
    """

    # 默认设置
    if theCConn is None:
        theCConn = cfdGetElementNbIndices()
        theCSize = [len(neighbours) for neighbours in theCConn]

    theNumberOfElements = len(theCConn)

    # 定义并初始化
    ac = [0.0] * theNumberOfElements
    ac_old = [0.0] * theNumberOfElements
    bc = [0.0] * theNumberOfElements

    anb = [np.zeros(size) for size in theCSize]

    # dc & rc added to be used in ILU Solver
    dc = [0.0] * theNumberOfElements
    rc = [0.0] * theNumberOfElements

    # Correction
    dphi = [0.0] * theNumberOfElements

    # 存储在字典结构中
    theCoefficients = {
        'ac': ac,
        'ac_old': ac_old,
        'bc': bc,
        'anb': anb,
        'dc': dc,
        'rc': rc,
        'dphi': dphi,
        'cconn': theCConn,
        'csize': theCSize,
        'numberOfElements': theNumberOfElements
    }

    return theCoefficients

def cfdGetElementNbIndices():
    """
    This function retrieves the element neighbor indices.
    功能: 
        获取每个元素的邻接元素索引列表，用于后续计算。
    过程:
        从全局变量中获取邻接元素索引: 通过 Region['mesh']['elementNeighbours'] 获取每个元素的邻接元素。
    """

    global Region

    # 获取网格中每个元素的邻接元素索引
    elementNbIndices = Region.mesh['elementNeighbours']

    return elementNbIndices


def cfdAssembleAgglomeratedLHS(iLevel):
    """
    Calculate coarse level's LHS coefficients (ac, anb, etc).
    功能: 
        为粗网格层级计算左侧系数（ac, anb 等）。
    过程:
        聚合系数: 将细网格的系数聚合到粗网格上，以计算粗网格的 ac 和 anb。
        存储: 将计算结果存储到粗网格层级中。
    """

    # 获取信息
    theCoefficients = cfdGetCoefficients(iLevel - 1)
    parents = theCoefficients['parents']
    ac = theCoefficients['ac']
    anb = theCoefficients['anb']
    cconn = theCoefficients['cconn']
    csize = theCoefficients['csize']

    theParentCoefficients = cfdGetCoefficients(iLevel)
    AC = theParentCoefficients['ac']
    ANB = theParentCoefficients['anb']
    CCONN = theParentCoefficients['cconn']

    theNumberOfFineElements = theCoefficients['numberOfElements']
    for iElement in range(theNumberOfFineElements):
        iParent = parents[iElement]
        AC[iParent] += ac[iElement]
        theNumberOfNeighbours = csize[iElement]
        for iNB_local in range(theNumberOfNeighbours):
            iNB = cconn[iElement][iNB_local]
            iNBParent = parents[iNB]
            if iNBParent == iParent:
                AC[iParent] += anb[iElement][iNB_local]
            else:
                iNBParent_local = CCONN[iParent].index(iNBParent)
                ANB[iParent][iNBParent_local] += anb[iElement][iNB_local]

    # Store
    theParentCoefficients['ac'] = AC
    theParentCoefficients['anb'] = ANB
    theParentCoefficients['cconn'] = CCONN

    cfdSetCoefficients(theParentCoefficients, iLevel)

def cfdUpdateRHS(gridLevel, residual):
    """
    Calculate coarse level's RHS coefficients (bc).
    功能: 
        计算粗网格层级的右侧系数 bc，即残差的聚合。
    过程:
        获取细网格信息: 获取细网格层级的残差和父子关系。
        累加残差: 将细网格层级的残差累加到相应的粗网格父节点上。
        存储粗网格 RHS: 将计算的 bc 存储到粗网格层级。
    """

    # 获取细网格层级的信息
    theCoefficients = cfdGetCoefficients(gridLevel - 1)
    theParents = theCoefficients['parents']
    theNumberOfElements = theCoefficients['numberOfElements']

    # 获取粗网格层级的信息
    theCoarseLevelCoefficients = cfdGetCoefficients(gridLevel)
    theNumberOfCoarseElements = theCoarseLevelCoefficients['numberOfElements']

    # 初始化粗网格层级的 RHS 系数 bc
    BC = np.zeros(theNumberOfCoarseElements)

    # 将细网格的残差累加到相应的粗网格父节点上
    for iFineElement in range(theNumberOfElements):
        iParent = theParents[iFineElement]
        BC[iParent] += residual[iFineElement]

    # 存储粗网格层级的 RHS 系数
    theCoarseLevelCoefficients['bc'] = BC
    cfdSetCoefficients(theCoarseLevelCoefficients, gridLevel)

def cfdSetCoefficients(coefficients, iLevel):
    """
    Set the coefficients for a specific grid level in the global Region.
    存储系数: 将传递进来的 coefficients 字典存储到 Region 的相应层级中。
    """
    global Region
    Region['coefficients'][iLevel] = coefficients
