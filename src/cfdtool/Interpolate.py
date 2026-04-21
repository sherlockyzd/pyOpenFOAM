import numpy as np
import cfdtool.IO as io
import cfdtool.Math as mth
from cfdtool.backend import be

def cfdInterpolateFromElementsToFaces(Region,theInterpolationScheme,field,*args):
    return cfdInterpolateFromElementsToFaces_tool(Region,theInterpolationScheme,field,*args)

def cfdInterpolateFromElementsToFaces_tool(Region,theInterpolationScheme,field,*args):
    '''
    将单元格（元素）上的场数据插值到网格的面（faces）上。
    
    参数：
    - Region: 包含流体区域信息的对象。
    - theInterpolationScheme: 插值方案的名称（例如 'linear', 'harmonic mean' 等）。
    - field: 要插值的场的数据，形状为 (numberOfElements, numberOfComponents)。
    
    返回：
    - phi_f: 插值后的面场数据，形状为 (numberOfFaces, numberOfComponents)。
    '''
    # theInterpolationScheme=scheme
    if field.shape[0]!=Region.mesh.numberOfElements+Region.mesh.numberOfBElements:
        raise  ValueError("The shape of gradPhi does not match the number of elements in the mesh.")
    try:
        theNumberOfComponents = field.shape[1]
    except:
        theNumberOfComponents =1

    numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
    numberOfFaces = Region.mesh.numberOfFaces
    phi_f=be.zeros((numberOfFaces,theNumberOfComponents))

    # 插值内部面
    phi_f = be.set_at(phi_f, (slice(None, numberOfInteriorFaces), slice(None)), cfdInterpolateFromElementsToInteriorFaces_tool(Region,theInterpolationScheme,field[:Region.mesh.numberOfElements]))
    ''' **边界面插值**：
    - 对于边界面（从 `numberOfInteriorFaces` 到 `numberOfFaces`），将单元格中心的值直接赋给 `phi_f`，因为边界面没有邻居单元格。
    '''
    boundaryFaceOwners = Region.mesh.owners[numberOfInteriorFaces:]
    neighbourFaceOwners= Region.mesh.owners_b
    # 处理边界面
    phi_f = be.set_at(phi_f, (slice(numberOfInteriorFaces, numberOfFaces), slice(None)), 0.5*(field[boundaryFaceOwners, :]+field[neighbourFaceOwners, :]))# TODO check theInterpolation
    
    return phi_f


def cfdInterpolateFromElementsToInteriorFaces(Region,theInterpolationScheme,field, *args):
    return cfdInterpolateFromElementsToInteriorFaces_tool(Region,theInterpolationScheme,field, *args)

def  cfdInterpolateFromElementsToInteriorFaces_tool(Region,theInterpolationScheme,field, *args):
    '''
    根据指定的插值方案，将场数据从单元格插值到内部面上。
    
    参数：
    - Region: 包含流体区域信息的对象。
    - theInterpolationScheme: 插值方案的名称（例如 'linear', 'harmonic mean' 等）。
    - field: 要插值的场的数据，形状为 (numberOfElements, numberOfComponents)。
    - *args: 其他参数（例如 'linearUpwind' 方案需要的流量数据）。
    
    返回：
    - phi_f: 插值后的内部面场数据，形状为 (numberOfInteriorFaces, numberOfComponents)。
    '''
    if field.shape[0]!=Region.mesh.numberOfElements:
        raise  ValueError("The shape of gradPhi does not match the number of elements in the mesh.")
    try:
        theNumberOfComponents = field.shape[1]
    except:
        theNumberOfComponents =1
    numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
    owners = Region.mesh.interiorFaceOwners
    neighbours = Region.mesh.interiorFaceNeighbours
    
    #interpolation factor p.160 in Moukalled
    g_f=Region.mesh.interiorFaceWeights
    
    if theInterpolationScheme == 'linear':
        # 使用空列表 + concatenate 避免 np.zeros 预分配（结果直接是连续内存）
        g_f_2d = g_f[:, None]
        g_f_compl = 1.0 - g_f_2d
        phi_f = g_f_2d * field[owners, :] + g_f_compl * field[neighbours, :]

    elif theInterpolationScheme == 'harmonic mean':
        # 谐和平均插值： phi_f = 1 / ((1 - w)/phi_O + w/phi_N)
        g_f_2d = g_f[:, None]
        phi_f = 1.0 / ((1.0 - g_f_2d) / field[owners, :] + g_f_2d / field[neighbours, :])

    elif theInterpolationScheme == 'vanLeerV':
        vol = Region.mesh.elementVolumes
        phi_O = field[owners, :]
        phi_N = field[neighbours, :]
        phi_f = ((vol[owners, None] + vol[neighbours, None]) * phi_O * phi_N) / \
                (vol[neighbours, None] * phi_O + vol[owners, None] * phi_N)
    
    elif theInterpolationScheme == 'linearUpwind':
        if args:
            mdot_f=args[0]
            pos = (mdot_f > 0).astype(int)[:numberOfInteriorFaces,:]
            # 插值： phi_f = pos * phi_O + (1 - pos) * phi_N
            phi_f = pos * field[owners, :] + (1 - pos) * field[neighbours, :]
        else:
            io.cfdError("linearUpwind is implemented in interpolateFromElementsToFaces Error!!!")
    else:
        io.cfdError(theInterpolationScheme+" is not yet implemented in interpolateFromElementsToFaces")
    return phi_f

def cfdInterpolateGradientsFromElementsToInteriorFaces(Region,gradPhi,scheme,*args):
    return cfdInterpolateGradientsFromElementsToInteriorFaces_tool(Region,gradPhi,scheme,*args)
    
def cfdInterpolateGradientsFromElementsToInteriorFaces_tool(Region,gradPhi,scheme,*args):
    '''
    将单元中心计算的梯度插值到内部面上。
    
    参数：
    - Region: 包含流体区域信息的对象。
    - gradPhi: 在单元格中心计算得到的梯度，形状为 (numberOfElements, 3, numberOfComponents)。
    - scheme: 插值方案的名称（例如 'linear', 'Gauss linear' 等）。
    - *args: 其他参数（例如 'Gauss linear corrected' 需要的场数据 phi）。
    
    返回：
    - grad_f: 插值后的内部面梯度数据，形状为 (numberOfInteriorFaces, 3, numberOfComponents)。
    '''
    if gradPhi.shape[0]!=Region.mesh.numberOfElements:
        raise  ValueError("The shape of gradPhi does not match the number of elements in the mesh.")
    numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
    # owners of elements of faces
    owners_f=Region.mesh.interiorFaceOwners
    # neighbour elements of faces
    neighbours_f=Region.mesh.interiorFaceNeighbours
    # face weights
    g_f=Region.mesh.interiorFaceWeights
    # vector formed between owner (C) and neighbour (f) elements
    CF = Region.mesh.interiorFaceCF
    # vector of ones
    # face gradient matrix
    grad_f= be.zeros((numberOfInteriorFaces, 3))
    
    if scheme == 'linear' or scheme == 'Gauss linear':
        # 线性插值： grad_f = (1 - w) * gradPhi_N + w * gradPhi_O
        grad_f = be.set_at(grad_f, (slice(None, numberOfInteriorFaces), slice(None)), (1-g_f)[:,None]*gradPhi[neighbours_f,:]+g_f[:,None]*gradPhi[owners_f,:])

    elif scheme == 'Gauss linear corrected':
        #书籍《The Finite Volume Method in Computational Fluid Dynamics》Page 289页
        grad_f = be.set_at(grad_f, (slice(None, numberOfInteriorFaces), slice(None)), (1-g_f)[:,None]*gradPhi[neighbours_f,:]+g_f[:,None]*gradPhi[owners_f,:])
        # ScfdUrface-normal gradient
        dcfdMag = mth.cfdMag(CF)
        e_CF = mth.cfdUnit(CF)

        # local_grad=np.zeros((numberOfInteriorFaces, 3))
        if args:
            phi=args[0]
            local_grad_cfdMag_f = (phi[neighbours_f]-phi[owners_f])/dcfdMag
            local_grad=local_grad_cfdMag_f[:,None]*e_CF
        else:
            io.cfdError('No phi provided for Gauss linear corrected interpolation')

        local_avg_grad = mth.cfdDot(grad_f, e_CF)[:, None] * e_CF

        # Corrected gradient
        grad_f = grad_f + (local_grad- local_avg_grad)
    
    elif scheme == 'Gauss upwind':
        #not yet implemented, but it will have to be
        local_grad_f=(1-g_f)[:,None]*gradPhi[neighbours_f,:]+g_f[:,None]*gradPhi[owners_f,:]
        if args:
            mdot_f=args[1]
        else:
            io.cfdError('Gauss upwind requires flow rate (mdot_f) as a second argument')
        
        # 根据流量方向选择插值方向
        pos = (mdot_f > 0).astype(int)[:numberOfInteriorFaces,:]
        grad_f = be.set_at(grad_f, (slice(None, numberOfInteriorFaces), slice(None)), pos*local_grad_f[owners_f,:] + (1-pos)*local_grad_f[neighbours_f,:])
    else:
        io.cfdError(f"{scheme} is not yet implemented, but it will have to be")

    return grad_f
