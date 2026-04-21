import numpy as np
import cfdtool.IO as io
import cfdtool.Math as mth
from cfdtool.quantities import Quantity as Q_

def cfdInterpolateFromElementsToFaces(Region,theInterpolationScheme,field,*args):
    return Q_(cfdInterpolateFromElementsToFaces_tool(Region,theInterpolationScheme,field.value,*args),field.dimension)

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
    phi_f=np.zeros((numberOfFaces,theNumberOfComponents))

    # 插值内部面
    phi_f[:numberOfInteriorFaces,:]=cfdInterpolateFromElementsToInteriorFaces_tool(Region,theInterpolationScheme,field[:Region.mesh.numberOfElements])
    ''' **边界面插值**：
    - 对于边界面（从 `numberOfInteriorFaces` 到 `numberOfFaces`），将单元格中心的值直接赋给 `phi_f`，因为边界面没有邻居单元格。
    '''
    boundaryFaceOwners = Region.mesh.owners[numberOfInteriorFaces:]
    neighbourFaceOwners= Region.mesh.owners_b
    # 处理边界面
    phi_f[numberOfInteriorFaces:numberOfFaces, :] = 0.5*(field[boundaryFaceOwners, :]+field[neighbourFaceOwners, :])# TODO check theInterpolation
    
    return phi_f


def  cfdInterpolateFromElementsToInteriorFaces(Region,theInterpolationScheme,field, *args):
    if args:
        return Q_(cfdInterpolateFromElementsToInteriorFaces_tool(Region,theInterpolationScheme,field.value, args[0].value),field.dimension)
    else:
        return Q_(cfdInterpolateFromElementsToInteriorFaces_tool(Region,theInterpolationScheme,field.value),field.dimension)

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
    phi_f=np.zeros((numberOfInteriorFaces,theNumberOfComponents))
    
    if theInterpolationScheme == 'linear':
        # Calculate face-based interpolation weights
        # w = dN / (dO + dN), where dO is distance from owner to face, dN is distance from face to neighbor
        # phi_f = w * phi_O + (1 - w) * phi_N
        phi_f[:numberOfInteriorFaces, :] = g_f[:, None] * field[owners, :] + (1 - g_f)[:, None] * field[neighbours, :]

    elif theInterpolationScheme == 'harmonic mean':
        # 谐和平均插值： phi_f = 1 / ((1 - w)/phi_O + w/phi_N)
        phi_f[:numberOfInteriorFaces,:] = 1 / ((1 - g_f)[:, None] / field[owners, :] + g_f[:, None] / field[neighbours, :])
        # for iComponent in range(theNumberOfComponents):
        #     phi_f[0:numberOfInteriorFaces,iComponent]=1/((1-g_f)/field[owners][:,iComponent]+g_f/field[neighbours][:,iComponent])

    elif theInterpolationScheme == 'vanLeerV':
        vol = Region.mesh.elementVolumes
        phi_f[:numberOfInteriorFaces,:] = ((vol[owners] + vol[neighbours]) * field[owners, :] * field[neighbours, :]) / \
                (vol[neighbours] * field[owners, :] + vol[owners] * field[neighbours, :])
        # for iComponent in range(theNumberOfComponents):
        #     phi_f[0:numberOfInteriorFaces,iComponent] = (vol[owners]+vol[neighbours])*field[owners,iComponent]*field[neighbours,iComponent]/(vol[neighbours]*field[owners,iComponent]+vol[owners]*field[neighbours,iComponent])
    
    elif theInterpolationScheme == 'linearUpwind':
        if args:
            mdot_f=args[0]
            pos = (mdot_f > 0).astype(int)[:numberOfInteriorFaces,:]
            # 插值： phi_f = pos * phi_O + (1 - pos) * phi_N
            phi_f[:numberOfInteriorFaces,:] = pos * field[owners, :] + (1 - pos) * field[neighbours, :]
            # for iComponent in range(theNumberOfComponents):
            #     phi_f[0:numberOfInteriorFaces,iComponent] = field[owners,iComponent]*pos + field[neighbours,iComponent]*(1 - pos)
        else:
            io.cfdError("linearUpwind is implemented in interpolateFromElementsToFaces Error!!!")
    else:
        io.cfdError(theInterpolationScheme+" is not yet implemented in interpolateFromElementsToFaces")
    return phi_f

def cfdInterpolateGradientsFromElementsToInteriorFaces(Region,gradPhi,scheme,*args):
    if args:
        return Q_(cfdInterpolateGradientsFromElementsToInteriorFaces_tool(Region,gradPhi.value,scheme,args[0].value),gradPhi.dimension)
    else:
        return Q_(cfdInterpolateGradientsFromElementsToInteriorFaces_tool(Region,gradPhi.value,scheme),gradPhi.dimension)
    
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
    CF = Region.mesh.interiorFaceCF.value
    # vector of ones
    # face gradient matrix
    grad_f= np.zeros((numberOfInteriorFaces, 3),dtype='float')
    
    if scheme == 'linear' or scheme == 'Gauss linear':
        # for i in range(theNumberOfComponents):
        """ 
        Example of what is going one in one of the lines below:
            
        a = ones - g_f                              <--- returns a 1D (:,) array (1 - faceWeight)
        b = gradPhi[self.neighbours_f][:,0,i]       <--- grabs only rows in self.neighbours, in column 0, for component 'i'
        c = g_f*self.gradPhi[self.owners_f][:,0,i]  <--- multiplies faceWeight by owners in column 0, for component 'i'
        
        grad_f[:,0,i] = a*self.b + c                <--- fills (:) column '0' for component 'i' with the result of self.a*self.b + self.c 
        
        In words, what is going is:
            
            gradient of phi at face = (1 - faceWeight for cell)*neighbouring element's gradient + faceWeight*owner element's gradient
            
            If the faceWeight (g_f) of the owner cell is high, then its gradient contributes more to the face's gradient than the neighbour cell.
        """
        # 线性插值： grad_f = (1 - w) * gradPhi_N + w * gradPhi_O
        grad_f[:numberOfInteriorFaces,:]=(1-g_f)[:,None]*gradPhi[neighbours_f,:]+g_f[:,None]*gradPhi[owners_f,:]

    elif scheme == 'Gauss linear corrected':
        #书籍《The Finite Volume Method in Computational Fluid Dynamics》Page 289页
        grad_f[:numberOfInteriorFaces,:]=(1-g_f)[:,None]*gradPhi[neighbours_f,:]+g_f[:,None]*gradPhi[owners_f,:]
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
        grad_f += (local_grad- local_avg_grad)
    
    elif scheme == 'Gauss upwind':
        #not yet implemented, but it will have to be
        local_grad_f=(1-g_f)[:,None]*gradPhi[neighbours_f,:]+g_f[:,None]*gradPhi[owners_f,:]
        if args:
            mdot_f=args[1]
        else:
            io.cfdError('Gauss upwind requires flow rate (mdot_f) as a second argument')
        
        # 根据流量方向选择插值方向
        pos = (mdot_f > 0).astype(int)[:numberOfInteriorFaces,:]
        grad_f[:numberOfInteriorFaces,:] = pos*local_grad_f[owners_f,:] + (1-pos)*local_grad_f[neighbours_f,:]
    else:
        io.cfdError(f"{scheme} is not yet implemented, but it will have to be")

    return grad_f
