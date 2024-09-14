import numpy as np
import pyFVM.IO as io
import pyFVM.Math as mth


def cfdinterpolateFromElementsToFaces(Region,scheme,field,*args):
    '''
    这段Python代码定义了一个名为 `interpolateFromElementsToFaces` 的函数，它用于根据指定的插值方案，将单元格（元素）上的场数据插值到网格的面（faces）上。这种插值在计算流体动力学（CFD）中是常见的操作，用于准备和计算通量。以下是对这个方法的详细解释：

    1. **函数定义**：
    - `def interpolateFromElementsToFaces(region, scheme, field):` 定义了一个函数，接收三个参数：`region`（包含流体区域信息的对象），`scheme`（插值方案的名称），以及 `field`（要插值的场的名称）。

    2. **插值方案**：
    - `theInterpolationScheme = scheme` 存储插值方案的名称。

    3. **获取场的维度和网格信息**：
    - `theNumberOfComponents` 获取场 `field` 的组件数量，即 `region.fluid[field].phi.shape[1]`。
    - `theNumberOfElements` 获取网格中单元格的数量。
    - `numberOfInteriorFaces` 和 `numberOfFaces` 分别获取内部面和总面的数目。
    - `owners` 和 `neighbours` 分别获取内部面的所有者和邻居单元格的索引。

    4. **插值权重**：
    - `g_f` 通过 `region.mesh.interiorFaceWeights` 获取内部面的插值权重。

    5. **初始化数组**：
    - `ones` 初始化一个与内部面数目相同大小的全1数组。
    - `phi_f` 初始化一个大小为 `(numberOfFaces, theNumberOfComponents)` 的零数组，用于存储插值后的面场数据。

    6. **线性插值**：
    - 如果 `theInterpolationScheme` 是 `'linear'`，则使用线性插值方法将单元格中心的值插值到面上。插值公式考虑了面权重 `g_f` 和相邻单元格的值。

    7. **其他插值方案**：
    - 如果 `theInterpolationScheme` 是 `'vanLeerV'` 或 `'linearUpwind'`，函数打印一条消息说明这些方案尚未实现，并退出程序。

    8. **边界面插值**：
    - 对于边界面（从 `numberOfInteriorFaces` 到 `numberOfFaces`），将单元格中心的值直接赋给 `phi_f`，因为边界面没有邻居单元格。

    9. **返回结果**：
    - 函数返回 `phi_f`，即插值后的面场数据。

    ### 注意事项：
    - 这个函数假设 `region` 对象已经包含了所有必要的网格和场信息。
    - `region.fluid[field].phi` 应该是一个 NumPy 数组，存储了场 `field` 在单元格中心的值。
    - 插值方案 `'linear'` 被实现了，而 `'vanLeerV'` 和 `'linearUpwind'` 尚未实现。
    - 边界面的插值简单地复制了单元格中心的值，这通常意味着这些面位于计算域的边界上，没有对面的邻居单元格。

    这个函数是CFD数值求解过程的一部分，用于准备面通量的计算，是实现有限体积方法中的一个重要步骤。
    '''
    theInterpolationScheme=scheme
    try:
        theNumberOfComponents = np.asarray(field).shape[1]
    except:
        theNumberOfComponents =1
    theNumberOfElements=Region.mesh.numberOfElements
    numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
    numberOfFaces = Region.mesh.numberOfFaces
    owners = Region.mesh.interiorFaceOwners
    neighbours = Region.mesh.interiorFaceNeighbours
    # owners_f = Region.mesh.interiorFaceOwners
    # neighbours_f = Region.mesh.interiorFaceNeighbours
    
    #interpolation factor p.160 in Moukalled
    g_f=np.asarray(Region.mesh.interiorFaceWeights)
    
    #array of ones for subtraction operation below
    # ones=np.ones((numberOfInteriorFaces))
    
    #empty array to hold values of phi at the faces
    # if theNumberOfComponents==1:
    #     phi_f=np.zeros((numberOfFaces))
    # else:
    phi_f=np.zeros((numberOfFaces,theNumberOfComponents))
    
    if theInterpolationScheme == 'linear':
        #interpolate centroid values to faces
        # if theNumberOfComponents==1:
        #     phi_f[0:numberOfInteriorFaces]=g_f*region.fluid[field].phi[neighbours]+(ones-g_f)*region.fluid[field].phi[owners]
        # else:
        for iComponent in range(theNumberOfComponents):
            # phi_f[0:numberOfInteriorFaces,iComponent]=g_f*Region.fluid[field].phi[neighbours][:,iComponent]+(ones-g_f)*Region.fluid[field].phi[owners][:,iComponent]
            phi_f[0:numberOfInteriorFaces,iComponent]=g_f*field[neighbours][:,iComponent]+(1-g_f)*field[owners][:,iComponent]

    elif theInterpolationScheme == 'harmonic mean':
        #interpolate centroid values to faces
        # if theNumberOfComponents==1:
        #     phi_f[0:numberOfInteriorFaces]=g_f*region.fluid[field].phi[neighbours]+(ones-g_f)*region.fluid[field].phi[owners]
        # else:
        for iComponent in range(theNumberOfComponents):
            # phi_f[0:numberOfInteriorFaces,iComponent]=g_f*Region.fluid[field].phi[neighbours][:,iComponent]+(ones-g_f)*Region.fluid[field].phi[owners][:,iComponent]
            phi_f[0:numberOfInteriorFaces,iComponent]=1/((1-g_f)/field[neighbours][:,iComponent]+g_f/field[owners][:,iComponent])

    elif theInterpolationScheme == 'vanLeerV':
        # io.cfdError("vanLeerV is not yet implemented in interpolateFromElementsToFaces")
        # sys.exit()
        vol = Region.mesh.elementVolumes
        for iComponent in range(theNumberOfComponents):
            phi_f[0:numberOfInteriorFaces,iComponent] = (vol[owners]+vol[neighbours])*field[owners,iComponent]*field[neighbours,iComponent]/(vol[neighbours]*field[owners,iComponent]+vol[owners]*field[neighbours,iComponent])
    
    elif theInterpolationScheme == 'linearUpwind':
        if args:
            mdot_f=args[0]
            pos = np.zeros(mdot_f.shape)
            pos[mdot_f>0] = 1
            for iComponent in range(theNumberOfComponents):
                phi_f[0:numberOfInteriorFaces,iComponent] = field[owners,iComponent]*pos + field[neighbours,iComponent]*(1 - pos)
        else:
            io.cfdError("linearUpwind is implemented in interpolateFromElementsToFaces Error!!!")
        # sys.exit()
    else:
        io.cfdError(theInterpolationScheme+" is not yet implemented in interpolateFromElementsToFaces")
    # if theNumberOfComponents==1:
    #     phi_f[numberOfInteriorFaces:numberOfFaces]=region.fluid[field].phi[theNumberOfElements:numberOfFaces]
    # else:
    for iComponent in range(theNumberOfComponents):
        phi_f[numberOfInteriorFaces:numberOfFaces,iComponent]=field[theNumberOfElements:numberOfFaces,iComponent]
    ''' **边界面插值**：
    - 对于边界面（从 `numberOfInteriorFaces` 到 `numberOfFaces`），将单元格中心的值直接赋给 `phi_f`，因为边界面没有邻居单元格。
    '''
    return phi_f

def  cfdInterpolateFromElementsToInteriorFaces(Region,theInterpolationScheme, phi, *args):
# %==========================================================================
# % Routine Description:
# %   This function interpolates a field phi to faces
# %--------------------------------------------------------------------------
    # % Get field type
    theNumberOfComponents = np.size(phi,axis=1)
    # % Get info
    owners_f = Region.mesh.interiorFaceOwners
    neighbours_f = Region.mesh.interiorFaceNeighbours
    g_f = np.asarray(Region.mesh.interiorFaceWeights)
    volumes = Region.mesh.elementVolumes
    theNumberOfInteriorFaces = Region.mesh.numberOfInteriorFaces

    # % Initialize face array
    # if theNumberOfComponents==1:
    #     phi_f = np.zeros(theNumberOfInteriorFaces)[:,None]
    # else:
    phi_f = np.zeros((theNumberOfInteriorFaces,theNumberOfComponents),dtype='float')

    if theInterpolationScheme=='vanLeerV':    #% BODGE
        for iComponent in range(theNumberOfComponents):
            phi_f[:,iComponent] = (volumes[owners_f]+volumes[neighbours_f])*phi[owners_f,iComponent]*phi[neighbours_f,iComponent]/(volumes[neighbours_f]*phi[owners_f,iComponent]+volumes[owners_f]*phi[neighbours_f,iComponent])
    elif theInterpolationScheme=='linearUpwind':
        if args:
            mdot_f=args[0]
        else:
            io.cfdError('No mdot_f')
        pos = np.zeros(mdot_f.shape)
        pos[mdot_f>0] = 1
        pos=np.squeeze(pos)
        for iComponent in range(theNumberOfComponents):
            phi_f[:,iComponent] = phi[owners_f,iComponent]*pos + phi[neighbours_f,iComponent]*(1 - pos)
    elif theInterpolationScheme=='linear':
        for iComponent in range(theNumberOfComponents):
            phi_f[:,iComponent] = g_f*phi[neighbours_f,iComponent] + (1-g_f)*phi[owners_f,iComponent]
    else:
        io.cfdError(theInterpolationScheme,+' interpolation scheme incorrect\n')
 
    return np.squeeze(phi_f)

def cfdInterpolateGradientsFromElementsToInteriorFaces(Region,gradPhi,scheme,*args):
    
    """ Interpolates the gradient's calculated at the cell centers to the face
    这段Python代码定义了一个名为`cfdInterpolateGradientFromElementsToInteriorFaces`的函数，它用于将单元格中心处计算得到的梯度插值到内部面上。这种插值在CFD中用于准备和计算通量，是有限体积方法中的一个重要步骤。以下是对这个方法的详细解释：

    1. **函数定义**：
    - `def cfdInterpolateGradientFromElementsToInteriorFaces(region, gradPhi, scheme, theNumberOfComponents):` 定义了一个函数，接收四个参数：`region`（包含流体区域信息的对象），`gradPhi`（在单元格中心计算得到的梯度），`scheme`（插值方案的名称），以及 `theNumberOfComponents`（梯度的组件数量）。

    2. **获取网格信息**：
    - 获取网格的单元格数量、内部面的数量和总面的数量。
    - 获取内部面的所有者和邻居单元格的索引。
    - 获取内部面的权重和中心向量。

    3. **初始化数组**：
    - `ones` 初始化一个与内部面数量相同大小的全1数组。
    - `grad_f` 初始化一个大小为 `(numberOfInteriorFaces, 3, theNumberOfComponents)` 的零三维数组，用于存储插值后的面梯度数据。

    4. **线性插值**：
    - 如果 `scheme` 是 `'linear'` 或 `'Gauss linear'`，则使用线性插值方法将单元格中心的梯度插值到面上。插值公式考虑了面权重 `g_f` 和相邻单元格的梯度。

    5. **插值实现**：
    - 通过循环遍历每个组件 `i`，使用 `(ones - g_f)` 和 `g_f` 作为权重，将邻居单元格和所有者单元格的梯度插值到面上。这个过程对梯度的每个分量（0、1、2，通常对应x、y、z方向）进行。

    6. **返回结果**：
    - 函数返回 `grad_f`，即插值后的面梯度数据。

    7. **未实现的插值方案**：
    - 对于 `'Gauss linear corrected'` 和 `'Gauss upwind'` 方案，代码中尚未实现，只是预留了位置。

    ### 注意事项：
    - 这个函数假设 `region` 对象已经包含了所有必要的网格信息和场梯度 `gradPhi`。
    - `gradPhi` 应该是一个 NumPy 数组，存储了在单元格中心计算得到的梯度值。
    - 插值方案 `'linear'` 和 `'Gauss linear'` 被实现了，而 `'Gauss linear corrected'` 和 `'Gauss upwind'` 尚未实现。
    - 函数中包含的注释详细解释了线性插值的过程和原理。

    这个函数是CFD数值求解过程的一部分，用于准备面通量的计算，特别是在需要在控制体积界面上计算物理量的梯度时非常有用。
    """
    # theNumberOfElements=region.mesh.numberOfElements
    numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
    # numberOfFaces = region.mesh.numberOfFaces
    ## owners of elements of faces
    owners_f=Region.mesh.interiorFaceOwners
    ## neighbour elements of faces
    neighbours_f=Region.mesh.interiorFaceNeighbours
    ## face weights
    g_f=Region.mesh.interiorFaceWeights
    ## vector formed between owner (C) and neighbour (f) elements
    CF = Region.mesh.interiorFaceCF
    ## vector of ones
    ones=np.ones((numberOfInteriorFaces))
    # if args:
    #     theNumberOfComponents=args[1]
    ## face gradient matrix
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
        grad_f=(ones-g_f)[:,None]*gradPhi[neighbours_f,:]+np.asarray(g_f)[:,None]*gradPhi[owners_f,:]
        # grad_f[:,0]=(ones-g_f)*gradPhi[neighbours_f][:,0]+\
        # g_f*gradPhi[owners_f][:,0]
        
        # grad_f[:,1]=(ones-g_f)*gradPhi[neighbours_f][:,1]+\
        # g_f*gradPhi[owners_f][:,1]
        
        # grad_f[:,2]=(ones-g_f)*gradPhi[neighbours_f][:,2]+\
        # g_f*gradPhi[owners_f][:,2] 

    elif scheme == 'Gauss linear corrected':
        #书籍《The Finite Volume Method in Computational Fluid Dynamics》Page 289页
        # pass
        # # for i in range(theNumberOfComponents):       
        # grad_f[:,0]=(ones-g_f)*gradPhi[neighbours_f][:,0]+\
        # g_f*gradPhi[owners_f][:,0]
        
        # grad_f[:,1]=(ones-g_f)*gradPhi[neighbours_f][:,1]+\
        # g_f*gradPhi[owners_f][:,1]
        
        # grad_f[:,2]=(ones-g_f)*gradPhi[neighbours_f][:,2]+\
        # g_f*gradPhi[owners_f][:,2]
        grad_f=(ones-g_f)[:,None]*gradPhi[neighbours_f,:]+np.asarray(g_f)[:,None]*gradPhi[owners_f,:]
        # % ScfdUrface-normal gradient
        dcfdMag = mth.cfdMag(CF)
        e_CF = mth.cfdUnit(CF)

        local_grad=np.zeros((numberOfInteriorFaces, 3))
        if args:
            phi=args[0]
        # local_grad_cfdMag_f = np.squeeze((phi[neighbours_f]-phi[owners_f]))/dcfdMag
        # local_grad[:,0] = local_grad_cfdMag_f*e_CF[:,0] 
        # local_grad[:,1] = local_grad_cfdMag_f*e_CF[:,1]
        # local_grad[:,2] = local_grad_cfdMag_f*e_CF[:,2]
            local_grad_cfdMag_f = np.squeeze((phi[neighbours_f]-phi[owners_f]))/dcfdMag
            local_grad=local_grad_cfdMag_f[:,None]*e_CF
        else:
            io.cfdError('No phi exist!!')
        local_avg_grad=(grad_f*e_CF).sum(1)[:,None]*e_CF
        # for i in range(numberOfInteriorFaces):
        #     local_avg_grad_cfdMag =np.dot(grad_f[i,:],e_CF[i,:])
        #     local_avg_grad[i,:]=local_avg_grad_cfdMag*e_CF[i,:]
            # local_avg_grad[i,0] = local_avg_grad_cfdMag*e_CF[i,0]
            # local_avg_grad[i,1] = local_avg_grad_cfdMag*e_CF[i,1]
            # local_avg_grad[i,2] = local_avg_grad_cfdMag*e_CF[i,2]    
        # % Corrected gradient
        grad_f += (local_grad- local_avg_grad)
    
    elif scheme == 'Gauss upwind':
        #not yet implemented, but it will have to be
        # io.cfdError(scheme+'not yet implemented, but it will have to be')
        # local_grad_f=np.zeros((numberOfInteriorFaces,3),dtype='float')
        local_grad_f=(ones-g_f)[:,None]*gradPhi[neighbours_f,:]+np.asarray(g_f)[:,None]*gradPhi[owners_f,:]
        # local_grad_f[:,0]=(ones-g_f)*gradPhi[neighbours_f][:,0]+\
        # g_f*gradPhi[owners_f][:,0]
        
        # local_grad_f[:,1]=(ones-g_f)*gradPhi[neighbours_f][:,1]+\
        # g_f*gradPhi[owners_f][:,1]
        
        # local_grad_f[:,2]=(ones-g_f)*gradPhi[neighbours_f][:,2]+\
        # g_f*gradPhi[owners_f][:,2] 
        if args:
            mdot_f=args[1]
        else:
            io.cfdError('mdot_f not exist')
        
        pos = np.zeros(mdot_f.shape)
        pos[mdot_f>0] = 1
        pos=np.squeeze(pos) 
        # for i in range(numberOfInteriorFaces):
        grad_f = pos*local_grad_f[neighbours_f,:] + (ones-pos)*local_grad_f[owners_f,:]
        # grad_f[:,0]  = pos*local_grad_f(neighbours_f,1) + (1-pos)*local_grad_f(owners_f,1)
        # grad_f[:,1]  = pos*local_grad_f(neighbours_f,2) + (1-pos)*local_grad_f(owners_f,2)
        # grad_f[:,2]  = pos*local_grad_f(neighbours_f,3) + (1-pos)*local_grad_f(owners_f,3)
    else:
        io.cfdError(scheme+'not yet implemented, but it will have to be')

    return grad_f


    
    
    
    
    
    
    
    
    