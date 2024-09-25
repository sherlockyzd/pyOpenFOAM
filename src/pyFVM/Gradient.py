import numpy as np
import cfdtool.Math as mth
import cfdtool.IO as io


class Gradient():
    
    def __init__(self, Region, phiName):  
        """
        Handles gradient calculations on the field specified by phiName. First,
        create an instance of the class, then call cfdComputeGradientGaussLinear0,
        which calculates the gradients at each cell. At the end of 
        cfdComputeGradientGaussLinear0, cfdUpdateGradient() is called which updates
        the gradients on the boundary faces.
        这段Python代码定义了一个名为 `Gradient` 的类，它用于计算指定场（由 `phiName` 指定）的梯度。以下是对类的构造器和相关属性的详细解释：
        1. **类定义**：
        - `class Gradient():` 定义了一个名为 `Gradient` 的类。

        2. **构造器**：
        - `def __init__(self, Region, phiName):` 构造器接收两个参数：`Region`（区域案例的实例）和 `phiName`（要计算梯度的场的名称）。

        3. **初始化属性**：
        - 构造器中初始化了一系列属性，包括场名称、场数据、场类型、梯度计算所需的网格信息等。

        4. **场属性**：
        - `self.phiName`: 存储场名称。
        - `self.phi`: 存储从 `Region.fluid` 字典中获取的场数据。
        - `self.type`: 存储场类型。

        5. **梯度计算属性**：
        - `theSize`: 根据场数据的维度确定是标量场还是矢量场。
        - `self.theNumberOfComponents`: 根据场的维度存储组件数量。

        6. **网格几何属性**：
        - `self.elementCentroids`: 存储单元格质心的数组。
        - `self.theNumberOfElements`: 存储网格单元格数量。

        7. **内部面属性**：
        - `self.owners_f` 和 `self.neighbours_f`: 分别存储内部面的所有者和邻居单元格的索引。
        - `self.Sf` 和 `self.g_f`: 分别存储内部面的法向量和权重。
        - `self.iFaces`: 存储内部面的总数。
        - `self.ones`: 存储全1数组，用于内部面的数量。
        - `self.phi_f`: 初始化为零的数组，用于存储面场数据。

        8. **边界面属性**：
        - `self.boundaryPatches`: 存储边界补丁信息。
        - `self.theBoundaryArraySize`: 存储边界元素的数量。
        - `self.iBElements`: 存储边界元素的索引。
        - `self.phi_b`: 存储边界上的场数据。
        - `self.owners_b` 和 `self.Sf_b`: 分别存储边界面的所有者和法向量。

        9. **梯度数组**：
        - `self.phiGrad`: 初始化为零的数组，用于存储内部和边界质心处的梯度值。数组的形状取决于场是标量还是矢量。

        ### 注意事项：
        - 类的构造器中注释掉了 `self.Region`，表明可能在初始设计中考虑过，但在最终实现中未使用。
        - 构造器中直接调用了 `cfdComputeGradientGaussLinear0` 方法来计算梯度，但这个方法的实现没有在代码中给出。
        - `cfdUpdateGradient()` 方法在梯度计算结束后被调用以更新边界面上的梯度，但这个方法的实现也没有在代码中给出。
        - 代码中的注释提供了对每个属性用途的说明，有助于理解每个属性的预期用途。

        `Gradient` 类是CFD模拟中用于梯度计算的关键组件，它提供了一种机制来存储和操作场的梯度数据。通过这种方式，可以方便地访问和更新梯度信息，以实现模拟中的数值求解和边界条件处理。
        """
        
        # self.Region=Region
        self.phiName=phiName
        # self.phi=Region.fluid[self.phiName].phi
        self.type=Region.fluid[self.phiName].type
        
        theSize=Region.fluid[self.phiName].iComponent
        
        if theSize == 3:
            self.theNumberOfComponents=3
        else:
            self.theNumberOfComponents=1
        
        #domain-wide variables
        # self.elementCentroids=Region.mesh.elementCentroids
        # self.theNumberOfElements=Region.mesh.numberOfElements
        
        #interior variables
        # self.owners_f=Region.mesh.interiorFaceOwners
        # self.neighbours_f=Region.mesh.interiorFaceNeighbours
        # self.Sf=Region.mesh.interiorFaceSf
        # self.g_f=Region.mesh.interiorFaceWeights
        self.iFaces=Region.mesh.numberOfInteriorFaces
        # self.ones=np.ones((self.iFaces))
        self.phi_f=np.zeros((self.iFaces,self.theNumberOfComponents))
        # self.volumes = Region.mesh.elementVolumes

        #boundary variables
        # self.boundaryPatches=Region.mesh.cfdBoundaryPatchesArray
        # self.theBoundaryArraySize = Region.mesh.numberOfBElements
        # self.iBElements=np.arange(Region.mesh.numberOfElements,Region.mesh.numberOfElements+Region.mesh.numberOfBFaces,1)
        # self.iBElements = np.arange(Region.mesh.numberOfElements, Region.mesh.numberOfElements+Region.mesh.numberOfBFaces, dtype=int)
        self.phi_b=Region.fluid[self.phiName].phi[Region.mesh.iBElements]
        # self.owners_b=Region.mesh.owners_b
        # self.Sf_b=Region.mesh.Sf_b
        
        ## phiGrad array to hold gradient values at interior and boundary centroids. For scalars this array has shape (self.theNumberOfElements+self.theBoundaryArraySize,3,1) for vectors this array has shape (self.theNumberOfElements+self.theBoundaryArraySize,3,3)
        self.phiGrad=np.zeros((Region.mesh.numberOfElements+Region.mesh.numberOfBElements,3,self.theNumberOfComponents))
        if self.phiName=='U':
            self.phiGrad_TR=np.zeros((Region.mesh.numberOfElements+Region.mesh.numberOfBElements,3,self.theNumberOfComponents))
            if Region.cfdIsCompressible:
                self.phiGrad_Trace=np.zeros((Region.mesh.numberOfElements+Region.mesh.numberOfBElements))
            
        # self.cfdComputeGradientGaussLinear0(Region)
        self.cfdUpdateGradient(Region)
            


    def cfdUpdateGradient(self,Region):
        # %==========================================================================
        # % Routine Description:
        # %   This function updates gradient of the field
        # %--------------------------------------------------------------------------
        gradSchemes = Region.dictionaries.fvSchemes['gradSchemes']['default']
        if gradSchemes=='Gauss linear':
            self.cfdComputeGradientGaussLinear0(Region)
        else:
            io.cfdError('\n%s is incorrect\n', gradSchemes)
  

    def cfdComputeGradientGaussLinear0(self,Region):
        """ 
        This function computes the gradient for a field at the centroids of 
        the elements using a first order gauss interpolation. No correction for 
        non-conjuntionality is applied. 'phi' is the name of a field 
        used when the class is instantiated.
        To-do: Check this in-depth over a number of scenarios
        这段Python代码定义了一个名为`cfdComputeGradientGaussLinear0`的方法，它是`Gradient`类的一部分，用于使用一阶高斯线性插值计算场在单元格质心处的梯度。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdComputeGradientGaussLinear0(self, Region):` 定义了一个实例方法，接收一个参数`Region`，表示区域案例的实例。

        2. **内部面贡献**：
        - 通过遍历场的每个组件（标量或矢量场的每个分量），使用线性插值计算面场`phi_f`。
        - 对于每个内部面，更新面的所有者和邻居单元格质心处的梯度累积器。

        3. **边界面贡献**：
        - 对于边界面上的每个组件，直接将边界面上的场值乘以面法向量，累加到边界面所有者单元格质心的梯度。

        4. **计算体积加权梯度**：
        - 通过将累积的梯度除以单元格体积，计算体积加权平均梯度。

        5. **边界梯度更新**：
        - 遍历边界补丁，根据边界的物理类型（如墙壁、入口、出口、对称性或空），调用相应的方法更新边界梯度。

        6. **设置场梯度**：
        - 将计算得到的梯度赋值给`Region.fluid[self.phiName].phiGrad`，这样在`Region`的流体字典中，指定场的梯度被更新。

        ### 注意事项：
        - 方法中有一些被注释掉的代码，例如`self.cfdUpdateGradient(Region)`和`def cfdUpdateGradient(self, Region):`，这表明这些部分可能还在开发中或已被弃用。
        - 方法中的`To-do`注释表明需要在多种情况下深入检查这个方法。
        - 边界梯度更新部分打印了边界的类型，这有助于调试和验证边界条件的处理。
        - 方法假设`Region`对象包含网格信息和流体场数据，这些信息用于梯度计算和边界条件处理。
        `cfdComputeGradientGaussLinear0`方法在CFD模拟中是计算梯度的关键步骤，它为求解流体动力学方程提供了必要的空间导数信息。通过这种方法，可以获取场在网格质心处的梯度，这对于理解和模拟流体现象的局部变化非常重要。
        """
        
        # interior face contribution, treats vectors as three scalars (u,v,w)
        for iComponent in range(self.theNumberOfComponents):
            #vectorized linear interpolation (same as Interpolate.interpolateFromElementsToFaces('linear'))
            self.phi_f[0:self.iFaces,iComponent]=Region.mesh.interiorFaceWeights*Region.fluid[self.phiName].phi[Region.mesh.interiorFaceNeighbours][:,iComponent]+(np.ones((self.iFaces))-Region.mesh.interiorFaceWeights)*Region.fluid[self.phiName].phi[Region.mesh.interiorFaceOwners][:,iComponent]
            
            for iFace in range(self.iFaces):
                #this only updates phiGrad at the centroids of interior elements, the contribution of boundary faces to boundary centroids is done directly below. 
                #accumlator of phi_f*Sf for the owner centroid of the face  
                # self.phiGrad[self.owners_f[iFace],:,iComponent]=self.phiGrad[self.owners_f[iFace],:,iComponent]+self.phi_f[iFace,iComponent]*self.Sf[iFace]
                self.phiGrad[Region.mesh.interiorFaceOwners[iFace],:,iComponent] += self.phi_f[iFace,iComponent]*Region.mesh.interiorFaceSf[iFace]
                #accumlator of phi_f*Sf for the neighbour centroid of the face  
                # self.phiGrad[self.neighbours_f[iFace],:,iComponent]=self.phiGrad[self.neighbours_f[iFace],:,iComponent]-self.phi_f[iFace,iComponent]*self.Sf[iFace]
                self.phiGrad[Region.mesh.interiorFaceNeighbours[iFace],:,iComponent] -= self.phi_f[iFace,iComponent]*Region.mesh.interiorFaceSf[iFace]

        # Boundary face contributions
        for iComponent in range(self.theNumberOfComponents):
            for iFace in range(Region.mesh.numberOfBFaces):
                # self.phiGrad[self.owners_b[iFace],:,iComponent]=self.phiGrad[self.owners_b[iFace],:,iComponent]+self.phi_b[iFace,iComponent]*self.Sf_b[iFace]
                self.phiGrad[Region.mesh.owners_b[iFace],:,iComponent] += self.phi_b[iFace,iComponent]*Region.mesh.Sf_b[iFace]

        #calculate volume averaged gradient by dividing self.phiGrad by element volumes
        for iComponent in range(self.theNumberOfComponents):
            for iElement in range(Region.mesh.numberOfElements):
                # self.phiGrad[iElement,:,iComponent] = self.phiGrad[iElement,:,iComponent]/self.volumes[iElement]
                self.phiGrad[iElement,:,iComponent] /= Region.mesh.elementVolumes[iElement]
            
        # self.iBElements = np.arange(Region.mesh.numberOfElements, Region.mesh.numberOfElements+Region.mesh.numberOfBFaces, dtype=int)
        
        self.phiGrad[Region.mesh.iBElements,:,:] = self.phiGrad[Region.mesh.owners_b,:,:]
        
        self.cfdUpdateBoundaryGradient(Region)
        if self.phiName=='U':
            self.cfdTransposeComputeGradient(Region)
            if Region.cfdIsCompressible:
                self.cfdTraceGradient(Region)

    def cfdTransposeComputeGradient(self,Region):
        for iElement in range(Region.mesh.numberOfElements+Region.mesh.numberOfBElements):
            #vectorized linear interpolation (same as Interpolate.interpolateFromElementsToFaces('linear'))
            self.phiGrad_TR[iElement,:,:]=np.transpose(self.phiGrad[iElement,:,:])
        # pass
    def cfdTraceGradient(self,Region):
        for iElement in range(Region.mesh.numberOfElements+Region.mesh.numberOfBElements):
            #vectorized linear interpolation (same as Interpolate.interpolateFromElementsToFaces('linear'))
            self.phiGrad_Trace[iElement]=np.trace(self.phiGrad[iElement,:,:])
        # pass

    def cfdUpdateBoundaryGradient(self,Region):
        """ Prints the boundary type and then assigns the calculated phiGrad field to self.Region.fluid[self.phiName].phiGrad
        """
        # % BOUNDARY Gradients
        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():
        # % Get Physical and Equation Boundary Conditions
            thePhysicalType = theBCInfo['type']
            # theBCType =Region.fluid[self.phiName].boundaryPatchRef[iBPatch]['type'] 
            # % BC info
            if thePhysicalType=='wall':
                self.updateWallGradients(Region,iBPatch)
            elif thePhysicalType=='inlet':
                self.updateInletGradients(Region,iBPatch)
            elif thePhysicalType=='outlet':
                self.updateOutletGradients(Region,iBPatch)
            elif thePhysicalType=='symmetry' or thePhysicalType=='empty':
                self.updateSymmetryGradients(Region,iBPatch)
            else:
                io.cfdError('Boundary condition not recognized')

        # for patch, patchInfo in Region.mesh.cfdBoundaryPatchesArray.items():
        #     thePhysicalType =patchInfo['type']
        #     if thePhysicalType =='wall':
        #         self.updateWallGradients(Region,patch)
        #         print('This patch is of type %s' %thePhysicalType)
        #     elif thePhysicalType =='inlet':
        #         self.updateInletGradients(Region,patch)
        #         print('This patch is of type %s' %thePhysicalType)
        #     elif thePhysicalType =='outlet':
        #         self.updateOutletGradients(Region,patch)
        #         print('This patch is of type %s' %thePhysicalType)
        #     elif thePhysicalType =='symmetry' or 'empty':
        #         print('This patch is of type %s' %thePhysicalType)
        #     else:
        #         print('Boundary condition is not correctly spelled or unrecognized ...')
        #         break 
        # Region.fluid[self.phiName].phiGrad=self.phiGrad

    def updateWallGradients(self,Region,patch):
        owners_b=Region.mesh.cfdBoundaryPatchesArray[patch]['owners_b']
        # faceCentroids_b=Region.mesh.cfdBoundaryPatchesArray[patch]['faceCentroids']
        iBElements=Region.mesh.cfdBoundaryPatchesArray[patch]['iBElements']
        numberOfBFaces = Region.mesh.cfdBoundaryPatchesArray[patch]['numberOfBFaces']
        startBFace = Region.mesh.cfdBoundaryPatchesArray[patch]['startFaceIndex']
        # startBElement = startBFace - self.iFaces + Region.mesh.numberOfElements
        # endBElement = startBElement + numberOfBFaces-1
        grad_b=np.zeros((numberOfBFaces, 3,self.theNumberOfComponents))
        # for iComponent in range(self.theNumberOfComponents):
        #     for iBFace in range(numberOfBFaces):
        #         iBElement = startBElement+iBFace
        #         iOwner = owners_b[iBFace]
        #         Cf=faceCentroids_b[iBFace]
        #         C = self.elementCentroids[iOwner]
        #         dCf=Cf-C
        #         e=dCf/mth.cfdMag(dCf)
        #         grad_b[iBFace,:,iComponent] = self.phiGrad[iOwner,:,iComponent] - (self.phiGrad[iOwner,:,iComponent]*e)*e + ((self.phi[iBElement,iComponent] - self.phi[iOwner,iComponent])/mth.cfdMag(dCf))*e
        for iBFace in range(numberOfBFaces):
            iBElement = iBElements[iBFace]
            iOwner = owners_b[iBFace]
            # Cf=faceCentroids_b[iBFace]
            # C = Region.mesh.elementCentroids[iOwner]
            # dCf=Cf-C
            dCf=Region.mesh.faceCF[startBFace+iBFace]
            e=dCf/mth.cfdMag(dCf)
            for iComponent in range(self.theNumberOfComponents):    
                #保留切向分量，再考虑边界面的相邻单元的梯度计算 书籍《The Finite Volume Method in Computational Fluid Dynamics》Page 289页
                grad_b[iBFace,:,iComponent] = self.phiGrad[iOwner,:,iComponent] - (self.phiGrad[iOwner,:,iComponent]*e)*e + ((Region.fluid[self.phiName].phi[iBElement,iComponent] - Region.fluid[self.phiName].phi[iOwner,iComponent])/mth.cfdMag(dCf))*e        
        self.phiGrad[iBElements,:,:]=grad_b
                
   
    def updateInletGradients(self,Region,patch):
        '''
        这段Python代码定义了一个名为`updateInletGradients`的方法，它用于更新边界条件为入口（inlet）的梯度。这个方法是`Gradient`类的一部分，用于计算和设置边界面上的梯度，以确保物理上合理的流动条件。以下是对这个方法的详细解释：
        1. **方法定义**：
        - `def updateInletGradients(self, Region, patch):` 定义了一个实例方法，接收两个参数：`Region`（区域案例的实例）和`patch`（当前处理的边界补丁的名称或索引）。

        2. **获取边界补丁信息**：
        - 从`Region.mesh.cfdBoundaryPatchesArray`中获取与`patch`相关的所有者单元格、面质心、边界元素索引、边界面数量和起始面索引。

        3. **计算边界元素索引范围**：
        - `startBElement`：计算边界单元格的起始索引。
        - `endBElement`：计算边界单元格的结束索引。

        4. **初始化边界梯度数组**：
        - `grad_b`：初始化一个用于存储边界面上梯度的三维数组。

        5. **计算边界梯度**：
        - 通过双重循环遍历每个分量`iComponent`和每个边界面`iBFace`。
        - 对于每个边界面，计算面质心`Cf`与所有者单元格质心`C`之间的向量`dCf`。
        - 计算`dCf`的单位向量`e`。

        6. **设置边界梯度**：
        - 根据所有者单元格的梯度、面质心与单元格质心之间的距离，以及边界面上的场值与所有者单元格场值之间的差，计算边界梯度`grad_b`。

        7. **更新`phiGrad`**：
        - 将计算得到的边界梯度`grad_b`赋值给`self.phiGrad[iBElements, :, :]`，这样边界面上的梯度就被更新了。

        ### 注意事项：
        - 方法中使用了`mth.cfdMag`函数来计算向量的模长，这个函数可能在类的其他部分或外部模块定义。
        - `self.phiGrad`和`self.phi`分别存储了梯度和场值，这些属性在类的构造器中被初始化。
        - 代码中的计算考虑了边界面上的流动特性，特别是入口处的梯度条件，这对于模拟入口处的流动非常重要。
        - 该方法假设`Region`对象包含网格信息和流体场数据，这些信息用于梯度计算和边界条件处理。

        `updateInletGradients`方法在CFD模拟中是应用入口边界条件的关键步骤，它确保了在入口处梯度的物理合理性，从而影响了模拟的准确性和稳定性。
        '''
        self.updateWallGradients(Region,patch)
        # owners_b=Region.mesh.cfdBoundaryPatchesArray[patch]['owners_b']
        # faceCentroids_b=Region.mesh.cfdBoundaryPatchesArray[patch]['faceCentroids']
        
        # iBElements=Region.mesh.cfdBoundaryPatchesArray[patch]['iBElements']
        
        # numberOfBFaces = Region.mesh.cfdBoundaryPatchesArray[patch]['numberOfBFaces']
        # startBFace = Region.mesh.cfdBoundaryPatchesArray[patch]['startFaceIndex']
        
        # startBElement = startBFace - self.iFaces + Region.mesh.numberOfElements
        # # endBElement = startBElement + numberOfBFaces-1
        
        # grad_b=np.zeros((numberOfBFaces, 3,self.theNumberOfComponents))
        
        # for iBFace in range(numberOfBFaces):
        #     iBElement = startBElement+iBFace
        #     iOwner = owners_b[iBFace]
            
        #     Cf=faceCentroids_b[iBFace]
        #     C = Region.mesh.elementCentroids[iOwner]
        #     dCf=Cf-C
        #     e=dCf/mth.cfdMag(dCf)
        #     for iComponent in range(self.theNumberOfComponents):    
        #         #保留切向分量，再考虑边界面的相邻单元的梯度计算  书籍《The Finite Volume Method in Computational Fluid Dynamics》Page 289页 及图9.11(c)
        #         grad_b[iBFace,:,iComponent] = self.phiGrad[iOwner,:,iComponent] - (self.phiGrad[iOwner,:,iComponent]*e)*e + ((Region.fluid[self.phiName].phi[iBElement,iComponent] - Region.fluid[self.phiName].phi[iOwner,iComponent])/mth.cfdMag(dCf))*e

        # self.phiGrad[iBElements,:,:]=grad_b       
                
    def updateOutletGradients(self,Region,patch):
        self.updateWallGradients(Region,patch)
        # owners_b=Region.mesh.cfdBoundaryPatchesArray[patch]['owners_b']
        # faceCentroids_b=Region.mesh.cfdBoundaryPatchesArray[patch]['faceCentroids']
        
        # iBElements=Region.mesh.cfdBoundaryPatchesArray[patch]['iBElements']
        
        # numberOfBFaces = Region.mesh.cfdBoundaryPatchesArray[patch]['numberOfBFaces']
        # startBFace = Region.mesh.cfdBoundaryPatchesArray[patch]['startFaceIndex']
        
        # startBElement = startBFace - self.iFaces + Region.mesh.numberOfElements
        # # endBElement = startBElement + numberOfBFaces-1
        
        # grad_b=np.zeros((numberOfBFaces, 3,self.theNumberOfComponents))        
        
        # # for iComponent in range(self.theNumberOfComponents):
        # #     for iBFace in range(numberOfBFaces):
        # #         iBElement = startBElement+iBFace
        # #         iOwner = owners_b[iBFace]
        # #         Cf=faceCentroids_b[iBFace]
        # #         C = self.elementCentroids[iOwner]
        # #         dCf=Cf-C
        # #         e=dCf/mth.cfdMag(dCf)
                
        # #         grad_b[iBFace,:,iComponent] = self.phiGrad[iOwner,:,iComponent] - (self.phiGrad[iOwner,:,iComponent]*e)*e + ((self.phi[iBElement,iComponent] - self.phi[iOwner,iComponent])/mth.cfdMag(dCf))*e
        # for iBFace in range(numberOfBFaces):
        #     iBElement = startBElement+iBFace
        #     iOwner = owners_b[iBFace]
        #     Cf=faceCentroids_b[iBFace]
        #     C = Region.mesh.elementCentroids[iOwner]
        #     dCf=Cf-C
        #     e=dCf/mth.cfdMag(dCf)
        #     for iComponent in range(self.theNumberOfComponents):    
        #         #保留切向分量，再考虑边界面的相邻单元的梯度计算 书籍《The Finite Volume Method in Computational Fluid Dynamics》Page 289页
        #         grad_b[iBFace,:,iComponent] = self.phiGrad[iOwner,:,iComponent] - (self.phiGrad[iOwner,:,iComponent]*e)*e + ((Region.fluid[self.phiName].phi[iBElement,iComponent] - Region.fluid[self.phiName].phi[iOwner,iComponent])/mth.cfdMag(dCf))*e        

        # self.phiGrad[iBElements,:,:]=grad_b 

    def updateSymmetryGradients(self,Region,patch):
        pass


    def cfdGetGradientSubArrayForInterior(self,Region,*args):
        if self.type == "surfaceScalarField":
            self.phiGradInter=self.phiGrad[0:Region.mesh.numberOfInteriorFaces,:,0]
        elif self.type == "volScalarField":
            self.phiGradInter=self.phiGrad[0:Region.mesh.numberOfInteriorFaces,:,0]
        elif self.type == "volVectorField":
            if args:
                iComponet=args[0]
                self.phiGradInter=self.phiGrad[0:Region.mesh.numberOfInteriorFaces,:,iComponet]
            else:
                self.phiGradInter=self.phiGrad[0:Region.mesh.numberOfInteriorFaces,:,:]
        if self.phiName=='U':
            if args:
                iComponet=args[0]
                self.phiGradInter_TR=self.phiGrad_TR[0:Region.mesh.numberOfInteriorFaces,:,iComponet]
            else:
                self.phiGradInter_TR=self.phiGrad_TR[0:Region.mesh.numberOfInteriorFaces,:,:]
        return self.phiGradInter

