import cfdtool.IO as io
import numpy as np
import pyFVM.Interpolate as interp
import cfdtool.Math as mth


class Field():
    
    def __init__(self,Region,fieldName, fieldType):
        """Creates an empty field class that will be populated later on.
        Detects if field is either type volScalar, volVector, surfaceScalar, or 
        surfaceVector3 and creates an empty container with an adequate number of rows
        and columns (i.e., scalar = 1 column, vector = 3 columns) to hold field data.
        Attributes:
        Example usage:
        这段Python代码定义了一个名为`Field`的类，它用于创建和初始化不同类型的CFD（计算流体动力学）场数据结构。下面是对这个类的详细解释：

        1. **类的初始化方法**：
        - `def __init__(self, Region, fieldName, fieldType):` 是类的构造器，用于创建类的实例。它接收三个参数：`Region`（区域信息），`fieldName`（场的名称），和`fieldType`（场的类型）。
        - `__init__`方法的文档字符串说明了这个方法的用途，但具体的属性和使用示例部分是空的。
        这个`Field`类是CFD模拟中用于管理场数据的基础结构，它可以根据不同的场类型创建适当大小的数据容器。通过初始化这些容器，CFD模拟可以存储和操作场数据，如速度、压力、温度等。

        2. **实例属性**：
        - `self.Region`：存储区域信息。
        - `self.name`：存储场的名称。
        - `self.type`：存储场的类型，可以是`'volScalarField'`、`'volVectorField'`、`'surfaceScalarField'`或`'surfaceVector3Field'`。
        - `self.dimensions`：用于存储场的维度信息，当前为空列表。
        - `self.boundaryPatchRef`：用于存储边界补丁的引用，当前为空字典。

        3. **场类型特定逻辑**：
        - 根据`self.type`的值，构造器会初始化不同的属性：
            - 对于体积标量场（`volScalarField`），创建一个大小为`self.theInteriorArraySize+self.theBoundaryArraySize`乘以1的NumPy数组`self.phi`，用于存储场值，以及一个相同大小的数组`self.phi_old`用于存储上一次迭代的场值。
            - 对于体积矢量场（`volVectorField`），数组大小相同，但是有3列，用于存储三个方向的矢量分量。
            - 对于表面标量场（`surfaceScalarField`），数组大小基于内部面和边界面的数量。
            - 对于表面矢量场（`surfaceVector3Field`），数组同样有3列，用于存储表面每个点的矢量分量。

        4. **打印信息**：
        - 每种类型的场初始化后，会打印一条消息，说明创建的是哪种类型的场。

        5. **迭代和时间步存储**：
        - `self.prevIter`和`self.prevTimeStep`：字典，用于存储上一次迭代和时间步的场数据。

        6. **更新比例尺方法**：
        - `self.cfdUpdateScale()`：一个方法，可能用于更新场的比例尺或相关属性，但具体实现未在代码中给出。

        7. **梯度存储**：
        - `self.phiGrad`：一个空列表，可能用于存储场的梯度信息。

        8. **注释掉的代码**：
        - 代码中有几行被注释掉的代码，这些代码看起来是用于初始化`self.phi`和`self.phi_f`的，但实际使用中选择了使用NumPy数组。
        """
        
        # self.Region=Region
        self.name = fieldName
        self.type = fieldType
        self.dimensions=[]
        '''
        在 OpenFOAM 中，`dimensions` 用于定义物理量的维度。每个物理量的维度是一个具有七个分量的数组，分别表示它在以下七个基础单位下的幂次：
        1. 质量（mass）
        2. 长度（length）
        3. 时间（time）
        4. 温度（temperature）
        5. 电流（electric current）
        6. 物质的量（amount of substance）
        7. 发光强度（luminous intensity）
        这些单位遵循国际单位制（SI）。`dimensions` 主要用于确保单位的正确性，并帮助 OpenFOAM 在计算过程中执行单位检查。
        dimensions [M L T Θ I N J] 每个字母表示一个单位的幂次，按照如下解释：
        - `M`: 质量，单位 kg
        - `L`: 长度，单位 m
        - `T`: 时间，单位 s
        - `Θ`: 温度，单位 K
        - `I`: 电流，单位 A
        - `N`: 物质的量，单位 mol
        - `J`: 发光强度，单位 cd
        在 OpenFOAM 中，物理量的维度通常写在文件开头。
        速度的维度（单位 m/s）dimensions [0 1 -1 0 0 0 0];
        压力的维度（单位 N/m² 或者 kg/(m·s²)）dimensions [1 -1 -2 0 0 0 0];
        体积：`[0 3 0 0 0 0 0]` （m³）
        力：`[1 1 -2 0 0 0 0]` （N 或 kg·m/s²）
        密度：`[1 -3 0 0 0 0 0]` （kg/m³）
        通过这种方式，OpenFOAM 可以确保所有物理量的单位在计算中保持一致。如果在模拟过程中出现单位不匹配的问题，OpenFOAM 将给出错误信息。
        '''
        self.boundaryPatchRef={}
        
        if self.type == 'volScalarField':
            self.theInteriorArraySize = Region.mesh.numberOfElements
            self.theBoundaryArraySize = Region.mesh.numberOfBElements
            self.iComponent=1

        if self.type == 'volVectorField':
            self.theInteriorArraySize = Region.mesh.numberOfElements
            self.theBoundaryArraySize = Region.mesh.numberOfBElements
            self.iComponent=3
        
        if self.type == 'surfaceScalarField':
            self.theInteriorArraySize = Region.mesh.numberOfInteriorFaces
            self.theBoundaryArraySize = Region.mesh.numberOfBFaces
            self.iComponent=1
            
        if self.type == 'surfaceVectorField':
            self.theInteriorArraySize = Region.mesh.numberOfInteriorFaces
            self.theBoundaryArraySize =Region.mesh.numberOfBFaces
            self.iComponent=3

        self.phi = np.zeros((self.theInteriorArraySize+self.theBoundaryArraySize, self.iComponent))
        self.phi_old = np.zeros((self.theInteriorArraySize+self.theBoundaryArraySize, self.iComponent))
        self.cfdUpdateScale(Region)

    def initializeMdotFromU(self,Region):
        """
        初始化面上的质量流量 phi，通过将速度 U 和密度 rho 从单元格插值到面上，并计算质量流量。
        
        参数：
        - Region: 包含流体区域信息的对象。
        
        过程：
        1. 使用线性插值方法将速度 U 和密度 rho 从单元格插值到面上。
        2. 计算每个面的质量流量 phi = rho_f * (Sf ⋅ U_f)。
        
        返回：
        - self.phi: 形状为 (numberOfFaces, numberOfComponents) 的质量流量数组。
        """
        U_f=interp.cfdinterpolateFromElementsToFaces(Region,'linear',Region.fluid['U'].phi)
        rho_f=interp.cfdinterpolateFromElementsToFaces(Region,'linear',Region.fluid['rho'].phi)
        Sf=Region.mesh.faceSf
        #calculate mass flux through faces, 必须写成二维数组的形式，便于后续与U的数组比较运算!
            # 确保插值结果的形状匹配
        if U_f.ndim != 2 or rho_f.ndim != 2:
            io.cfdError('插值后的 U_f 和 rho_f 必须是二维数组')
        
        if Sf.shape[0] != U_f.shape[0] or Sf.shape[0] != rho_f.shape[0]:
            io.cfdError('Sf、U_f 和 rho_f 的面数量不匹配')
        
        # 计算通量 Sf ⋅ U_f，得到每个面的流量，形状为 (nFaces, 1)
        flux =mth.cfdDot(Sf, U_f)[:, np.newaxis]  # 使用 np.einsum 进行高效的点积计算
        
        # 计算质量流量 phi = rho_f * flux，形状为 (nFaces, 1)
        self.phi = rho_f * flux  # 形状: (nFaces, 1)
        
        # 检查 phi 是否包含非有限值（如 NaN 或无穷大）
        if not np.all(np.isfinite(self.phi)):
            io.cfdError('计算得到的质量流量 phi 包含非有限值')

    def setDimensions(self,dimensions):
        self.dimensions=dimensions

    def setPreviousTimeStep(self,*args):
        if args:
            iComponent = args[0]
            self.phi_old[:,iComponent]=self.phi[:,iComponent]
        else:
            self.phi_old=self.phi

    def setPreviousIter(self,*args):
        if args:
            iComponent = args[0]
            self.prevIter[:,iComponent]=self.phi[:,iComponent]
        else:
            self.prevIter=self.phi
        
    def cfdGetSubArrayForInterior0(self,Region,*args):    
        if self.type == 'surfaceScalarField':
            self.phiInteriorSubArray = self.phi[0:Region.mesh.numberOfInteriorFaces]
        elif self.type == 'volScalarField':
            self.phiInteriorSubArray = self.phi[0:Region.mesh.numberOfElements]    
        elif self.type == 'volVectorField':
            if args:
                iComponent = args[0]
                self.phiInteriorSubArray = self.phi[0:Region.mesh.numberOfElements,iComponent] 
            else:
                self.phiInteriorSubArray = self.phi[0:Region.mesh.numberOfElements, :]

    def cfdGetSubArrayForBoundary0(self,Region,*args):
        if self.type == 'surfaceScalarField':
            self.phiBoundarySubArray = self.phi[Region.mesh.numberOfInteriorFaces:Region.mesh.numberOfFaces]
        elif self.type == 'volScalarField':
            self.phiBoundarySubArray = self.phi[Region.mesh.numberOfElements:Region.mesh.numberOfElements+Region.mesh.numberOfBElements]
        elif self.type == 'volVectorField':
            if args:
                iComponent = args[0]
                self.phiBoundarySubArray = self.phi[Region.mesh.numberOfElements:Region.mesh.numberOfElements+Region.mesh.numberOfBElements,iComponent] 
            else:
                self.phiBoundarySubArray = self.phi[Region.mesh.numberOfElements:Region.mesh.numberOfElements+Region.mesh.numberOfBElements, :]

    def cfdGetPrevTimeStepSubArrayForInterior0(self,Region,*args):    
        if self.type == 'surfaceScalarField':
            self.phi_oldInteriorSubArray = self.phi_old[0:Region.mesh.numberOfInteriorFaces]
        elif self.type == 'volScalarField':
            self.phi_oldInteriorSubArray = self.phi_old[0:Region.mesh.numberOfElements]
        elif self.type == 'volVectorField':
            if args:
                iComponent = args[0]
                self.phi_oldInteriorSubArray = self.phi_old[0:Region.mesh.numberOfElements,iComponent]
            else:
                self.phi_oldInteriorSubArray = self.phi_old[0:Region.mesh.numberOfElements,:]

    def cfdGetPrevTimeStepSubArrayForBoundary0(self,Region,*args):    
        if self.type == 'surfaceScalarField':
            self.phiBoundarySubArray = self.phi_old[Region.mesh.numberOfInteriorFaces:Region.mesh.numberOfFaces]
        elif self.type == 'volScalarField':
            self.phiBoundarySubArray = self.phi_old[Region.mesh.numberOfInteriorFaces:Region.mesh.numberOfFaces]
        elif self.type == 'volVectorField':
            if args:
                iComponent = args[0]
                self.phiBoundarySubArray = self.phi_old[Region.mesh.numberOfElements:Region.mesh.numberOfElements+Region.mesh.numberOfBElements,iComponent] 
            else:
                self.phiBoundarySubArray = self.phi_old[Region.mesh.numberOfElements:Region.mesh.numberOfElements+Region.mesh.numberOfBElements,:]
        
        
    def cfdUpdateScale(self,Region):
        """Update the min, max and scale values of a field in Region
        Attributes:
            
            Region (str): the cfd Region.
            field (str): the field in Region.fields
            
        Example usage:
            
            cfdUpdateScale(Region,'rho')

        这段Python代码定义了一个名为`cfdUpdateScale`的方法，它用于更新CFD（计算流体动力学）场的最小值、最大值和比例尺。这个方法似乎是设计为`Field`类的一个实例方法。下面是对这段代码的详细解释：

        1. **方法定义和文档字符串**：
        - `def cfdUpdateScale(self):` 定义了一个实例方法，没有接收额外的参数，而是使用实例属性。
        - 文档字符串说明了方法的用途，预期的属性和使用示例。

        2. **计算场的模**：
        - `theMagnitude = mth.cfdMag(self.phi)`：使用`mth.cfdMag`函数计算场`self.phi`的模。这里`mth`可能是一个包含数学工具的模块，`self.phi`是场数据。

        3. **处理向量场和标量场**：
        - 使用`try...except`结构来区分场是向量场还是标量场。
        - 如果`theMagnitude`可迭代，说明`self.phi`是一个向量场，计算其模的最大值`phiMax`和最小值`phiMin`。
        - 如果`theMagnitude`不可迭代（引发`TypeError`），说明`self.phi`是一个标量场，其最大值和最小值就是`theMagnitude`本身。

        4. **特定场的缩放逻辑**：
        - 根据场的名称`self.name`，使用不同的逻辑来确定比例尺`phiScale`：
        - 如果场名为`'p'`（压力场），计算动力压力`p_dyn`，并将其与`phiMax`比较，取较大者作为比例尺。
        - 如果场名为`'U'`（速度场），取区域长度尺度`self.Region.lengthScale`和`phiMax`的较大者作为比例尺。
        - 对于其他场，直接使用`phiMax`作为比例尺。

        5. **更新实例属性**：
        - 更新实例的`max`、`min`和`scale`属性，分别存储场的最大值、最小值和比例尺。

        6. **代码中的一些问题**：
        - 在计算动力压力`p_dyn`时，使用了`vel_scale^2`，这应该是一个指数运算，但在Python中应写作`vel_scale ** 2`。
        - 文档字符串中提到的属性`Region`和`field`在方法定义中并未使用，这可能是文档字符串的一个遗留问题或错误。

        7. **打印最大值**：
        - `print(phiMax)`：在控制台打印出最大值，这可能是用于调试的输出。

        这个方法的目的是确保CFD模拟中的每个场都有一个合适的比例尺，这对于后续的计算和可视化是很重要的。通过比较场的最大值和其他相关参数，这个方法为每个场设置了一个合适的比例尺。
        """    
        theMagnitude = mth.cfdMag(self.phi)
        
        try:
            #see if it is a vector
            iter(theMagnitude)
            phiMax=max(mth.cfdMag(self.phi))
            phiMin=min(mth.cfdMag(self.phi))
            # print(phiMax)
        except TypeError:
            #knows it is scalar, so ...
            phiMax=theMagnitude
            phiMin=theMagnitude

        if self.name=='p' or self.name=='pprime':
            vel_scale = Region.fluid['U'].scale
            rho_scale = Region.fluid['rho'].scale
            p_dyn = 0.5 * rho_scale * (vel_scale**2)
            phiScale = max(phiMax,p_dyn)

        elif self.name=='U':
            phiScale = max(Region.lengthScale,phiMax)
        else: 
            phiScale = phiMax

        self.max=phiMax
        self.min=phiMin
        self.scale=phiScale            

    def cfdfieldUpdate(self,Region,*args):
        #只在计算一开始运行一次，在迭代运行之前
        self.updateFieldForAllBoundaryPatches(Region)
        self.cfdfieldUpdateGradient_Scale(Region)

    def cfdfieldUpdateGradient_Scale(self,Region):
        #更新梯度，和比例尺
        self.phiGrad.cfdUpdateGradient(Region)
        self.cfdUpdateScale(Region)

    def cfdCorrectField(self,Region,iComponent):
    #==========================================================================
    #    Correct Fluid Field 更新场值
    #--------------------------------------------------------------------------
        self.cfdCorrectForInterior(Region,iComponent)
        self.cfdCorrectForBoundaryPatches(Region,iComponent)

    def cfdCorrectForInterior(self,Region,iComponent):
        theNumberOfElements = Region.coefficients.NumberOfElements
        if self.name=='p':
            self.setupPressureCorrection(Region)
            urfP=Region.dictionaries.fvSolution['relaxationFactors']['fields']['p']
            self.phi[0:theNumberOfElements,iComponent] += urfP*Region.coefficients.dphi
        else:
            self.phi[0:theNumberOfElements,iComponent] += Region.coefficients.dphi

    def setupPressureCorrection(self,Region):
        # Check if needs reference pressure
        if Region.mesh.cfdIsClosedCavity:
            # Timesolver=Region.Timesolver
            # Get the pressure at the fixed value
            try:
                pRefCell = int(Region.dictionaries.fvSolution[Region.Timesolver]['pRefCell'])
                pRefValue = Region.coefficients.dphi[pRefCell]
                Region.coefficients.dphi -= pRefValue
            except KeyError:
                io.cfdError('pRefCell not found')

    def cfdCorrectForBoundaryPatches(self,Region,iComponent):
        '''
        Correct velocity field at cfdBoundary patches
        '''
        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():
            # Get Physical and Equation Boundary Conditions
            thePhysicalType = theBCInfo['type']
            theBCType =self.boundaryPatchRef[iBPatch]['type']
            # WALL
            if thePhysicalType=='wall':
                if theBCType=='noSlip':
                    continue
                elif theBCType=='slip':
                    self.correctSlipWall(Region,iBPatch,iComponent)
                elif theBCType=='fixedValue':
                    # correctVelocitySlipWall(Region,iBPatch,iComponent)
                    continue
                elif theBCType=='zeroGradient':
                    self.correctZeroGradient(Region,iBPatch,iComponent)
                else:
                    io.cfdError(thePhysicalType+' Condition '+theBCType+' not implemented')
            # INLET
            elif thePhysicalType=='inlet':
                if theBCType=='fixedValue':
                    continue
                elif theBCType=='zeroGradient' or theBCType=='inlet':
                    self.correctZeroGradient(Region,iBPatch,iComponent)
                else:
                    io.cfdError(thePhysicalType+' Condition '+theBCType+' not implemented')
            # OUTLET
            elif thePhysicalType=='outlet':
                if theBCType=='outlet' or theBCType=='zeroGradient':
                    self.correctZeroGradient(Region,iBPatch,iComponent)
                elif theBCType=='fixedValue':
                    continue
                else:
                    io.cfdError(thePhysicalType+' Condition '+theBCType+' not implemented')
            # SYMMETRY/EMPTY
            elif thePhysicalType=='symmetry' or thePhysicalType=='symmetryPlane' or thePhysicalType=='empty':
                self.correctSymmetry(Region,iBPatch,iComponent)
            else:
                io.cfdError(thePhysicalType+' Condition '+theBCType+' not implemented')

    def correctSlipWall(self,Region,iBPatch,iComponent):
        # Get info
        iBElements = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        # Sb = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
        n = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facen']
        # Get Vector Field at Boundary
        # theVelocityField = cfdGetMeshField('U')
        # U_C = Region.fluid['U'].phi
        # Update field by imposing the parallel to wall component of velocity. This
        # is done by removing the normal component
        phi_normal = mth.cfdDot(self.phi[owners_b,:],n)
        self.phi[iBElements,iComponent] -=  phi_normal*n[:,iComponent]
        # Store
        # Region.fluid['U'].phi[iBElements,iComponent] = U_C[iBElements,iComponent]


    def correctZeroGradient(self,Region,iBPatch,iComponent):
        # Get info
        iBElements = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        iOwners = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        # Copy value from owner cell
        self.phi[iBElements, iComponent] = self.phi[iOwners, iComponent]

    def correctSymmetry(self,Region,iBPatch,iComponent):
        # Get info
        iBElements = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        # Sb = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
        if iComponent !=-1:
            n = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facen']
            # Get Vector Field at Boundary
            # # Update field by imposing the parallel to wall component of velocity. This
            # # is done by removing the normal component
            phi_normal = mth.cfdDot(self.phi[owners_b,:],n)
            self.phi[iBElements,iComponent] -=  phi_normal*n[:,iComponent]
        else:
            self.phi[iBElements] =  self.phi[owners_b]

    def updateFieldForAllBoundaryPatches(self,Region):
        """Updates the field values of the faces on each of the boundary patches.
        这段Python代码定义了一个名为`updateFieldForAllBoundaryPatches`的方法，它用于更新CFD（计算流体动力学）模拟中边界补丁上的所有场值。这个方法可能是`Field`类的一个实例方法。下面是对这段代码的详细解释：

        1. **方法定义**：
        - `def updateFieldForAllBoundaryPatches(self):` 定义了一个实例方法，没有接收额外的参数，而是使用实例属性。

        2. **遍历边界补丁**：
        - `for iBPatch, theBCInfo in self.Region.mesh.cfdBoundaryPatchesArray.items():` 遍历`self.Region.mesh.cfdBoundaryPatchesArray`字典，其中包含边界补丁的信息。

        3. **获取边界条件信息**：
        - `self.iBPatch = iBPatch`：将当前边界补丁的索引保存在实例变量中，供其他函数使用。
        - `thePhysicalPatchType = theBCInfo['type']`：从边界补丁信息中获取物理补丁类型。
        - `theBCType = self.boundaryPatchRef[iBPatch]['type']`：从实例的`boundaryPatchRef`字典中获取边界条件类型。

        4. **根据边界类型更新场值**：
        - 根据`thePhysicalPatchType`和`theBCType`的值，执行不同的更新策略：
            - 对于`wall`（墙）补丁，如果边界条件是`fixedValue`（固定值）或`zeroGradient`（零梯度），则调用`updateFixedValue`或`updateZeroGradient`方法。
            - 对于`inlet`（入口）补丁，根据边界条件类型，调用`updateFixedValue`或`updateZeroGradient`方法。
            - 对于`outlet`（出口）补丁，如果边界条件是`fixedValue`或特殊的`outlet`类型，则调用`updateFixedValue`；如果是`zeroGradient`，则调用`updateZeroGradient`。
            - 对于`symmetry`（对称性）和`empty`（空）补丁，通常调用`updateZeroGradient`方法；对于矢量场，还可能调用`updateSymmetry`方法。

        5. **错误处理**：
        - 如果边界条件类型未定义或不正确，打印错误消息并根据情况可能退出程序（`sys.exit()`）。

        6. **代码中的一些问题**：
        - 方法中有一些重复的逻辑，例如对于`inlet`和`outlet`补丁，`fixedValue`和`zeroGradient`的处理是相同的，这可能是一个错误或遗漏。
        - 对于`volVectorField`类型，`updateSymmetry`方法被调用，但这个方法在代码中没有给出定义，可能是在类的其他部分定义。

        7. **注释和文档**：
        - 方法的文档字符串是空的，没有提供方法的具体描述或使用示例。

        这个方法的目的是确保CFD模拟中每个边界补丁上的场值都根据其物理条件和边界条件正确设置。这通常涉及到设置固定值、零梯度或对称性条件，以保证模拟的准确性和稳定性。
        """
        
        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():
            
            # self.iBPatch = iBPatch #for using in other functions
            
            #boundary type for patch defined in 'boundary' file in polyMesh folder
            thePhysicalPatchType=theBCInfo['type']
            
            #boundary type defined for same patch in "0" file
            #  theBCType=self.boundaryPatchRef[iBPatch]['type']
            try:
                theBCType = self.boundaryPatchRef[iBPatch]['type']
            except KeyError:
                # 如果 iBPatch 不是 self.boundaryPatchRef 的一个键
                theBCType = 'zeroGradient'
                self.boundaryPatchRef[iBPatch] = {'type': theBCType}
                # 可以在这里记录日志或进行其他错误处理
            except TypeError:
                # 如果 self.boundaryPatchRef[iBPatch] 没有 'type' 属性或者不是字典
                theBCType = 'zeroGradient'
                self.boundaryPatchRef[iBPatch] = {'type': theBCType}
                # 同样可以在这里记录日志或进行其他错误处理

            if thePhysicalPatchType == 'wall':
                if theBCType == 'fixedValue':
                    self.updateFixedValue(Region,iBPatch)
                elif theBCType == 'noSlip':
                    self.updateNoSlip(Region,iBPatch)
                elif theBCType == 'slip' :
                    if self.type == 'volScalarField':
                        self.updateZeroGradient(Region,iBPatch)
                    elif self.type == 'volVectorField':
                        self.updateSymmetry(Region,iBPatch)
                elif theBCType == 'zeroGradient'  :
                    # if self.type == 'volScalarField':
                    #     self.updateZeroGradient(Region)
                    # if self.type == 'volVectorField':
                    self.updateZeroGradient(Region,iBPatch)
                else:
                    io.cfdError('The '+iBPatch+' patch type '+theBCType+' is ill defined or missing!')
                
            elif thePhysicalPatchType == 'inlet':
                if theBCType == 'fixedValue':
                    # if self.type == 'volScalarField':
                    #     self.updateFixedValue(Region)
                    # if self.type == 'volVectorField':
                    self.updateFixedValue(Region,iBPatch)  
                elif theBCType == 'zeroGradient':
                    # if self.type == 'volScalarField':
                    #     self.updateZeroGradient(Region)
                    # if self.type == 'volVectorField':
                    self.updateZeroGradient(Region,iBPatch) 
                else:
                    io.cfdError('The '+iBPatch+' patch type is ill defined or missing!')

            elif thePhysicalPatchType == 'outlet':
                if theBCType == 'fixedValue' :
                    # if self.type == 'volScalarField':
                    #     self.updateFixedValue(Region)
                    # if self.type == 'volVectorField':
                    self.updateFixedValue(Region,iBPatch)
                elif theBCType == 'zeroGradient' or theBCType == 'outlet':
                    # if self.type == 'volScalarField':
                    #     self.updateZeroGradient(Region)
                    # if self.type == 'volVectorField':
                    self.updateZeroGradient(Region,iBPatch)
                else:
                    io.cfdError('The '+iBPatch+' patch type is ill defined or missing!')       
 
            elif thePhysicalPatchType == 'symmetry':
                if self.type == 'volScalarField':
                    self.updateZeroGradient(Region,iBPatch)
                if self.type == 'volVectorField':
                    self.updateSymmetry(Region,iBPatch)
            elif thePhysicalPatchType == 'empty':
                if self.type == 'volScalarField':
                    self.updateZeroGradient(Region,iBPatch)
                if self.type == 'volVectorField':
                    self.updateSymmetry(Region,iBPatch)
            else:
                io.cfdError('Physical condition bc not defined correctly for the '+iBPatch+' patch in "boundary" file !')



    def updateFixedValue(self,Region,iBPatch):
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        value = self.boundaryPatchRef[iBPatch]['value']
        self.phi[iBElements] = value

    def updateNoSlip(self,Region,iBPatch):
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        # value = self.boundaryPatchRef[iBPatch]['value']
        self.phi[iBElements].fill(0)

    def updateZeroGradient(self,Region,iBPatch):
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        #elements that own the boundary faces
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        self.phi[iBElements] = [self.phi[index] for index in owners_b]
        # for index in range(len(owners_b)):
        #     self.phi[iBElements[index]]=self.phi[owners_b[index]]
        # newValues=[]
        # for index in owners_b:
        #     newValues.append(self.phi[index])
        # for count, index in enumerate(iBElements):
        #     self.phi[index]=newValues[count]
            
    def updateSymmetry(self,Region,iBPatch):
        #get indices for self.iBPatch's boundary faces in self.phi array
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        #get indices for the owners (i.e. cells) for self.iBPatch's boundary faces in self.phi array 
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        #get vector (direction in which the face points) for self.iBpatch's boundary faces 
        # Sb = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
        #normalize Sb vector
        # normSb = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['normSb']
        #normalize Sb components and horizontally stack them into the columns of array n
        if self.iComponent==1:
            self.phi[iBElements]=self.phi[owners_b]
        else:
            n=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facen']
            # np.column_stack((self.Sb[:,0]/self.normSb,self.Sb[:,1]/self.normSb,self.Sb[:,2]/self.normSb))
            #perform elementwise multiplication of owner's values with boundary face normals
            U_normal_cfdMag=mth.cfdDot(self.phi[owners_b],n)
            #seems to do the same thing a the above line without the .sum(1)
            U_normal=np.column_stack((U_normal_cfdMag*n[:,0],U_normal_cfdMag*n[:,1],U_normal_cfdMag*n[:,2]))
            self.phi[iBElements]=self.phi[owners_b]-U_normal

    def cfdCorrectNSFields(self,Region,*args):
        # ==========================================================================
        #     Correct Fluid Velocity Field 更新速度场值
        # --------------------------------------------------------------------------
        if self.name=='U':
            theNumberOfElements = Region.coefficients.NumberOfElements
            # Get fields
            # DU0 = np.squeeze(Region.fluid['DU0'].phi)[0:theNumberOfElements]
            # DU1 = np.squeeze(Region.fluid['DU1'].phi)[0:theNumberOfElements]
            # DU2 = np.squeeze(Region.fluid['DU2'].phi)[0:theNumberOfElements]
            DU  =Region.fluid['DU'].phi[:theNumberOfElements,:]
            ppGrad = np.squeeze(Region.fluid['pprime'].phiGrad.phiGrad)[0:theNumberOfElements,:]
            #  Calculate Dc*gradP
            # DUPPGRAD = np.asarray([DU0*ppGrad[:,1],DU1*ppGrad[:,2],DU2*ppGrad[:,2]]).T
            DUPPGRAD = DU*ppGrad
            # Correct velocity
            self.phi[0:theNumberOfElements,:] -= DUPPGRAD
            for iComponent in range(3):
                self.cfdCorrectForBoundaryPatches(Region,iComponent)
            # self.cfdfieldUpdateGradient_Scale(Region)
            theNumberOfInteriorFaces=Region.mesh.numberOfInteriorFaces
            owners_f=Region.mesh.owners[:theNumberOfInteriorFaces]
            neighbours_f=Region.mesh.neighbours[:theNumberOfInteriorFaces]
            Region.fluid['mdot_f'].phi[:theNumberOfInteriorFaces] +=0.75*(Region.fluxes.FluxCf[:theNumberOfInteriorFaces][:,None]*Region.fluid['pprime'].phi[owners_f]
                                               +Region.fluxes.FluxFf[:theNumberOfInteriorFaces][:,None]*Region.fluid['pprime'].phi[neighbours_f])  # Update mdot_f
            Region.fluid['mdot_f'].cfdCorrectForBoundaryPatches(Region,-1)
        else:
            pass

