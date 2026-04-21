import numpy as np
import cfdtool.IO as io
import cfdtool.Math as mth
import cfdtool.Interpolate as interp
from cfdtool.backend import be
import pyFVM.cfdGetTools as tools
import cfdtool.dimensions as dm

class Assemble:
    def __init__(self,Region,theEquationName):
        """ Initiates the Assemble class instance
        """
        self.theEquationName=theEquationName
        ## The instance of Equation stored in the self.region.model dictionary
        self.theEquation=Region.model.equations[self.theEquationName]
        self.CoffeDim=self.theEquation.CoffeDim
        self.Dim=Region.fluid[self.theEquationName].dimension*self.CoffeDim

    def cfdAssembleEquation(self,Region,*args): 
        # 入口量纲一致性断言
        self._check_equation_dimensions(Region)

        if args:
            iComponent=args[0]
            self.cfdPreAssembleEquation(Region,iComponent)
            self.cfdAssembleEquationTerms(Region,iComponent)
            self.cfdPostAssembleEquation(Region,iComponent)
        else:            
            if self.theEquationName=='p':
                self.cfdPreAssembleContinuityEquation(Region)
                self.cfdAssembleContinuityEquationTerms(Region)
                self.cfdPostAssembleContinuityEquation(Region)
            else:
                self.cfdPreAssembleEquation(Region)
                self.cfdAssembleEquationTerms(Region)
                self.cfdPostAssembleEquation(Region)

    def cfdPreAssembleEquation(self,Region,*args):
        '''初始化方程的组装，系数置零'''
        Region.coefficients.cfdZeroCoefficients()

    def cfdAssembleEquationTerms(self,Region,*args): 
        """
        Assembles the equation's terms
        """
        if args:
            iComponent = args[0]
            self.iComponent=iComponent
        else:
            self.iComponent= int(-1)

        for iTerm in self.theEquation.terms:
            if iTerm == 'Transient':
                # print('Inside Transient Term')
                self.cfdZeroElementFLUXCoefficients(Region)
                self.cfdAssembleTransientTerm(Region)
                self.cfdAssembleIntoGlobalMatrixElementFluxes(Region) 

            elif iTerm == 'FalseTransient':
                # print('It is Steady State')
                pass

            elif iTerm == 'Convection':
                # print('Inside convection Term')
                self.cfdZeroFaceFLUXCoefficients(Region)
                self.cfdAssembleConvectionTerm(Region)
                self.cfdAssembleDCSchemeTerm(Region)
                self.cfdAssembleIntoGlobalMatrixFaceFluxes(Region)

                # self.cfdZeroElementFLUXCoefficients(Region)
                # self.cfdAssembleMomentumDivergenceCorrectionTerm(Region)# TODO Check
                # self.cfdAssembleIntoGlobalMatrixElementFluxes(Region)

            elif iTerm == 'Diffusion':
                # print('Inside Diffusion Stress Term')
                self.cfdZeroFaceFLUXCoefficients(Region)
                self.cfdAssembleDiffusionTerm(Region)
                self.cfdAssembleIntoGlobalMatrixFaceFluxes(Region)

            elif iTerm == 'Buoyancy':
                # print('Inside Buoyancy Term')
                self.cfdZeroElementFLUXCoefficients(Region)
                self.cfdAssembleMomentumBuoyancyTerm(Region)
                self.cfdAssembleIntoGlobalMatrixElementFluxes(Region)
                
            elif iTerm =='PressureGradient':
                # print('Inside PressureGradient Term')
                self.cfdZeroElementFLUXCoefficients(Region)
                self.cfdAssemblePressureGradientTerm(Region)
                self.cfdAssembleIntoGlobalMatrixElementFluxes(Region)

            else:
                io.cfdError(iTerm + ' term is not defined') 

    def cfdPostAssembleEquation(self,Region,*args):
        if args:
            iComponent=args[0]
            # Apply under-relaxation
            self.cfdAssembleImplicitRelaxation(Region,iComponent)

            # Store DU and DUT
            if 'p' in Region.model.equations:
                self.cfdAssembleDCoefficients(Region,iComponent)
            # Compute RMS and MAX Residuals
            Region.model.equations[self.theEquationName].cfdComputeScaledRMSResiduals(Region,iComponent)
        else:
            self.cfdAssembleImplicitRelaxation(Region)
            Region.model.equations[self.theEquationName].cfdComputeScaledRMSResiduals(Region)

    def cfdAssembleDCoefficients(self,Region,iComponent): 
        """
        Assembles the diffusion coefficients for the specified component in the given region.
        """
        theNumberOfElements=Region.mesh.numberOfElements
        # if strcmp(cfdGetAlgorithm,'SIMPLE')
        if Region.MatrixFormat == 'acnb':
            Diag=Region.coefficients.ac
        elif Region.MatrixFormat == 'ldu':
            Diag=Region.coefficients.Diag
        elif Region.MatrixFormat == 'csr':
            Diag=Region.coefficients.csrdata[Region.coefficients._indptr[:-1]]
        elif Region.MatrixFormat == 'coo':
            Diag=Region.coefficients.coodata[Region.coefficients._coodiagPositions]
        else:
            raise ValueError(f"Unsupported MatrixFormat: {Region.MatrixFormat}")

        Region.fluid['DU'].phi = be.set_at(Region.fluid['DU'].phi, (slice(None, theNumberOfElements), iComponent), Region.mesh.elementVolumes/Diag)
        # else
        # Region.fluid['DUT'].phi[:theNumberOfElements,iComponent]= Region.coefficients.ac_old/Region.coefficients.ac
        #Update at cfdBoundary patches
        Region.fluid['DU'].updateFieldForAllBoundaryPatches(Region)
        # Region.fluid['DUT'].updateFieldForAllBoundaryPatches(Region)

    def cfdAssembleIntoGlobalMatrixElementFluxes(self,Region,*args):
        """
        Add the face and volume contributions to obtain ac, bc and ac_old
        These are the ac and bc coefficients in the linear system of equations
        """
        #《The FVM in CFD》P. 545
        if Region.MatrixFormat == 'acnb':
            Region.coefficients.ac = Region.coefficients.ac + Region.fluxes.FluxC[self.theEquationName]
        elif Region.MatrixFormat == 'ldu':
            Region.coefficients.Diag = Region.coefficients.Diag + Region.fluxes.FluxC[self.theEquationName]
        elif Region.MatrixFormat == 'csr':
            diag_positions = Region.coefficients._indptr[:-1]
            Region.coefficients.csrdata = be.add_at(Region.coefficients.csrdata, diag_positions, Region.fluxes.FluxC[self.theEquationName])
        elif Region.MatrixFormat == 'coo':
            diag_positions = Region.coefficients._coodiagPositions
            Region.coefficients.coodata = be.add_at(Region.coefficients.coodata, diag_positions, Region.fluxes.FluxC[self.theEquationName])

        # Region.coefficients.ac_old  += Region.fluxes.FluxC_old[self.theEquationName].value
        Region.coefficients.bc = Region.coefficients.bc - Region.fluxes.FluxT[self.theEquationName]

    def cfdAssembleIntoGlobalMatrixFaceFluxes(self,Region,*args):
        #《The FVM in CFD》P. 545
        #   Assemble fluxes of interior faces
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        # numberOfFaces=Region.mesh.numberOfFaces
        # 获取所有内部面数据
        owners = Region.mesh.owners[:numberOfInteriorFaces]
        neighbours = Region.mesh.neighbours[:numberOfInteriorFaces]

        # 获取Flux数据
        FluxCf = Region.fluxes.FluxCf[self.theEquationName][:numberOfInteriorFaces]
        FluxFf = Region.fluxes.FluxFf[self.theEquationName][:numberOfInteriorFaces]
        FluxTf = Region.fluxes.FluxTf[self.theEquationName][:numberOfInteriorFaces]

        # Vectorized updates for interior faces
        # 通用的源项更新（所有格式都需要）
        Region.coefficients.bc = be.subtract_at(Region.coefficients.bc, owners, FluxTf)
        Region.coefficients.bc = be.add_at(Region.coefficients.bc, neighbours, FluxTf)

        # Assemble fluxes of boundary faces
        # 边界面处理（所有格式通用）
        boundary_owners = Region.mesh.owners[numberOfInteriorFaces:]
        FluxCf_b = Region.fluxes.FluxCf[self.theEquationName][numberOfInteriorFaces:]
        FluxTf_b = Region.fluxes.FluxTf[self.theEquationName][numberOfInteriorFaces:]
        # Vectorized updates for boundary faces
        Region.coefficients.bc = be.subtract_at(Region.coefficients.bc, boundary_owners, FluxTf_b)


        
        #根据矩阵格式处理矩阵元素
        if Region.MatrixFormat == 'acnb':
            #对角线元素更新
            Region.coefficients.ac = be.add_at(Region.coefficients.ac, owners, FluxCf)
            Region.coefficients.ac = be.subtract_at(Region.coefficients.ac, neighbours, FluxFf)
            Region.coefficients.ac = be.add_at(Region.coefficients.ac, boundary_owners, FluxCf_b)
            #非对角线元素组装 - 向量化版本：使用 flat anb 数组 + np.add.at
            # anb[i] 是 _acnb_flat 的视图，修改 flat 自动同步
            own_anb_index = Region.mesh.upperAnbCoeffIndex[:numberOfInteriorFaces]
            nei_anb_index = Region.mesh.lowerAnbCoeffIndex[:numberOfInteriorFaces]
            # 构建 flat 数组中的绝对偏移量
            upper_flat_idx = Region.coefficients._acnb_offsets[owners] + own_anb_index
            lower_flat_idx = Region.coefficients._acnb_offsets[neighbours] + nei_anb_index
            Region.coefficients._acnb_flat = be.add_at(Region.coefficients._acnb_flat, upper_flat_idx, FluxFf)
            Region.coefficients._acnb_flat = be.subtract_at(Region.coefficients._acnb_flat, lower_flat_idx, FluxCf)

        elif Region.MatrixFormat == 'ldu':
            """LDU格式的完全向量化组装"""
            # 对角线元素更新
            Region.coefficients.Diag = be.add_at(Region.coefficients.Diag, owners, FluxCf)
            Region.coefficients.Diag = be.subtract_at(Region.coefficients.Diag, neighbours, FluxFf)
            Region.coefficients.Diag = be.add_at(Region.coefficients.Diag, boundary_owners, FluxCf_b)
            #非对角线元素组装
            Region.coefficients.Upper = be.add_at(Region.coefficients.Upper, slice(None, numberOfInteriorFaces), FluxFf)
            Region.coefficients.Lower = be.subtract_at(Region.coefficients.Lower, slice(None, numberOfInteriorFaces), FluxCf)

        elif Region.MatrixFormat == 'csr':
            """CSR格式的向量化组装"""
            # 对角元素位置是每行的起始位置
            diag_positions = Region.coefficients._indptr[:-1]
            # 修复：正确的对角元素更新 - 使用正确的索引
            Region.coefficients.csrdata = be.add_at(Region.coefficients.csrdata, diag_positions[owners], FluxCf)
            Region.coefficients.csrdata = be.subtract_at(Region.coefficients.csrdata, diag_positions[neighbours], FluxFf)
            Region.coefficients.csrdata = be.add_at(Region.coefficients.csrdata, diag_positions[boundary_owners], FluxCf_b)

            #非对角线元素组装
            face_positions = Region.coefficients._csrfaceToRowIndex[:numberOfInteriorFaces]
            # 向量化更新CSR数据
            # neighbor_to_owner_pos = face_positions[:, 0]  # Lower值位置
            # owner_to_neighbor_pos = face_positions[:, 1]  # Upper值位置
            Region.coefficients.csrdata = be.add_at(Region.coefficients.csrdata, face_positions[:, 1], FluxFf)  # Upper
            Region.coefficients.csrdata = be.subtract_at(Region.coefficients.csrdata, face_positions[:, 0], FluxCf)  # Lower
            
            

        elif Region.MatrixFormat == 'coo':
            """COO格式的向量化组装"""
            # 对角线元素更新
            diag_positions = Region.coefficients._coodiagPositions
            # 修复：正确的对角元素更新 - 使用正确的索引
            Region.coefficients.coodata = be.add_at(Region.coefficients.coodata, diag_positions[owners], FluxCf)
            Region.coefficients.coodata = be.subtract_at(Region.coefficients.coodata, diag_positions[neighbours], FluxFf)
            Region.coefficients.coodata = be.add_at(Region.coefficients.coodata, diag_positions[boundary_owners], FluxCf_b)
            #非对角线元素组装
            face_positions = Region.coefficients._coofaceToRowIndex[:numberOfInteriorFaces]
            # 向量化更新COO数据
            # neighbor_to_owner_pos = face_positions[:, 0]  # Lower值位置
            # owner_to_neighbor_pos = face_positions[:, 1]  # Upper值位置
            Region.coefficients.coodata = be.add_at(Region.coefficients.coodata, face_positions[:, 1], FluxFf)  # Upper
            Region.coefficients.coodata = be.subtract_at(Region.coefficients.coodata, face_positions[:, 0], FluxCf)  # Lower



    def cfdAssembleImplicitRelaxation(self,Region,*args):
        """
        Add the face and volume contributions to obtain ac, bc and ac_old
        These are the ac and bc coefficients in the linear system of equations
        """
        try:
            urf = Region.dictionaries.fvSolution['relaxationFactors']['equations'][self.theEquationName]
            # if self.theEquationName in equations_dict:
            #     urf = equations_dict[self.theEquationName]
            # else:
            #     print(f"Key '{self.theEquationName}' not found. Using default value 0.7.")
            #     urf = 0.7  # 默认值
            
            
            # 关键修复：对于CSR格式，同步更新对角元素
            if Region.MatrixFormat == 'acnb':
                Region.coefficients.ac = Region.coefficients.ac / urf
            elif Region.MatrixFormat == 'ldu':
                Region.coefficients.Diag = Region.coefficients.Diag / urf
            elif Region.MatrixFormat == 'csr':
                diag_positions = Region.coefficients._indptr[:-1]
                Region.coefficients.csrdata = be.set_at(Region.coefficients.csrdata, diag_positions, Region.coefficients.csrdata[diag_positions] / urf)
            elif Region.MatrixFormat == 'coo':
                diag_positions = Region.coefficients._coodiagPositions
                Region.coefficients.coodata = be.set_at(Region.coefficients.coodata, diag_positions, Region.coefficients.coodata[diag_positions] / urf)

        except KeyError as e:
            io.cfdError(f"Key '{self.theEquationName}' not found in the dictionary.")

    def cfdZeroElementFLUXCoefficients(self,Region):
        # print('Inside cfdZeroElementFLUXCoefficients')
        Region.fluxes.cfdZeroElementFLUXCoefficients(self.theEquationName)


    def cfdZeroFaceFLUXCoefficients(self,Region):
        # print('Inside cfdZeroFaceFLUXCoefficients')
        Region.fluxes.cfdZeroFaceFLUXCoefficients(self.theEquationName)


    def cfdPreAssembleContinuityEquation(self,Region,*args):
        '''初始化连续性方程的组装，系数置零，并初始化压力修正方程'''
        self.cfdPreAssembleEquation(Region)
        #   Get DU field
        theNumberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        DUf = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear',tools.cfdGetSubArrayForInterior('DU',Region))
        Sf = Region.mesh.faceSf[0:theNumberOfInteriorFaces,:]
        #   Calculated info
        e = Region.mesh.faceCFn[0:theNumberOfInteriorFaces,:]
        DUSf=DUf*Sf
        # DUSf = np.column_stack((np.squeeze(DU0_f)*Sf[:,0],np.squeeze(DU1_f)*Sf[:,1],np.squeeze(DU2_f)*Sf[:,2]))
        Region.fluid['DUSf'].phi = be.set_at(Region.fluid['DUSf'].phi, slice(None, theNumberOfInteriorFaces), DUSf)
        magDUSf = mth.cfdMag(DUSf)
        if Region.mesh.OrthogonalCorrectionMethod in ['Minimum','minimum','corrected']:
            Region.fluid['DUEf'].phi = be.set_at(Region.fluid['DUEf'].phi, slice(None, theNumberOfInteriorFaces), mth.cfdDot(DUSf,e)[:,None]*e)
        elif Region.mesh.OrthogonalCorrectionMethod in ['Orthogonal','orthogonal']:
            Region.fluid['DUEf'].phi = be.set_at(Region.fluid['DUEf'].phi, slice(None, theNumberOfInteriorFaces), magDUSf[:,None]*e)
        elif Region.mesh.OrthogonalCorrectionMethod in ['OverRelaxed','overRelaxed']:
            eDUSf = mth.cfdUnit(DUSf)
            epsilon = 1e-10  # 定义一个小的正常数
            Region.fluid['DUEf'].phi = be.set_at(Region.fluid['DUEf'].phi, slice(None, theNumberOfInteriorFaces), (magDUSf/(mth.cfdDot(eDUSf,e)+ epsilon))[:,None]*e)
        else:
            io.cfdError('Region.mesh.OrthogonalCorrectionMethod not exist')

        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():    
            # Find the Physical Type
            # theBoundary = cfdGetBoundaryPatchRef(iBPatch)
            # thePhysicalType = theBCInfo['type']
            # theBCType = Region.fluid[self.theEquationName].boundaryPatchRef[iBPatch]['type']
            Sf_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
            iBFaces=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBFaces']
            CF_b = Region.mesh.faceCF[iBFaces]
            e = mth.cfdUnit(CF_b)
            DUb = tools.cfdGetSubArrayForBoundaryPatch('DU', iBPatch,Region)
            # DU1_b = tools.cfdGetSubArrayForBoundaryPatch('DU1', iBPatch,Region)
            # DU2_b = tools.cfdGetSubArrayForBoundaryPatch('DU2', iBPatch,Region)
            # DUSb = np.column_stack((DU0_b*Sf_b[:,1],DU1_b*Sf_b[:,2],DU2_b*Sf_b[:,2]))
            DUSb=DUb*Sf_b
            Region.fluid['DUSf'].phi = be.set_at(Region.fluid['DUSf'].phi, iBFaces, DUSb)
            magSUDb = mth.cfdMag(DUSb)
            if Region.mesh.OrthogonalCorrectionMethod in ['Minimum','minimum','corrected']:
                Region.fluid['DUEf'].phi = be.set_at(Region.fluid['DUEf'].phi, iBFaces, mth.cfdDot(DUSb,e)[:,None]*e)
            elif Region.mesh.OrthogonalCorrectionMethod in ['Orthogonal','orthogonal']:
                Region.fluid['DUEf'].phi = be.set_at(Region.fluid['DUEf'].phi, iBFaces, magSUDb[:,None]*e)
            elif Region.mesh.OrthogonalCorrectionMethod in ['OverRelaxed','overRelaxed']:
                epsilon = 1e-10  # 定义一个小的正常数
                eDUSb =  mth.cfdUnit(DUSb)
                Region.fluid['DUEf'].phi = be.set_at(Region.fluid['DUEf'].phi, iBFaces, (magSUDb/(mth.cfdDot(eDUSb,e )+ epsilon))[:,None]*e)
            else:
                io.cfdError('Region.mesh.OrthogonalCorrectionMethod not exist')
        # DUEb = [magSUDb./dot(e',eDUSb')'.*e(:,1),magSUDb./dot(e',eDUSb')'.*e(:,2),magSUDb./dot(e',eDUSb')'.*e(:,3)];

    def cfdAssembleContinuityEquationTerms(self,Region,*args): 
        """
        Assembles the equation's terms
        """
        if args:
            iComponent = args[0]
            self.iComponent=iComponent
        else:
            self.iComponent=-1

        for iTerm in self.theEquation.terms:
            if iTerm == 'Transient':
                if Region.cfdIsCompressible:
                    print('Inside Transient Term')
                    io.cfdError('Transient term not implemented for compressible flow')
                    # self.cfdZeroElementFLUXCoefficients(Region)
                    # self.cfdAssembleTransientTerm(Region)
                    # self.cfdAssembleIntoGlobalMatrixElementFluxes(Region)
            elif iTerm == 'FalseTransient':
                # print('It is Steady State')
                io.cfdError('FalseTransient term not implemented for compressible flow')

            elif iTerm == 'Convection':
                io.cfdError('Convection term not implemented for compressible flow')

            elif iTerm == 'massDivergenceTerm':
                # print('Inside massDivergence Term')
                self.cfdZeroFaceFLUXCoefficients(Region)
                self.cfdAssembleMassDivergenceTerm(Region)
                if Region.cfdIsCompressible:
                    io.cfdError('Compressible flow not implemented for massDivergenceTerm')
                    # self.cfdAssembleMassDivergenceAdvectionTerm()#not yet written
                # self.cfdStoreMassFlowRate(Region)#更新 mdot_f
                self.cfdAssembleIntoGlobalMatrixFaceFluxes(Region)

            else:
                io.cfdError(iTerm + ' term is not defined')


    def cfdPostAssembleContinuityEquation(self,Region,*args):
        # self.cfdAssembleDiagDominance(Region)
        # self.cfdFixPressure(Region)
        Region.model.equations[self.theEquationName].cfdComputeScaledRMSResiduals(Region)
        
    # def cfdFixPressure(self,Region):
    #     '''
    #     Fix Pressure
    #     '''
    #     # if self.theEquationName=='p':
    #     #     if Region.mesh.cfdIsClosedCavity:
    #     #         # Timesolver=Region.Timesolver
    #     #         # Get the pressure at the fixed value
    #     #         try:
    #     #             pRefCell = int(Region.dictionaries.fvSolution[Region.Timesolver]['pRefCell'])
    #     #             # theElementNbIndices = Region.mesh.elementNeighbours[pRefCell]
    #     #             # for iNBElement in range(len(Region.mesh.elementNeighbours[pRefCell])):
    #     #             #     Region.coefficients.anb[pRefCell][iNBElement] = 0
    #     #             # Region.coefficients.bc[pRefCell]= 0
    #     #             Region.coefficients.ac[pRefCell] += 1e3 
    #     #         except KeyError:
    #     #             io.cfdError('pRefCell not found')
    #     pass

    # def cfdAssembleDiagDominance(self,Region,*args):
    # # ==========================================================================
    # # Enforce Diagonal Dominance as this may not be ensured
    # # --------------------------------------------------------------------------
    # # Get info and fields
    #     # for iElement in range(Region.mesh.numberOfElements):
    #     #     theNumberOfNeighbours = len(Region.coefficients.theCConn[iElement])
    #     #     SumAik = 0
    #     #     #   adding all the off diagonal pressure terms
    #     #     for k in range(theNumberOfNeighbours):
    #     #         Region.coefficients.anb[iElement][k]=min(Region.coefficients.anb[iElement][k],-1e-10)
    #     #         SumAik -= Region.coefficients.anb[iElement][k]     
    #     #     Region.coefficients.ac[iElement] = max(Region.coefficients.ac[iElement],SumAik)
    #     pass

    def cfdAssembleMassDivergenceAdvectionTerm(self,Region):
        io.cfdError('cfdCompressible solver not yet written')

    # def cfdStoreMassFlowRate(self,Region):
    #     Region.fluid['mdot_f'].phi=Region.fluxes.FluxVf[self.theEquationName][:,None]
    
    def cfdAssembleMassDivergenceTerm(self,Region):
        """
        Routine Description: This function assembles pressure correction equation
        """
        #   Assemble at interior faces
        self.cfdAssembleMassDivergenceTermInterior(Region)
        #   Assemble at cfdBoundary patch faces
        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():    
            #   Find the Physical Type
            # theBoundary = cfdGetBoundaryPatchRef(iBPatch)
            thePhysicalType = theBCInfo['type']
            theBCType = Region.fluid[self.theEquationName].boundaryPatchRef[iBPatch]['type']
            if thePhysicalType=='wall':
                if theBCType=='noSlip':
                    self.cfdAssembleMassDivergenceTermWallNoslipBC(Region,iBPatch)
                elif theBCType=='slip':
                    self.cfdAssembleMassDivergenceTermWallSlipBC(Region,iBPatch)
                elif theBCType=='zeroGradient':
                    self.cfdAssembleMassDivergenceTermWallZeroGradientBC(Region,iBPatch)
                else:
                    io.cfdError(theBCType+'<<<< not implemented')
            elif thePhysicalType=='inlet':
                if theBCType=='inlet' or theBCType=='zeroGradient':
                    #Specified Velocity
                    self.cfdAssembleMassDivergenceTermInletZeroGradientBC(Region,iBPatch)
                elif theBCType=='fixedValue':
                    #Specified Pressure and Velocity Direction
                    self.cfdAssembleMassDivergenceTermInletFixedValueBC(Region,iBPatch)
                else:
                    #Specified Total Pressure and Velocity Direction
                    io.cfdError(theBCType+'<<<< not implemented')
            
            elif thePhysicalType=='outlet':
                if theBCType=='outlet' or theBCType=='zeroGradient':
                    #Specified Mass Flow Rate
                    self.cfdAssembleMassDivergenceTermOutletZeroGradientBC(Region,iBPatch)
                elif theBCType=='fixedValue':
                    #Specified Pressure
                    self.cfdAssembleMassDivergenceTermOutletFixedValueBC(Region,iBPatch)
                else:
                    io.cfdError(theBCType+'<<<< not implemented')
            elif thePhysicalType=='empty' or thePhysicalType=='symmetry' or thePhysicalType=='symmetryPlane':
                self.cfdAssembleMassDivergenceTermWallSlipBC(Region,iBPatch)
            else:
                io.cfdError(thePhysicalType+'<<<< not implemented')
    #  ===================================================
    #   WALL- slip Conditions
    #  ===================================================
    def cfdAssembleMassDivergenceTermWallSlipBC(self,Region,iBPatch):
        #P618页，只需压力边界施加zerogradient
        pass

    #  ===================================================
    #   WALL-noslip Condition
    #  ===================================================
    def cfdAssembleMassDivergenceTermWallNoslipBC(self,Region,iBPatch):
        #P618页，只需压力边界施加zerogradient
        pass

    #  ===================================================
    #   WALL-zeroGradient Condition
    #  ===================================================
    def cfdAssembleMassDivergenceTermWallZeroGradientBC(self,Region,iBPatch):
        #P618页，只需压力边界施加zerogradient
        pass

    #  ===================================================
    #   INLET - Zero Gradient
    #  ===================================================
    def  cfdAssembleMassDivergenceTermInletZeroGradientBC(self,Region,iBPatch):
        #   Get info
        # owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        Sf_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
        theNumberOfBFaces = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        startFaceIndex=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']
        iBFaces=list(range(startFaceIndex,startFaceIndex+theNumberOfBFaces))
        #   Initialize local fluxes
        #   Get Fields
        U_b = tools.cfdGetSubArrayForBoundaryPatch('U', iBPatch,Region)
        rho_b = tools.cfdGetSubArrayForBoundaryPatch('rho', iBPatch,Region)
        p_grad_b = tools.cfdGetGradientSubArrayForBoundaryPatch('p', iBPatch,Region)
        owners_b=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        p_grad_C=be.squeeze(Region.fluid['p'].Grad.phi[owners_b,:,0])

        #   ---------
        #    STEP 1
        #   ---------
        #    Assemble FLUXCb, FLUXFb and FLUXVb coefficients
        #  ---------------------------------------------------
        #   assemble RHIE-CHOW Interpolation Term I
        #  ---------------------------------------------------
        #   ONLY assemble term I
        #《the FVM in CFD》P. 618
        flux = mth.cfdDot(Sf_b,U_b)
        local_FluxVb = rho_b*flux
        local_FluxVb += rho_b*mth.cfdDot((p_grad_b-p_grad_C),Region.fluid['DUSf'].phi[iBFaces,:])
        #   Total flux
        #   local_FluxTb = local_FluxCb.*p(owners_b) + local_FluxFb.*p_b + local_FluxVb;
        #   Update global fluxes
        Region.fluxes.FluxVf[self.theEquationName] = be.add_at(Region.fluxes.FluxVf[self.theEquationName], iBFaces, local_FluxVb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxVb)  #TODO check FluxTf pprime boundarycondition


    #  ===================================================
    #   OUTLET - Zero Gradient
    #  ===================================================
    def cfdAssembleMassDivergenceTermOutletZeroGradientBC(self,Region,iBPatch):
        self.cfdAssembleMassDivergenceTermInletZeroGradientBC(Region,iBPatch)
        # #   Get info
        # # owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        # Sf_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
        # theNumberOfBFaces = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        # startFaceIndex=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']
        # iBFaces=list(range(startFaceIndex,startFaceIndex+theNumberOfBFaces))
        # #   Initialize local fluxes
        # local_FluxCb= np.zeros(theNumberOfBFaces)
        # local_FluxFb= np.zeros(theNumberOfBFaces)
        # # local_FluxVb= np.zeros(theNumberOfBFaces)
        # #   Get Fields
        # U_b = tools.cfdGetSubArrayForBoundaryPatch('U', iBPatch,Region)
        # rho_b = tools.cfdGetSubArrayForBoundaryPatch('rho', iBPatch,Region)
        # # p_b = tools.cfdGetSubArrayForBoundaryPatch('p', iBPatch,Region)
        # # p = tools.cfdGetSubArrayForInterior('p',Region)
        # #   ---------
        # #    STEP 1
        # #   ---------
        # #    Assemble FLUXCb, FLUXFb and FLUXVb coefficients
        # #  ---------------------------------------------------
        # #   assemble RHIE-CHOW Interpolation Term I
        # #  ---------------------------------------------------
        # #   ONLY assemble term I
        # U_b = (Sf_b*U_b).sum(1)
        # local_FluxVb = rho_b*U_b
        # #   Total flux
        # #   local_FluxTb = local_FluxCb.*p(owners_b) + local_FluxFb.*p_b + local_FluxVb;
        # #   Update global fluxes
        # Region.fluxes.FluxCf[iBFaces] =  local_FluxCb
        # Region.fluxes.FluxFf[iBFaces] =  local_FluxFb
        # Region.fluxes.FluxVf[iBFaces] =  local_FluxVb
        # #   theFluxes.FluxTf(iBFaces) =  local_FluxTb;

    #  ===================================================
    #   OUTLET - Fixed Value
    #  ===================================================
    def cfdAssembleMassDivergenceTermOutletFixedValueBC(self,Region,iBPatch):
        #   Get info
        Sf_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
        theNumberOfBFaces = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        iBFaces=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBFaces']
        # e = mth.cfdUnit(CF_b)
        #   Get Fields
        U_b = tools.cfdGetSubArrayForBoundaryPatch('U', iBPatch,Region)
        rho_b = tools.cfdGetSubArrayForBoundaryPatch('rho', iBPatch,Region)
        # p_b = tools.cfdGetSubArrayForBoundaryPatch('p', iBPatch,Region)
        # p = tools.cfdGetSubArrayForInterior('p',Region)
        p_grad_b = tools.cfdGetGradientSubArrayForBoundaryPatch('p', iBPatch,Region)
        owners_b=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']
        p_grad_C=be.squeeze(Region.fluid['p'].Grad.phi[owners_b,:,0])
        #   ---------
        #    STEP 1
        #   ---------
        #    Assemble FLUXCb, FLUXFb and FLUXVb coefficients
        #  ---------------------------------------------------------------
        #   assemble RHIE-CHOW Interpolation Term I, II, III, VIII and IX
        #  ---------------------------------------------------------------
        #《the FVM in CFD》P. 619 eq.15.161  
        #   Assemble Coefficients
        #   The DU field for cfdBoundary
        CF_b = Region.mesh.faceCF[iBFaces]
        geoDiff = mth.cfdMag(Region.fluid['DUEf'].phi[iBFaces,:])/mth.cfdMag(CF_b)
        #   Initialize local fluxes
        #   Assemble term I
        # U_bar_b = (U_b*Sf_b).sum(1)
        local_FluxVb = rho_b*mth.cfdDot(U_b,Sf_b)
        #   Assemble term II and linearize it
        local_FluxCb =  rho_b*geoDiff
        # local_FluxFb = -rho_b*geoDiff
        #   Assemble term III
        local_FluxVb += rho_b*mth.cfdDot((p_grad_b-p_grad_C),Region.fluid['DUSf'].phi[iBFaces,:])
        #   local_FluxTb = local_FluxCb.*p(owners_b) + local_FluxFb.*p_b + local_FluxVb;
        #   Update global fluxes
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], iBFaces, local_FluxCb)
        Region.fluxes.FluxVf[self.theEquationName] = be.add_at(Region.fluxes.FluxVf[self.theEquationName], iBFaces, local_FluxVb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxVb)   #TODO check FluxTf



    #  ===================================================
    #   INLET - Fixed Value
    #  ===================================================
    def cfdAssembleMassDivergenceTermInletFixedValueBC(self,Region,iBPatch):
        self.cfdAssembleMassDivergenceTermOutletFixedValueBC(Region,iBPatch)
        # #   Get info
        # Sf_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['facesSf']
        # theNumberOfBFaces = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        # iBFaces=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBFaces']
        # # e = mth.cfdUnit(CF_b)
        # #   Initialize local fluxes
        # # local_FluxCb =np.zeros(theNumberOfBFaces)
        # # local_FluxFb =np.zeros(theNumberOfBFaces)
        # local_FluxVb =np.zeros(theNumberOfBFaces)
        # #   Get Fields
        # U_b = tools.cfdGetSubArrayForBoundaryPatch('U', iBPatch,Region)
        # rho_b = tools.cfdGetSubArrayForBoundaryPatch('rho', iBPatch,Region)
        # # p_b = tools.cfdGetSubArrayForBoundaryPatch('p', iBPatch,Region)
        # # p = tools.cfdGetSubArrayForInterior('p',Region)
        # p_grad_b = tools.cfdGetGradientSubArrayForBoundaryPatch('p', iBPatch,Region)
        # #   ---------
        # #    STEP 1
        # #   ---------
        # #    Assemble FLUXCb, FLUXFb and FLUXVb coefficients
        # #  ---------------------------------------------------------------
        # #   assemble RHIE-CHOW Interpolation Term I, II, III, VIII and IX
        # #  ---------------------------------------------------------------
        # #   Assemble Coefficients
        # #   The DU field for cfdBoundary
        # CF_b = Region.mesh.faceCF[iBFaces]
        # geoDiff = mth.cfdMag(Region.fluid['DUEf'].phi[iBFaces,:])/mth.cfdMag(CF_b)
        # #   Assemble term I
        # # U_bar_b = dot(U_b',Sf_b')';
        # local_FluxVb += rho_b*(U_b*Sf_b).sum(1)
        # #   Assemble term II and linearize it
        # local_FluxCb = rho_b*geoDiff
        # local_FluxFb = np.zeros(theNumberOfBFaces)
        # #   Assemble term III
        # local_FluxVb +=  rho_b*(p_grad_b*Region.fluid['DUSf'].phi[iBFaces,:]).sum(1)
        # #   local_FluxTb = local_FluxCb.*p(owners_b) + local_FluxFb.*p_b + local_FluxVb;
        # #   Update global fluxes
        # Region.fluxes.FluxCf(iBFaces) = local_FluxCb
        # Region.fluxes.FluxFf(iBFaces) = local_FluxFb
        # #   theFluxes.FluxTf(iBFaces,1) =  local_FluxTb;


    def cfdAssembleMassDivergenceTermInterior(self,Region,*args):
        #   Get mesh info
        theNumberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        # iFaces = list(range(Region.mesh.numberOfInteriorFaces))
        # owners_f = Region.mesh.owners[:theNumberOfInteriorFaces]
        # neighbours_f = Region.mesh.neighbours[:theNumberOfInteriorFaces]
        Sf = Region.mesh.faceSf[0:theNumberOfInteriorFaces,:]
        cfdMagCF = Region.mesh.faceDist[0:theNumberOfInteriorFaces]
        geoDiff = mth.cfdMag(Region.fluid['DUEf'].phi[:theNumberOfInteriorFaces])/cfdMagCF
        p_RhieChowValue=tools.cfdRhieChowValue('p',Region)
        #   Get rho field and assign density at faces as the convected one
        rho = tools.cfdGetSubArrayForInterior('rho',Region)
        scheme='linearUpwind'
        if scheme=='linearUpwind':
            #   Get first computed mdot_f
            mdot_f_prev = tools.cfdGetSubArrayForInterior('mdot_f',Region)
            rho_f = be.squeeze(interp.cfdInterpolateFromElementsToInteriorFaces(Region,scheme, rho, mdot_f_prev))
        else:
            rho_f = be.squeeze(interp.cfdInterpolateFromElementsToInteriorFaces(Region,scheme, rho))

        #   Get velocity field and interpolate to faces
        # U_bar_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear', tools.cfdGetSubArrayForInterior('U',Region))

        #   Initialize local fluxes
        local_FluxCf = rho_f*geoDiff
        local_FluxFf = -local_FluxCf
        local_FluxVf = be.squeeze(tools.cfdGetSubArrayForInterior('mdot_f',Region))-rho_f*mth.cfdDot(p_RhieChowValue,Region.fluid['DUSf'].phi[:theNumberOfInteriorFaces])#TODO check FluxVf

        if Region.pp_nonlinear_corrected:
            Region.fluid['pprime'].Grad.cfdUpdateGradient(Region)
            DUTf = Region.fluid['DUSf'].phi - Region.fluid['DUEf'].phi
            pp_RhieChowValue=tools.cfdRhieChowValue('pprime',Region)
            local_FluxVf += rho_f*mth.cfdDot(pp_RhieChowValue,DUTf[:theNumberOfInteriorFaces])

        #    assemble term III
        #      rho_f ([P_grad]_f.([DPVOL]_f.Sf))
        # local_FluxVf += rho_f*(p_grad_bar_f*DUSf).sum(1)
        #   assemble terms IV
        #       (1-URF)(U_f -[v]_f.S_f)
        #   urf_U = cfdGetEquationRelaxationFactor('U');
        #   local_FluxVf = local_FluxVf + (1 - urf_U)*(mdot_f_prev - rho_f.*U_bar_f);
        #   Assemble total flux
        # local_FluxTf = local_FluxCf.*p(owners_f) + local_FluxFf.*p(neighbours_f) + local_FluxVf;

        #   Update global fluxes
        #《the FVM in CFD》P. 595 eq.15.99
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], slice(None, theNumberOfInteriorFaces), local_FluxCf)
        Region.fluxes.FluxFf[self.theEquationName] = be.add_at(Region.fluxes.FluxFf[self.theEquationName], slice(None, theNumberOfInteriorFaces), local_FluxFf)
        Region.fluxes.FluxVf[self.theEquationName] = be.add_at(Region.fluxes.FluxVf[self.theEquationName], slice(None, theNumberOfInteriorFaces), local_FluxVf)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], slice(None, theNumberOfInteriorFaces), local_FluxVf)
        #   theFluxes.FluxTf(iFaces,1) = local_FluxTf;      #TODO check FluxTf



    def cfdAssembleTransientTerm(self,Region,*args):
        """Chooses time-stepping approach
        If ddtSchemes is 'steadyState' then pass, if 'Euler' then redirect
        towards assembleFirstOrderEulerTransientTerm() or potentially others
        later on.
        Args:
            self (class instance): Instance of Region class.
            theEquationName (str): Equation (or field) name for which the transient terms will be assembled.

        Returns:
            none
        """
        # print('Inside cfdAssembleTransientTerm')
        theScheme = Region.dictionaries.fvSchemes['ddtSchemes']['default']
        if theScheme == 'Euler':
            self.assembleFirstOrderEulerTransientTerm(Region)
        else:
            io.cfdError(theScheme+' ddtScheme is incorrect')

    def assembleFirstOrderEulerTransientTerm(self,Region,*args):
        """Assembles first order transient euler term
        这段Python代码定义了一个名为`assembleFirstOrderEulerTransientTerm`的方法，这个方法是CFD模拟中数值求解过程的一部分，用于组装瞬态项，这对于求解流体动力学方程是必要的。通过这种方式，可以方便地访问和更新通量信息，以实现模拟的数值求解。
        """   
        volumes = Region.mesh.elementVolumes
        deltaT = Region.dictionaries.controlDict['deltaT']
        local_FluxC =volumes*be.squeeze(Region.fluid['rho'].phi[:Region.mesh.numberOfElements])/deltaT
        local_FluxC_old = -volumes*be.squeeze(Region.fluid['rho'].phi_old[:Region.mesh.numberOfElements])/deltaT
        Region.fluxes.FluxC[self.theEquationName] = Region.fluxes.FluxC[self.theEquationName] + local_FluxC#.applyFun(np.squeeze)
        Region.fluxes.FluxC_old[self.theEquationName] = Region.fluxes.FluxC_old[self.theEquationName] + local_FluxC_old

        # Region.fluxes.FluxV = np.zeros(len(local_FluxC),dtype=float)
        Region.fluxes.FluxT[self.theEquationName] = Region.fluxes.FluxT[self.theEquationName] +\
        (local_FluxC*Region.fluid[self.theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
                              +local_FluxC_old*Region.fluid[self.theEquationName].phi_old[:Region.mesh.numberOfElements,self.iComponent])

    def cfdPostAssembleScalarEquation(self, theEquationName):
        """Empty function, not sure why it exists
        """
        pass

    def cfdAssembleDCSchemeTerm(self,Region,*args):
        """Empty function Now, add later
           Second order upwind scheme
        """
        # Region.dictionaries.fvSchemes['divSchemes']['div(phi,U)']
        theEquationName=self.theEquationName
        theScheme = Region.dictionaries.fvSchemes['divSchemes']['div(phi,'+theEquationName+')']
        if theScheme=='Gauss upwind':
            return
        elif theScheme=='Gauss linear':
            #   Second order upwind scheme
            self.processDCSOUScheme(Region)
        else:
            io.cfdError(theScheme+' divScheme incorrect')


    def processDCSOUScheme(self,Region):
        #   Get mesh info
        theElementCentroids = Region.mesh.elementCentroids
        theFaceCentroids = Region.mesh.faceCentroids
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces

        #   Get fields
        theEquationName=self.theEquationName
        phi=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        gradPhi = Region.fluid[theEquationName].Grad.phi[:Region.mesh.numberOfElements,:,self.iComponent]
        mdot_f = Region.fluid['mdot_f'].phi[:Region.mesh.numberOfInteriorFaces]
        pos = (be.squeeze(mdot_f) > 0).astype(int)
        iFaces = be.arange(numberOfInteriorFaces)
        owners_f = Region.mesh.owners[iFaces]
        neighbours_f = Region.mesh.neighbours[iFaces]
        iUpwind = pos*owners_f + (1-pos)*neighbours_f
        #   Get the upwind gradient at the interior faces
        phiGradC = gradPhi[iUpwind,:]
        #   Interpolated gradient to interior faces
        phiGrad_f = interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,gradPhi,'Gauss linear corrected',phi)

        rC = theElementCentroids[iUpwind,:]
        rf = theFaceCentroids[iFaces,:]
        rCf = rf - rC
        # for i in range(numberOfInteriorFaces)
        #     dc_corr[i,:] = mdot_f[i] * np.dot(2*phiGradC[i,:]-phiGrad_f[i,:],rCf[i,:])
        dc_corr = be.squeeze(mdot_f) * mth.cfdDot(2*phiGradC-phiGrad_f,rCf)

        #Update global fluxes
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iFaces, dc_corr)

    def cfdAssembleConvectionTerm(self,Region,*args):
        self.cfdAssembleConvectionTermInterior(Region)
        # for iBPatch=1:theNumberOfBPatches
        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():    
            #   Find the Physical Type
            # theBoundary = cfdGetBoundaryPatchRef(iBPatch)
            thePhysicalType = theBCInfo['type']
            theBCType = Region.fluid['U'].boundaryPatchRef[iBPatch]['type']

            if thePhysicalType=='wall':
                continue
            elif thePhysicalType=='inlet':
                if theBCType=='fixedValue':
                    self.cfdAssembleConvectionTermSpecifiedValue(Region,iBPatch)
                    # continue
                elif theBCType=='zeroGradient':
                    self.cfdAssembleConvectionTermZeroGradient(Region,iBPatch)
                    # continue
                else:
                    io.cfdError([theBCType,'<<<< Not implemented'])
                
            elif thePhysicalType=='outlet':
                if theBCType=='fixedValue':
                    self.cfdAssembleConvectionTermSpecifiedValue(Region,iBPatch)
                    # continue      
                elif theBCType=='zeroGradient':
                    self.cfdAssembleConvectionTermZeroGradient(Region,iBPatch)
                    # continue
                else:
                    print([theBCType,'<<<< Not implemented'])
                    break
                
            elif thePhysicalType=='empty' or thePhysicalType=='symmetry' or thePhysicalType=='symmetryPlane':
                continue
            else:
                print([thePhysicalType, '<<<< Not implemented'])
                break

    def cfdAssembleConvectionTermSpecifiedValue(self,Region,iBPatch,*args):
        #  ===================================================
        #   Fixed Value
        #  ===================================================
        theEquationName=self.theEquationName
        iBFaces = np.array(range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']))
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields — 使用 .value 避免 Quantity 包装
        phi_val = Region.fluid[theEquationName].phi
        Ui_C = phi_val[owners_b,self.iComponent]
        Ui_b = phi_val[iBElements,self.iComponent]
        mdot_b = Region.fluid['mdot_f'].phi[iBFaces,0]

        #   linear fluxes
        zeros=np.zeros_like(mdot_b)
        local_FluxCb =  np.maximum(mdot_b,zeros)
        local_FluxFb = -np.maximum(-mdot_b,zeros)

        #   Update global fluxes
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], iBFaces, local_FluxCb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxCb*Ui_C + local_FluxFb*Ui_b)

    def cfdAssembleConvectionTermZeroGradient(self,Region,iBPatch,*args):
        #  ===================================================
        #   Zero Gradient
        #  ===================================================
        theEquationName=self.theEquationName
        iBFaces = np.array(range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']))
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields — 使用 .value 避免 Quantity 包装
        phi_val = Region.fluid[theEquationName].phi
        Ui_C = phi_val[owners_b,self.iComponent]
        mdot_b = Region.fluid['mdot_f'].phi[iBFaces,0]

        #   linear fluxes
        local_FluxCb = mdot_b
        #   Update global fluxes
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], iBFaces, local_FluxCb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxCb*Ui_C)


    def cfdAssembleConvectionTermInterior(self,Region,*args):
        theEquationName=self.theEquationName
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        # 使用 .value 避免 Quantity 包装/拆包开销
        phi=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        mdot_f=Region.fluid['mdot_f'].phi[:numberOfInteriorFaces,0]
        zeros=np.zeros_like(mdot_f)
        local_FluxCf=np.maximum(mdot_f,zeros)
        local_FluxFf=-np.maximum(-mdot_f,zeros)
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], slice(None, numberOfInteriorFaces), local_FluxCf)
        Region.fluxes.FluxFf[self.theEquationName] = be.add_at(Region.fluxes.FluxFf[self.theEquationName], slice(None, numberOfInteriorFaces), local_FluxFf)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], slice(None, numberOfInteriorFaces),
            local_FluxCf*phi[Region.mesh.interiorFaceOwners]
            +local_FluxFf*phi[Region.mesh.interiorFaceNeighbours])


    def cfdAssembleMomentumDivergenceCorrectionTerm(self,Region,*args):
        effDiv = self.cfdComputeEffectiveDivergence(Region,self.theEquationName)
        numberOfElements = Region.mesh.numberOfElements
        # 计算 FluxC, FluxV, FluxT
        Ui=Region.fluid[self.theEquationName].phi[:numberOfElements,self.iComponent]
        zeros=np.zeros_like(effDiv[:numberOfElements])
        max_effDiv = np.maximum(effDiv[:numberOfElements], zeros)
        local_FluxC = max_effDiv - effDiv[:numberOfElements]
        local_FluxV = -max_effDiv * Ui
        # local_FluxT = local_FluxC * Ui + local_FluxV
        Region.fluxes.FluxC[self.theEquationName] = be.add_at(Region.fluxes.FluxC[self.theEquationName], slice(None, numberOfElements), local_FluxC)
        Region.fluxes.FluxV[self.theEquationName] = be.add_at(Region.fluxes.FluxV[self.theEquationName], slice(None, numberOfElements), local_FluxV)
        Region.fluxes.FluxT[self.theEquationName] = be.add_at(Region.fluxes.FluxT[self.theEquationName], slice(None, numberOfElements), local_FluxC * Ui + local_FluxV)

    # TODO Check
    def cfdComputeEffectiveDivergence(self,Region,*args):
        # if args:
        #     theEquationName =args[0]
        # else:
        #     theEquationName ='U'
        owners = Region.mesh.owners
        neighbours = Region.mesh.neighbours
        theNumberofInteriorFaces = Region.mesh.numberOfInteriorFaces
        theNumberOfFaces = Region.mesh.numberOfFaces
        # theNumberOfElements = Region.mesh.numberOfElements
        #   Get the mdot_f field. Multiply by specific heat if the equation is the
        #   energy equation
        mdot_f = Region.fluid['mdot_f'].phi.copy()
        # if theEquationName=='T':
        #     try:
        #         Cp = Region.fluid['Cp'].phi
        #         Cp_f = interp.cfdInterpolateFromElementsToFaces(Region,'linear', Cp)
        #         mdot_f *=  Cp_f
        #     except:
        #         io.cfdError('Cp field is not available')
        # Initialize effective divergence array
        mdot_f = np.squeeze(mdot_f)
        effDiv = np.zeros(theNumberOfFaces)
        #   Interior Faces Contribution
        effDiv = be.add_at(effDiv, owners[:theNumberofInteriorFaces], mdot_f[:theNumberofInteriorFaces])
        effDiv = be.subtract_at(effDiv, neighbours[:theNumberofInteriorFaces], mdot_f[:theNumberofInteriorFaces])
        #   Boundary Faces Contribution
        effDiv = be.add_at(effDiv, owners[theNumberofInteriorFaces:], mdot_f[theNumberofInteriorFaces:])
        return effDiv

    def cfdAssembleDiffusionTerm(self,Region,*args):
        self.cfdAssembleDiffusionTermInterior(Region)
        # theNumberOfBPatches = cfdGetNumberOfBPatches;
        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():    
            #   Find the Physical Type
            # theBoundary = cfdGetBoundaryPatchRef(iBPatch)
            thePhysicalType = theBCInfo['type']
            theBCType = Region.fluid[self.theEquationName].boundaryPatchRef[iBPatch]['type']

            if thePhysicalType=='wall':
                if theBCType=='slip':
                    continue
                elif theBCType=='noSlip':
                    self.cfdAssembleStressTermWallNoslipBC(Region,iBPatch)
                elif theBCType=='fixedValue':
                    self.cfdAssembleStressTermWallFixedValueBC(Region,iBPatch)
                elif theBCType=='zeroGradient':
                    self.cfdAssembleStressTermZeroGradientBC(Region,iBPatch)
                else:
                    io.cfdError([theBCType,'<<<< Not implemented'])
                    # os.exit()

            elif thePhysicalType=='inlet':
                if theBCType=='fixedValue':
                    self.cfdAssembleStressTermSpecifiedValue(Region,iBPatch)
                    # continue
                elif theBCType=='zeroGradient' or theBCType=='inlet':
                    self.cfdAssembleStressTermZeroGradientBC(Region,iBPatch)
                    # continue
                else:
                    io.cfdError([theBCType,'<<<< Not implemented'])
                    # os.exit()
                
            elif thePhysicalType=='outlet':
                if theBCType=='fixedValue':
                    self.cfdAssembleStressTermSpecifiedValue(Region,iBPatch)
                    # continue      
                elif theBCType=='zeroGradient'or theBCType=='outlet':
                    self.cfdAssembleStressTermZeroGradientBC(Region,iBPatch)
                    # continue
                else:
                    io.cfdError([theBCType,'<<<< Not implemented'])
                    # break
                
            elif thePhysicalType=='symmetry' or thePhysicalType=='symmetryPlane':
                if self.iComponent != -1:
                    self.cfdAssembleStressTermSymmetry(Region,iBPatch)
                else:
                    pass
            elif thePhysicalType=='empty':
                self.cfdAssembleStressTermEmptyBC(Region,iBPatch)
            else:
                io.cfdError([thePhysicalType, '<<<< Not implemented'])
                # break
        # pass

    def cfdAssembleDiffusionTermInterior(self,Region,*args):
        """
        Assembles a diffusion term for the interior 
        """
        theEquationName=self.theEquationName
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        # 直接调用 _tool 版本避免 Quantity 包装/拆包开销
        gamma_f=np.squeeze(interp.cfdInterpolateFromElementsToFaces_tool(Region,'harmonic mean',Region.model.equations[self.theEquationName].gamma))#调和平均值，《FVM》P226
        local_FluxCf  = gamma_f[:numberOfInteriorFaces]*Region.mesh.geoDiff_f[:numberOfInteriorFaces]
        # local_FluxFf  =-gamma_f[0:numberOfInteriorFaces]*Region.mesh.geoDiff_f[0:numberOfInteriorFaces]
        local_FluxFf  =-local_FluxCf

        # 处理非正交修正项（如果需要）
        # Interpolated gradients on interior faces - 直接用 _tool 避免 Quantity 包装
        gradPhi_val = Region.fluid[theEquationName].Grad.phi[:Region.mesh.numberOfElements,:,self.iComponent]
        gradPhi_f=interp.cfdInterpolateGradientsFromElementsToInteriorFaces_tool(Region, gradPhi_val, 'linear')
        local_FluxVf  =-gamma_f[:numberOfInteriorFaces]*mth.cfdDot_np(gradPhi_f,Region.mesh.faceTf[:numberOfInteriorFaces,:])
        if theEquationName=='U':
            gradPhi_f=interp.cfdInterpolateGradientsFromElementsToInteriorFaces_tool(Region, Region.fluid[theEquationName].Grad.phi_TR[:Region.mesh.numberOfElements,:,self.iComponent],'linear')
            local_FluxVf += gamma_f[:numberOfInteriorFaces]*mth.cfdDot_np(gradPhi_f,Region.mesh.interiorFaceSf)
            if Region.cfdIsCompressible:
                local_FluxVf += 2.0/3.0*gamma_f[:numberOfInteriorFaces]*(Region.fluid[theEquationName].Grad.phi_Trace[:numberOfInteriorFaces]*Region.mesh.interiorFaceSf[:,self.iComponent])

        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], slice(None, numberOfInteriorFaces), local_FluxCf)
        Region.fluxes.FluxFf[self.theEquationName] = be.add_at(Region.fluxes.FluxFf[self.theEquationName], slice(None, numberOfInteriorFaces), local_FluxFf)
        Region.fluxes.FluxVf[self.theEquationName] = be.add_at(Region.fluxes.FluxVf[self.theEquationName], slice(None, numberOfInteriorFaces), local_FluxVf)

        phi_val=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], slice(None, numberOfInteriorFaces),
            local_FluxCf*np.squeeze(phi_val[Region.mesh.interiorFaceOwners])
            +local_FluxFf*np.squeeze(phi_val[Region.mesh.interiorFaceNeighbours])+ local_FluxVf)

        
    def cfdAssembleStressTermWallNoslipBC(self,Region,iBPatch,*args):
        #   Get info
        theEquationName=self.theEquationName
        iComponent=self.iComponent
        iBFaces = np.array(range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']))
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields — 使用 .value 避免 Quantity 包装开销
        phi_val = Region.fluid[theEquationName].phi
        phi_C = phi_val[owners_b,:]
        gamma_val = Region.model.equations[self.theEquationName].gamma
        mu_b = np.squeeze(gamma_val[owners_b])
        geoDiff_b = Region.mesh.geoDiff_f[iBFaces]

        u_C = phi_C[:,0]
        v_C = phi_C[:,1]
        w_C = phi_C[:,2]
        n = mth.cfdUnit(Region.mesh.faceSf[iBFaces])
        #   Normals and components
        nx = n[:,0]
        ny = n[:,1]
        nz = n[:,2]

        nx2 = nx*nx
        ny2 = ny*ny
        nz2 = nz*nz

        #   Local fluxes
        if iComponent==0:
            local_FluxCb =  mu_b*geoDiff_b*(1-nx2)
            local_FluxVb = -mu_b*geoDiff_b*(v_C*ny*nx+w_C*nz*nx)
        elif iComponent==1:
            local_FluxCb =  mu_b*geoDiff_b*(1-ny2)
            local_FluxVb = -mu_b*geoDiff_b*(u_C*nx*ny+w_C*nz*ny)
        elif iComponent==2:
            local_FluxCb =  mu_b*geoDiff_b*(1-nz2)
            local_FluxVb = -mu_b*geoDiff_b*(u_C*nx*nz+v_C*ny*nz)

        #   Update global fluxes
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], iBFaces, local_FluxCb)
        Region.fluxes.FluxVf[self.theEquationName] = be.add_at(Region.fluxes.FluxVf[self.theEquationName], iBFaces, local_FluxVb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxCb*phi_C[:,iComponent] + local_FluxVb)

    def cfdAssembleStressTermWallFixedValueBC(self,Region,iBPatch,*args):
        #   Get info
        theEquationName=self.theEquationName
        iComponent=self.iComponent
        iBFaces = np.array(range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']))
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields — 使用 .value 避免 Quantity 包装开销
        phi_val = Region.fluid[theEquationName].phi
        phi_C = phi_val[owners_b,:]
        phi_b = phi_val[iBElements,:] 
        gamma_val = Region.model.equations[self.theEquationName].gamma
        mu_b = np.squeeze(gamma_val[owners_b])
        geoDiff_b = Region.mesh.geoDiff_f[iBFaces]
        
        #《the FVM in CFD》P604-608 
        if iComponent != -1:
            u_C = phi_C[:,0]
            v_C = phi_C[:,1]
            w_C = phi_C[:,2]
            u_b = phi_b[:,0]
            v_b = phi_b[:,1]
            w_b = phi_b[:,2]
            n = mth.cfdUnit(Region.mesh.faceSf[iBFaces])
            #   Normals and components
            nx = n[:,0]
            ny = n[:,1]
            nz = n[:,2]
            nx2 = nx*nx
            ny2 = ny*ny
            nz2 = nz*nz
            #   Local fluxes
            if iComponent==0:
                local_FluxCb =  mu_b*geoDiff_b*(1-nx2)
                local_FluxVb = -mu_b*geoDiff_b*(u_b*(1-nx2)+(v_C-v_b)*ny*nx+(w_C-w_b)*nz*nx)
            elif iComponent==1:
                local_FluxCb =  mu_b*geoDiff_b*(1-ny2)
                local_FluxVb = -mu_b*geoDiff_b*((u_C-u_b)*nx*ny+v_b*(1-ny2)+(w_C-w_b)*nz*ny)
            elif iComponent==2:
                local_FluxCb =  mu_b*geoDiff_b*(1-nz2)
                local_FluxVb = -mu_b*geoDiff_b*((u_C-u_b)*nx*nz+(v_C-v_b)*ny*nz+w_b*(1-nz2))
        else:
            local_FluxCb=mu_b*geoDiff_b
            local_FluxVb=-local_FluxCb*phi_b[:,iComponent]

        # Update global fluxes
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], iBFaces, local_FluxCb)
        Region.fluxes.FluxVf[self.theEquationName] = be.add_at(Region.fluxes.FluxVf[self.theEquationName], iBFaces, local_FluxVb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxCb*phi_C[:,iComponent]  + local_FluxVb)

    def cfdAssembleStressTermSymmetry(self,Region,iBPatch,*args):
        #   Get info
        theEquationName=self.theEquationName
        iComponent=self.iComponent
        iBFaces = np.array(range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']))
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields — 使用 .value 避免 Quantity 包装开销
        phi_val = Region.fluid[theEquationName].phi
        phi_C = phi_val[owners_b,:]
        gamma_val = Region.model.equations[self.theEquationName].gamma
        mu_b = np.squeeze(gamma_val[owners_b])
        geoDiff_b = Region.mesh.geoDiff_f[iBFaces]

        u_C = phi_C[:,0]
        v_C = phi_C[:,1]
        w_C = phi_C[:,2]
        n = mth.cfdUnit(Region.mesh.faceSf[iBFaces])
        #   Normals and components
        nx = n[:,0]
        ny = n[:,1]
        nz = n[:,2]
        nx2 = nx*nx
        ny2 = ny*ny
        nz2 = nz*nz

        #   Local fluxes
        if iComponent==0:
            local_FluxCb =  2*mu_b*geoDiff_b*nx2
            local_FluxVb =  2*mu_b*geoDiff_b*(v_C*ny*nx+w_C*nz*nx)
        elif iComponent==1:
            local_FluxCb =  2*mu_b*geoDiff_b*ny2
            local_FluxVb =  2*mu_b*geoDiff_b*(u_C*nx*ny+w_C*nz*ny)
        elif iComponent==2:
            local_FluxCb =  2*mu_b*geoDiff_b*nz2
            local_FluxVb =  2*mu_b*geoDiff_b*(u_C*nx*nz+v_C*ny*nz)

        #   Update global fluxes
        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], iBFaces, local_FluxCb)
        Region.fluxes.FluxVf[self.theEquationName] = be.add_at(Region.fluxes.FluxVf[self.theEquationName], iBFaces, local_FluxVb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxCb*phi_C[:,iComponent] + local_FluxVb)


    def cfdAssembleStressTermSpecifiedValue(self,Region,iBPatch):
        #   Get info
        theEquationName=self.theEquationName
        iBFaces = np.array(range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']))
        iBElements = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get required fields — 使用 .value 避免 Quantity 包装开销
        gamma_val = Region.model.equations[self.theEquationName].gamma
        mu_b = np.squeeze(gamma_val[owners_b])
        geoDiff_b = Region.mesh.geoDiff_f[iBFaces]

        #   Local fluxes
        local_FluxCb =  mu_b*geoDiff_b
        local_FluxFb = -local_FluxCb

        #   Get fields
        phi_val = Region.fluid[theEquationName].phi
        phi_C = phi_val[owners_b,self.iComponent]
        phi_b = phi_val[iBElements,self.iComponent]

        Region.fluxes.FluxCf[self.theEquationName] = be.add_at(Region.fluxes.FluxCf[self.theEquationName], iBFaces, local_FluxCb)
        Region.fluxes.FluxTf[self.theEquationName] = be.add_at(Region.fluxes.FluxTf[self.theEquationName], iBFaces, local_FluxCb*phi_C+ local_FluxFb*phi_b)


    def cfdAssembleStressTermZeroGradientBC(self,Region,iBPatch):
        self.cfdAssembleStressTermSpecifiedValue(Region,iBPatch)

    def cfdAssembleStressTermEmptyBC(self,*args):
        pass
                
    def cfdAssembleMomentumBuoyancyTerm(self,Region,*args):
        #   Get info
        iComponent=self.iComponent
        volumes = Region.mesh.elementVolumes
        #   Get fields
        rho = np.squeeze(Region.fluid['rho'].phi[:Region.mesh.numberOfElements])
        #   Get gravity
        gi = Region.dictionaries.g['value'][iComponent]
        #   Update and store
        Region.fluxes.FluxT[self.theEquationName] = Region.fluxes.FluxT[self.theEquationName] + rho*volumes*gi
 
    def cfdAssemblePressureGradientTerm(self,Region,*args):
        #   Get info and fields
        volumes = Region.mesh.elementVolumes
        p_grad = np.squeeze(Region.fluid['p'].Grad.phi[:Region.mesh.numberOfElements,self.iComponent])
        #   Update global fluxes
        Region.fluxes.FluxT[self.theEquationName] = Region.fluxes.FluxT[self.theEquationName] + volumes*p_grad

    def _check_equation_dimensions(self, Region):
        """
        轻量级量纲一致性检查。
        在 cfdAssembleEquation 入口调用一次，验证方程的量纲设定是否一致。
        所有数据均为 ndarray，不依赖 Quantity 对象。
        
        检查内容：
        1. 场的量纲 × 方程系数量纲 == 通量量纲
        2. 通量系数 (CoffeDim) 与场量纲的关系
        """
        eq = self.theEquation
        field_dim = Region.fluid[self.theEquationName].dimension
        flux_dims = Region.fluxes._flux_dims.get(self.theEquationName, {})

        if not flux_dims:
            return  # 无通量量纲信息，跳过

        CoffeDim = flux_dims.get('CoffeDim')
        FluxDim = flux_dims.get('Dim')

        if CoffeDim is not None and FluxDim is not None:
            # 检验：场量纲 × 系数量纲 == 通量量纲
            expected_flux_dim = field_dim * CoffeDim
            if expected_flux_dim != FluxDim:
                raise ValueError(
                    f"[量纲检查] 方程 '{self.theEquationName}' 量纲不一致:\n"
                    f"  场量纲 ({self.theEquationName}): {field_dim}\n"
                    f"  系数量纲 (CoffeDim): {CoffeDim}\n"
                    f"  预期通量量纲 (field × CoffeDim): {expected_flux_dim}\n"
                    f"  实际通量量纲 (Dim): {FluxDim}"
                )
