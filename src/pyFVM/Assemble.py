import numpy as np
import cfdtool.IO as io
import cfdtool.Math as mth
import cfdtool.Interpolate as interp
import pyFVM.cfdGetTools as tools
from cfdtool.quantities import Quantity as Q_
import cfdtool.dimensions as dm

class Assemble:
    def __init__(self,Region,theEquationName):
        """ Initiates the Assemble class instance
        """
        self.theEquationName=theEquationName
        ## The instance of Equation stored in the self.region.model dictionary
        self.theEquation=Region.model.equations[self.theEquationName]
        self.CoffeDim=self.theEquation.CoffeDim
        self.Dim=Region.fluid[self.theEquationName].phi.dimension*self.CoffeDim

    def cfdAssembleEquation(self,Region,*args): 
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
                print('Inside Transient Term')
                self.cfdZeroElementFLUXCoefficients(Region)
                self.cfdAssembleTransientTerm(Region)
                self.cfdAssembleIntoGlobalMatrixElementFluxes(Region) 

            elif iTerm == 'FalseTransient':
                # print('It is Steady State')
                pass

            elif iTerm == 'Convection':
                print('Inside convection Term')
                self.cfdZeroFaceFLUXCoefficients(Region)
                self.cfdAssembleConvectionTerm(Region)
                self.cfdAssembleDCSchemeTerm(Region)
                self.cfdAssembleIntoGlobalMatrixFaceFluxes(Region)

                self.cfdZeroElementFLUXCoefficients(Region)
                self.cfdAssembleMomentumDivergenceCorrectionTerm(Region)# TODO Check
                self.cfdAssembleIntoGlobalMatrixElementFluxes(Region)

            elif iTerm == 'Diffusion':
                print('Inside Diffusion Stress Term')
                self.cfdZeroFaceFLUXCoefficients(Region)
                self.cfdAssembleDiffusionTerm(Region)
                self.cfdAssembleIntoGlobalMatrixFaceFluxes(Region)

            elif iTerm == 'Buoyancy':
                print('Inside Buoyancy Term')
                self.cfdZeroElementFLUXCoefficients(Region)
                self.cfdAssembleMomentumBuoyancyTerm(Region)
                self.cfdAssembleIntoGlobalMatrixElementFluxes(Region)
                
            elif iTerm =='PressureGradient':
                print('Inside PressureGradient Term')
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
        Region.fluid['DU'].phi[:theNumberOfElements,iComponent] = Region.mesh.elementVolumes/Q_(Region.coefficients.ac,Region.fluxes.FluxC[self.theEquationName].dimension)
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
        Region.coefficients.ac      += Region.fluxes.FluxC[self.theEquationName].value
        # Region.coefficients.ac_old  += Region.fluxes.FluxC_old[self.theEquationName].value
        Region.coefficients.bc      -= Region.fluxes.FluxT[self.theEquationName].value

    def cfdAssembleIntoGlobalMatrixFaceFluxes(self,Region,*args):
        #《The FVM in CFD》P. 545
        #   Assemble fluxes of interior faces
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        # numberOfFaces=Region.mesh.numberOfFaces
        # 获取所有内部面数据
        owners = Region.mesh.owners[:numberOfInteriorFaces]
        neighbours = Region.mesh.neighbours[:numberOfInteriorFaces]
        own_anb_index = Region.mesh.upperAnbCoeffIndex[:numberOfInteriorFaces]
        nei_anb_index = Region.mesh.lowerAnbCoeffIndex[:numberOfInteriorFaces]

        # 获取Flux数据
        FluxCf = Region.fluxes.FluxCf[self.theEquationName][:numberOfInteriorFaces].value
        FluxFf = Region.fluxes.FluxFf[self.theEquationName][:numberOfInteriorFaces].value
        FluxTf = Region.fluxes.FluxTf[self.theEquationName][:numberOfInteriorFaces].value

        # Vectorized updates for interior faces
        np.add.at(Region.coefficients.ac, owners, FluxCf)
        # np.add.at(Region.coefficients.anb, (owners, own_anb_index), FluxFf)
        for i in range(numberOfInteriorFaces):
            Region.coefficients.anb[owners[i]][own_anb_index[i]] += FluxFf[i]
        np.subtract.at(Region.coefficients.bc, owners, FluxTf)

        np.subtract.at(Region.coefficients.ac, neighbours, FluxFf)
        # np.subtract.at(Region.coefficients.anb, (neighbours, nei_anb_index), FluxCf)
        for i in range(numberOfInteriorFaces):
            Region.coefficients.anb[neighbours[i]][nei_anb_index[i]] -= FluxCf[i]
        np.add.at(Region.coefficients.bc, neighbours, FluxTf)


        #   Assemble fluxes of boundary faces
        boundary_owners = Region.mesh.owners[numberOfInteriorFaces:]
        FluxCf_b = Region.fluxes.FluxCf[self.theEquationName][numberOfInteriorFaces:].value
        FluxTf_b = Region.fluxes.FluxTf[self.theEquationName][numberOfInteriorFaces:].value
        # Vectorized updates for boundary faces
        np.add.at(Region.coefficients.ac, boundary_owners, FluxCf_b)
        np.subtract.at(Region.coefficients.bc, boundary_owners, FluxTf_b)

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
            Region.coefficients.ac /= urf
        except KeyError as e:
            print(f"KeyError: {e}")
            print(f"Key '{self.theEquationName}' not found in the dictionary.")

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
        # DU1_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear', tools.cfdGetSubArrayForInterior('DU1',Region))
        # DU2_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear', tools.cfdGetSubArrayForInterior('DU2',Region))    
        #   Assemble Coefficients
        #   assemble term I
        #       rho_f [v]_f.Sf
        # local_FluxVf += rho_f*(U_bar_f*Sf).sum(1)
        #   Assemble term II and linearize it
        #        - rho_f ([DPVOL]_f.P_grad_f).Sf
        Sf = Region.mesh.faceSf[0:theNumberOfInteriorFaces,:]
        #   Calculated info
        e = Region.mesh.faceCFn[0:theNumberOfInteriorFaces,:]
        DUSf=DUf*Sf
        # DUSf = np.column_stack((np.squeeze(DU0_f)*Sf[:,0],np.squeeze(DU1_f)*Sf[:,1],np.squeeze(DU2_f)*Sf[:,2]))
        Region.fluid['DUSf'].phi[0:theNumberOfInteriorFaces,:]=DUSf
        magDUSf = mth.cfdMag(DUSf)
        if Region.mesh.OrthogonalCorrectionMethod=='Minimum'or Region.mesh.OrthogonalCorrectionMethod=='minimum'or Region.mesh.OrthogonalCorrectionMethod=='corrected':
            Region.fluid['DUEf'].phi[0:theNumberOfInteriorFaces,:] = mth.cfdDot(DUSf,e)[:,None]*e
        elif Region.mesh.OrthogonalCorrectionMethod=='Orthogonal'or Region.mesh.OrthogonalCorrectionMethod=='orthogonal':
            Region.fluid['DUEf'].phi[0:theNumberOfInteriorFaces,:] =magDUSf[:,None]*e
        elif Region.mesh.OrthogonalCorrectionMethod=='OverRelaxed'or Region.mesh.OrthogonalCorrectionMethod=='overRelaxed':
            eDUSf = mth.cfdUnit(DUSf.value)
            epsilon = 1e-10  # 定义一个小的正常数
            Region.fluid['DUEf'].phi[0:theNumberOfInteriorFaces,:] =(magDUSf/(mth.cfdDot(eDUSf,e)+ epsilon))[:,None]*e
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
            e = mth.cfdUnit(CF_b.value)
            DUb = tools.cfdGetSubArrayForBoundaryPatch('DU', iBPatch,Region)
            # DU1_b = tools.cfdGetSubArrayForBoundaryPatch('DU1', iBPatch,Region)
            # DU2_b = tools.cfdGetSubArrayForBoundaryPatch('DU2', iBPatch,Region)
            # DUSb = np.column_stack((DU0_b*Sf_b[:,1],DU1_b*Sf_b[:,2],DU2_b*Sf_b[:,2]))
            DUSb=DUb*Sf_b
            Region.fluid['DUSf'].phi[iBFaces,:]=DUSb
            magSUDb = mth.cfdMag(DUSb)
            if Region.mesh.OrthogonalCorrectionMethod=='Minimum'or Region.mesh.OrthogonalCorrectionMethod=='minimum'or Region.mesh.OrthogonalCorrectionMethod=='corrected':
                Region.fluid['DUEf'].phi[iBFaces,:] = mth.cfdDot(DUSb,e)[:,None]*e
            elif Region.mesh.OrthogonalCorrectionMethod=='Orthogonal'or Region.mesh.OrthogonalCorrectionMethod=='orthogonal':
                Region.fluid['DUEf'].phi[iBFaces,:] =magSUDb[:,None]*e
            elif Region.mesh.OrthogonalCorrectionMethod=='OverRelaxed'or Region.mesh.OrthogonalCorrectionMethod=='overRelaxed':
                epsilon = 1e-10  # 定义一个小的正常数
                eDUSb =  mth.cfdUnit(DUSb.value)
                Region.fluid['DUEf'].phi[iBFaces,:] =(magSUDb/(mth.cfdDot(eDUSb,e )+ epsilon))[:,None]*e
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
                    # self.cfdZeroElementFLUXCoefficients(Region)
                    # self.cfdAssembleTransientTerm(Region)
                    # self.cfdAssembleIntoGlobalMatrixElementFluxes(Region)

            elif iTerm == 'FalseTransient':
                # print('It is Steady State')
                pass

            elif iTerm == 'massDivergenceTerm':
                print('Inside massDivergence Term')
                self.cfdZeroFaceFLUXCoefficients(Region)
                self.cfdAssembleMassDivergenceTerm(Region)
                if Region.cfdIsCompressible:
                    self.cfdAssembleMassDivergenceAdvectionTerm()#not yet written
                self.cfdStoreMassFlowRate(Region)#更新 mdot_f
                self.cfdAssembleIntoGlobalMatrixFaceFluxes(Region)

            else:
                io.cfdError(iTerm + ' term is not defined')

    def cfdPostAssembleContinuityEquation(self,Region,*args):
        self.cfdAssembleDiagDominance(Region)
        self.cfdFixPressure(Region)
        Region.model.equations[self.theEquationName].cfdComputeScaledRMSResiduals(Region)
        
    def cfdFixPressure(self,Region):
        '''
        Fix Pressure
        '''
        if self.theEquationName=='p':
            if Region.mesh.cfdIsClosedCavity:
                # Timesolver=Region.Timesolver
                # Get the pressure at the fixed value
                try:
                    pRefCell = int(Region.dictionaries.fvSolution[Region.Timesolver]['pRefCell'])
                    # theElementNbIndices = Region.mesh.elementNeighbours[pRefCell]
                    for iNBElement in range(len(Region.mesh.elementNeighbours[pRefCell])):
                        Region.coefficients.anb[pRefCell][iNBElement] = 0
                    Region.coefficients.bc[pRefCell]= 0
                except KeyError:
                    io.cfdError('pRefCell not found')

    def cfdAssembleDiagDominance(self,Region,*args):#TODO check not needed
    # ==========================================================================
    # Enforce Diagonal Dominance as this may not be ensured
    # --------------------------------------------------------------------------
    # Get info and fields
        # for iElement in range(Region.mesh.numberOfElements):
        #     theNumberOfNeighbours = len(Region.coefficients.theCConn[iElement])
        #     SumAik = 0
        #     #   adding all the off diagonal pressure terms
        #     for k in range(theNumberOfNeighbours):
        #         Region.coefficients.anb[iElement][k]=min(Region.coefficients.anb[iElement][k],-1e-10)
        #         SumAik -= Region.coefficients.anb[iElement][k]     
        #     Region.coefficients.ac[iElement] = max(Region.coefficients.ac[iElement],SumAik)
        pass

    def cfdAssembleMassDivergenceAdvectionTerm(self,Region):
        io.cfdError('cfdCompressible solver not yet written')

    def cfdStoreMassFlowRate(self,Region):
        Region.fluid['mdot_f'].phi=Region.fluxes.FluxVf[self.theEquationName][:,None]
    
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
        p_grad_C=np.squeeze(Region.fluid['p'].phiGrad.phiGrad[owners_b])
        # p_b = tools.cfdGetSubArrayForBoundaryPatch('p', iBPatch,Region)
        # p = tools.cfdGetSubArrayForInterior('p',Region)
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
        Region.fluxes.FluxVf[self.theEquationName][iBFaces] +=  local_FluxVb
        Region.fluxes.FluxTf[self.theEquationName][iBFaces] +=  local_FluxVb  #TODO check FluxTf pprime boundarycondition


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
        p_grad_C=np.squeeze(Region.fluid['p'].phiGrad.phiGrad[owners_b])
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
        Region.fluxes.FluxCf[self.theEquationName][iBFaces] += local_FluxCb
        Region.fluxes.FluxVf[self.theEquationName][iBFaces] += local_FluxVb
        Region.fluxes.FluxTf[self.theEquationName][iBFaces] += local_FluxVb   #TODO check FluxTf



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
        geoDiff = mth.cfdMag(Region.fluid['DUEf'].phi[0:theNumberOfInteriorFaces])/cfdMagCF
        p_RhieChowValue=tools.cfdRhieChowValue('p',Region)
        #   Get rho field and assign density at faces as the convected one
        rho = tools.cfdGetSubArrayForInterior('rho',Region)
        scheme='linearUpwind'
        if scheme=='linearUpwind':
            #   Get first computed mdot_f
            mdot_f_prev = tools.cfdGetSubArrayForInterior('mdot_f',Region)
            rho_f = np.squeeze(interp.cfdInterpolateFromElementsToInteriorFaces(Region,scheme, rho, mdot_f_prev))
        else:
            rho_f = np.squeeze(interp.cfdInterpolateFromElementsToInteriorFaces(Region,scheme, rho))

        #   Get velocity field and interpolate to faces
        U_bar_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear', tools.cfdGetSubArrayForInterior('U',Region))

        #   Initialize local fluxes
        local_FluxCf = rho_f*geoDiff
        local_FluxFf = -local_FluxCf
        local_FluxVf = rho_f*(mth.cfdDot(U_bar_f,Sf)-mth.cfdDot(p_RhieChowValue,Region.fluid['DUSf'].phi[0:theNumberOfInteriorFaces]))#TODO check FluxVf

        if Region.pp_nonlinear_corrected:
            Region.fluid['pprime'].phiGrad.cfdUpdateGradient(Region)
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
        Region.fluxes.FluxCf[self.theEquationName][:theNumberOfInteriorFaces]  += local_FluxCf
        Region.fluxes.FluxFf[self.theEquationName][:theNumberOfInteriorFaces]  += local_FluxFf
        Region.fluxes.FluxVf[self.theEquationName][:theNumberOfInteriorFaces]  += local_FluxVf
        Region.fluxes.FluxTf[self.theEquationName][:theNumberOfInteriorFaces]  += local_FluxVf
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
        print('Inside cfdAssembleTransientTerm')
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
        deltaT = Q_(Region.dictionaries.controlDict['deltaT'],dm.time_dim)
        local_FluxC =volumes*np.squeeze(Region.fluid['rho'].phi[:Region.mesh.numberOfElements])/deltaT
        local_FluxC_old = -volumes*np.squeeze(Region.fluid['rho'].phi_old[:Region.mesh.numberOfElements])/deltaT
        Region.fluxes.FluxC[self.theEquationName] += local_FluxC#.applyFun(np.squeeze)
        Region.fluxes.FluxC_old[self.theEquationName] += local_FluxC_old

        # Region.fluxes.FluxV = np.zeros(len(local_FluxC),dtype=float)
        Region.fluxes.FluxT[self.theEquationName]+=\
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
        pos = (np.squeeze(mdot_f.value) > 0).astype(int) 
        iFaces = list(range(numberOfInteriorFaces))
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
        dc_corr = np.squeeze(mdot_f) * mth.cfdDot(2*phiGradC-phiGrad_f,rCf)

        #Update global fluxes
        Region.fluxes.FluxTf[self.theEquationName][iFaces] +=  dc_corr

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
        #   Get info
        theEquationName=self.theEquationName
        iBFaces = range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        # Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']-Region.mesh.numberOfElements
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields
        Ui_C =Region.fluid[theEquationName].phi[owners_b,self.iComponent]
        Ui_b =Region.fluid[theEquationName].phi[iBElements,self.iComponent]
        mdot_b = np.squeeze(Region.fluid['mdot_f'].phi[iBFaces])

        #   linear fluxes
        zeros=np.zeros_like(mdot_b)
        local_FluxCb =  np.maximum(mdot_b,zeros)
        local_FluxFb = -np.maximum(-mdot_b,zeros)# 《The SIMPLE Algorithm》 Page57 eq6.82

        #   Non-linear fluxes
        # local_FluxVb = np.zeros(len(local_FluxCb))
        #   Update global fluxes
        Region.fluxes.FluxCf[iBFaces]  += local_FluxCb
        # Region.fluxes.FluxFf[iBFaces]  = local_FluxFb
        Region.fluxes.FluxTf[iBFaces]  += local_FluxCb*Ui_C + local_FluxFb*Ui_b #+ local_FluxVb

    def cfdAssembleConvectionTermZeroGradient(self,Region,iBPatch,*args):
        #  ===================================================
        #   Fixed Value
        #  ===================================================
        #   Get info
        theEquationName=self.theEquationName
        iBFaces = range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])
        # iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields
        # Region.fluid[theEquationName].cfdGetSubArrayForInterior(Region)
        Ui_C =Region.fluid[theEquationName].phi[owners_b,self.iComponent]
        mdot_b = np.squeeze(Region.fluid['mdot_f'].phi[iBFaces])

        #   linear fluxes
        local_FluxCb = mdot_b
        #   Non-linear fluxes
        #   Update global fluxes
        Region.fluxes.FluxCf[iBFaces]  += local_FluxCb
        # Region.fluxes.FluxFf[iBFaces]  = local_FluxFb
        Region.fluxes.FluxTf[iBFaces]  += local_FluxCb*Ui_C
        # + local_FluxFb*Ui_b[iBElements] + local_FluxVb


    def cfdAssembleConvectionTermInterior(self,Region,*args):
        theEquationName=self.theEquationName
        # Region.fluid[theEquationName].cfdGetSubArrayForInterior(Region)
        phi=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        mdot_f=np.squeeze(Region.fluid['mdot_f'].phi[0:numberOfInteriorFaces])
        zeros=np.zeros_like(mdot_f)
        local_FluxCf=np.maximum(mdot_f,zeros)  #《The SIMPLE Algorithm》Chapter 5. Page39 公式5.66
        local_FluxFf=-np.maximum(-mdot_f,zeros)
        # local_FluxVf=np.zeros_like(Region.fluxes.FluxVf[self.theEquationName][:numberOfInteriorFaces])
        Region.fluxes.FluxCf[self.theEquationName][:numberOfInteriorFaces] += local_FluxCf
        Region.fluxes.FluxFf[self.theEquationName][:numberOfInteriorFaces] += local_FluxFf
        # Region.fluxes.FluxVf[self.theEquationName][:numberOfInteriorFaces] += local_FluxVf
        # Region.fluxes.FluxTf[0:numberOfInteriorFaces] = np.multiply(local_FluxCf[:, np.newaxis],np.squeeze(phi[self.owners_interior_f]))+np.multiply(local_FluxFf[:, np.newaxis],np.squeeze(phi[self.neighbours_interior_f]))+ local_FluxVf[:, np.newaxis]
        # for field in Region.fluxes.FluxTf:
        # if args:
        #     iComponent=args[0]
        #     Region.fluxes.FluxTf[0:numberOfInteriorFaces] = Region.fluxes.FluxCf[0:numberOfInteriorFaces]*phi[Region.mesh.interiorFaceOwners,iComponent]+Region.fluxes.FluxFf[0:numberOfInteriorFaces]*phi[Region.mesh.interiorFaceNeighbours,iComponent]+ Region.fluxes.FluxVf[0:numberOfInteriorFaces]
        # else:
        Region.fluxes.FluxTf[self.theEquationName][:numberOfInteriorFaces] +=local_FluxCf*np.squeeze(phi[Region.mesh.interiorFaceOwners])\
            +local_FluxFf*np.squeeze(phi[Region.mesh.interiorFaceNeighbours])#+ local_FluxVf


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
        Region.fluxes.FluxC[self.theEquationName][:numberOfElements] += local_FluxC
        Region.fluxes.FluxV[self.theEquationName][:numberOfElements] += local_FluxV
        Region.fluxes.FluxT[self.theEquationName][:numberOfElements] += local_FluxC * Ui + local_FluxV

    # TODO Check
    def cfdComputeEffectiveDivergence(self,Region,*args):
        if args:
            theEquationName =args[0]
        else:
            theEquationName ='U'

        owners = Region.mesh.owners
        neighbours = Region.mesh.neighbours
        theNumberofInteriorFaces = Region.mesh.numberOfInteriorFaces
        theNumberOfFaces = Region.mesh.numberOfFaces
        # theNumberOfElements = Region.mesh.numberOfElements
        #   Get the mdot_f field. Multiply by specific heat if the equation is the
        #   energy equation
        mdot_f = Region.fluid['mdot_f'].phi.copy()
        if theEquationName=='T':
            try:
                Cp = Region.fluid['Cp'].phi
                Cp_f = interp.cfdInterpolateFromElementsToFaces(Region,'linear', Cp)
                mdot_f *=  Cp_f
            except:
                print('Cp field is not available')
        # Initialize effective divergence array
        mdot_f = np.squeeze(mdot_f)
        effDiv = Q_(np.zeros(theNumberOfFaces),mdot_f.dimension)
        #   Interior Faces Contribution
        effDiv[owners[:theNumberofInteriorFaces]] += mdot_f[:theNumberofInteriorFaces]
        effDiv[neighbours[:theNumberofInteriorFaces]] -= mdot_f[:theNumberofInteriorFaces]
        #   Boundary Faces Contribution
        effDiv[owners[theNumberofInteriorFaces:]] += mdot_f[theNumberofInteriorFaces:]
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
        gamma_f=np.squeeze(interp.cfdInterpolateFromElementsToFaces(Region,'harmonic mean',Region.model.equations[self.theEquationName].gamma))#调和平均值，《FVM》P226
        local_FluxCf  = gamma_f[:numberOfInteriorFaces]*Region.mesh.geoDiff_f[:numberOfInteriorFaces]
        # local_FluxFf  =-gamma_f[0:numberOfInteriorFaces]*Region.mesh.geoDiff_f[0:numberOfInteriorFaces]
        local_FluxFf  =-local_FluxCf

        # 处理非正交修正项（如果需要）
        # Interpolated gradients on interior faces
        gradPhi_f=interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,Region.fluid[theEquationName].Grad.phi[:Region.mesh.numberOfElements,:,self.iComponent],'linear')
        local_FluxVf  =-gamma_f[:numberOfInteriorFaces]*mth.cfdDot(gradPhi_f,Region.mesh.faceTf[:numberOfInteriorFaces,:])
        if theEquationName=='U':
            gradPhi_f=interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,Region.fluid[theEquationName].Grad.phi_TR[:Region.mesh.numberOfElements,:,self.iComponent],'linear')
            local_FluxVf += gamma_f[:numberOfInteriorFaces]*mth.cfdDot(gradPhi_f,Region.mesh.interiorFaceSf)
            if Region.cfdIsCompressible:
                local_FluxVf += 2.0/3.0*gamma_f[:numberOfInteriorFaces]*(Region.fluid[theEquationName].Grad.phi_Trace[:numberOfInteriorFaces]*Region.mesh.interiorFaceSf[:,self.iComponent])

        Region.fluxes.FluxCf[self.theEquationName][:numberOfInteriorFaces] += local_FluxCf
        Region.fluxes.FluxFf[self.theEquationName][:numberOfInteriorFaces] += local_FluxFf
        Region.fluxes.FluxVf[self.theEquationName][:numberOfInteriorFaces] += local_FluxVf
        
        phi=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        Region.fluxes.FluxTf[self.theEquationName][:numberOfInteriorFaces] += local_FluxCf*np.squeeze(phi[Region.mesh.interiorFaceOwners])\
            +local_FluxFf*np.squeeze(phi[Region.mesh.interiorFaceNeighbours])+ local_FluxVf

        
    def cfdAssembleStressTermWallNoslipBC(self,Region,iBPatch,*args):
        #   Get info
        theEquationName=self.theEquationName
        iComponent=self.iComponent
        iBFaces = range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])
        # iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields
        # Region.fluid[theEquationName].cfdGetSubArrayForInterior(Region)
        phi_C =Region.fluid[theEquationName].phi[owners_b,:]
        #   Get required fields
        mu_b = np.squeeze(Region.model.equations[self.theEquationName].gamma[owners_b])


        u_C = phi_C[:,0]
        v_C = phi_C[:,1]
        w_C = phi_C[:,2]
        n = mth.cfdUnit(Region.mesh.faceSf[iBFaces].value)
        #   Normals and components
        nx = n[:,0]
        ny = n[:,1]
        nz = n[:,2]

        nx2 = nx*nx
        ny2 = ny*ny
        nz2 = nz*nz

        #   Local fluxes
        if iComponent==0:
            local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-nx2)
            local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*(v_C*ny*nx+w_C*nz*nx)
        elif iComponent==1:
            local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-ny2)
            local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*(u_C*nx*ny+w_C*nz*ny)
        elif iComponent==2:
            local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-nz2)
            local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*(u_C*nx*nz+v_C*ny*nz)

        #   Update global fluxes
        # local_FluxFb =np.zeros(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])

        Region.fluxes.FluxCf[self.theEquationName][iBFaces] += local_FluxCb
        # Region.fluxes.FluxFf[iBFaces] = local_FluxFb
        Region.fluxes.FluxVf[self.theEquationName][iBFaces] += local_FluxVb
        Region.fluxes.FluxTf[self.theEquationName][iBFaces] += local_FluxCb*phi_C[:,iComponent] + local_FluxVb#+ local_FluxFb*phi_b[:,iComponent]

    def cfdAssembleStressTermWallFixedValueBC(self,Region,iBPatch,*args):
        #   Get info
        theEquationName=self.theEquationName
        iComponent=self.iComponent
        iBFaces = range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])
        iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields
        # Region.fluid[theEquationName].cfdGetSubArrayForInterior(Region)
        phi_C =Region.fluid[theEquationName].phi[owners_b,:]
        phi_b =Region.fluid[theEquationName].phi[iBElements,:] 
        #   Get required fields
        mu_b = np.squeeze(Region.model.equations[self.theEquationName].gamma[owners_b])
        
        #《the FVM in CFD》P604-608 
        if iComponent != -1:
            u_C = phi_C[:,0]
            v_C = phi_C[:,1]
            w_C = phi_C[:,2]
            u_b = phi_b[:,0]
            v_b = phi_b[:,1]
            w_b = phi_b[:,2]
            n = mth.cfdUnit(Region.mesh.faceSf[iBFaces].value)
            #   Normals and components
            nx = n[:,0]
            ny = n[:,1]
            nz = n[:,2]
            nx2 = nx*nx
            ny2 = ny*ny
            nz2 = nz*nz
            #   Local fluxes
            if iComponent==0:
                local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-nx2)
                local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*(u_b*(1-nx2)+(v_C-v_b)*ny*nx+(w_C-w_b)*nz*nx)
            elif iComponent==1:
                local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-ny2)
                local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*((u_C-u_b)*nx*ny+v_b*(1-ny2)+(w_C-w_b)*nz*ny)
            elif iComponent==2:
                local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-nz2)
                local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*((u_C-u_b)*nx*nz+(v_C-v_b)*ny*nz+w_b*(1-nz2))
            # local_FluxFb =np.zeros_like(local_FluxCb)
        else:
            local_FluxCb=mu_b*Region.mesh.geoDiff_f[iBFaces]
            # local_FluxFb=-mu_b*Region.mesh.geoDiff_f[iBFaces] #TODO check boundary condition
            local_FluxVb=-local_FluxCb*phi_b[:,iComponent]


        # Update global fluxes
        Region.fluxes.FluxCf[self.theEquationName][iBFaces] += local_FluxCb
        Region.fluxes.FluxVf[self.theEquationName][iBFaces] += local_FluxVb
        Region.fluxes.FluxTf[self.theEquationName][iBFaces] += local_FluxCb*phi_C[:,iComponent]  + local_FluxVb
        # pass

    def cfdAssembleStressTermSymmetry(self,Region,iBPatch,*args):
        #   Get info
        theEquationName=self.theEquationName
        iComponent=self.iComponent
        iBFaces = range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])
        # iBElements=Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        #   Get fields
        # Region.fluid[theEquationName].cfdGetSubArrayForInterior(Region)
        phi_C =Region.fluid[theEquationName].phi[owners_b,:]
        #   Get required fields
        mu_b = np.squeeze(Region.model.equations[self.theEquationName].gamma[owners_b])

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
            local_FluxCb =  2*mu_b*Region.mesh.geoDiff_f[iBFaces]*nx2
            local_FluxVb =  2*mu_b*Region.mesh.geoDiff_f[iBFaces]*(v_C*ny*nx+w_C*nz*nx)
        elif iComponent==1:
            local_FluxCb =  2*mu_b*Region.mesh.geoDiff_f[iBFaces]*ny2
            local_FluxVb =  2*mu_b*Region.mesh.geoDiff_f[iBFaces]*(u_C*nx*ny+w_C*nz*ny)
        elif iComponent==2:
            local_FluxCb =  2*mu_b*Region.mesh.geoDiff_f[iBFaces]*nz2
            local_FluxVb =  2*mu_b*Region.mesh.geoDiff_f[iBFaces]*(u_C*nx*nz+v_C*ny*nz)

        #   Update global fluxes
        # local_FluxFb =np.zeros(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])

        Region.fluxes.FluxCf[self.theEquationName][iBFaces] += local_FluxCb
        # Region.fluxes.FluxFf[iBFaces] = local_FluxFb
        Region.fluxes.FluxVf[self.theEquationName][iBFaces] += local_FluxVb
        Region.fluxes.FluxTf[self.theEquationName][iBFaces] += local_FluxCb*phi_C[:,iComponent] + local_FluxVb #+ local_FluxFb*U_b[:,iComponent]


    def cfdAssembleStressTermSpecifiedValue(self,Region,iBPatch):
        #   Get info
        theEquationName=self.theEquationName
        # iComponent=self.iComponent
        iBFaces = range(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'],Region.mesh.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']+Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])
        iBElements = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']
        # iBElements=[index - Region.mesh.numberOfElements for index in Region.mesh.cfdBoundaryPatchesArray[iBPatch]['iBElements']]
        owners_b = Region.mesh.cfdBoundaryPatchesArray[iBPatch]['owners_b']

        # magSb = Region.mesh.faceAreas[iBFaces]
        # wallDist_b = Region.mesh.wallDist[iBFaces]
        #   Get required fields
        mu_b = np.squeeze(Region.model.equations[self.theEquationName].gamma[owners_b])

        #   Local fluxes
        local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]
        local_FluxFb = -local_FluxCb
        #   Update global fluxes
        # local_FluxVb = np.zeros(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])

        #   Get fields
        phi_C =Region.fluid[theEquationName].phi[owners_b,self.iComponent]
        phi_b =Region.fluid[theEquationName].phi[iBElements,self.iComponent]

        Region.fluxes.FluxCf[self.theEquationName][iBFaces] += local_FluxCb
        # Region.fluxes.FluxFf[self.theEquationName][iBFaces] += local_FluxFb
        # Region.fluxes.FluxVf[self.theEquationName][iBFaces] += local_FluxVb
        Region.fluxes.FluxTf[self.theEquationName][iBFaces] += local_FluxCb*phi_C+ local_FluxFb*phi_b #+ local_FluxVb


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
        gi = Q_(Region.dictionaries.g['value'][iComponent],dm.acceleration_dim)
        #   Update and store
        Region.fluxes.FluxT[self.theEquationName] += rho*volumes*gi
 
    def cfdAssemblePressureGradientTerm(self,Region,*args):
        #   Get info and fields
        volumes = Region.mesh.elementVolumes
        p_grad = np.squeeze(Region.fluid['p'].Grad.phi[:Region.mesh.numberOfElements,self.iComponent])
        #   Update global fluxes
        Region.fluxes.FluxT[self.theEquationName] += volumes*p_grad

    
