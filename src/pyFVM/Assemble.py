import numpy as np
import os
import pyFVM.IO as io
# import pyFVM.Field as field
import pyFVM.Math as mth
import pyFVM.Interpolate as interp
import pyFVM.cfdGetTools as tools

class Assemble:
    def __init__(self,Region,theEquationName):
        """ Initiates the Assemble class instance
        """
        self.theEquationName=theEquationName
        ## The instance of Equation stored in the self.region.model dictionary
        self.theEquation=Region.model.equations[self.theEquationName]

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
            Region.fluid[self.theEquationName].phiGrad.cfdGetGradientSubArrayForInterior(Region,iComponent)
        else:
            self.iComponent= int(-1)
            Region.fluid[self.theEquationName].phiGrad.cfdGetGradientSubArrayForInterior(Region)

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
                self.cfdAssembleMomentumDivergenceCorrectionTerm(Region)
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
        #   Get coefficients
        theDUField  = 'DU'+str(iComponent)
        theDUTField = 'DUT'+str(iComponent)
        # if strcmp(cfdGetAlgorithm,'SIMPLE')    
        Region.fluid[theDUField].phi[0:theNumberOfElements,0] = Region.mesh.elementVolumes/Region.coefficients.ac
        Region.fluid[theDUTField].phi[0:theNumberOfElements,0]= Region.coefficients.ac_old/Region.coefficients.ac
        # elif(strcmp(theAlgorithm,'PIMPLE'))
            #   BODGE
        #   Store in data base
        #   Update at cfdBoundary patches
        Region.fluid[theDUField].updateFieldForAllBoundaryPatches(Region)
        Region.fluid[theDUTField].updateFieldForAllBoundaryPatches(Region)

    def cfdAssembleIntoGlobalMatrixElementFluxes(self,Region,*args):
        """
        Add the face and volume contributions to obtain ac, bc and ac_old
        These are the ac and bc coefficients in the linear system of equations
        """
        Region.coefficients.ac      += Region.fluxes.FluxC
        Region.coefficients.ac_old  += Region.fluxes.FluxC_old
        Region.coefficients.bc      -= Region.fluxes.FluxT

    def cfdAssembleIntoGlobalMatrixFaceFluxes(self,Region,*args):
        #   Assemble fluxes of interior faces
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        numberOfFaces=Region.mesh.numberOfFaces

        for iFace in range(numberOfInteriorFaces):
            own = Region.mesh.owners[iFace]
            nei = Region.mesh.neighbours[iFace]
            own_anb_index = Region.mesh.upperAnbCoeffIndex[iFace]
            nei_anb_index = Region.mesh.lowerAnbCoeffIndex[iFace]
            # Assemble fluxes for owner cell    Region.coefficients.anb[0][0]
            Region.coefficients.ac[own]                       +=  Region.fluxes.FluxCf[iFace]
            Region.coefficients.anb[own][own_anb_index] +=  Region.fluxes.FluxFf[iFace]
            Region.coefficients.bc[own]                       -=  Region.fluxes.FluxTf[iFace]
            #   Assemble fluxes for neighbour cell
            Region.coefficients.ac[nei]                       -=  Region.fluxes.FluxFf[iFace] #需调换顺序
            Region.coefficients.anb[nei][nei_anb_index] -=  Region.fluxes.FluxCf[iFace]
            Region.coefficients.bc[nei]                       +=  Region.fluxes.FluxTf[iFace]

        #   Assemble fluxes of cfdBoundary faces
        for iBFace in range(numberOfInteriorFaces,numberOfFaces):
            own = Region.mesh.owners[iBFace]
            #   Assemble fluxes for owner cell
            Region.coefficients.ac[own]   +=  Region.fluxes.FluxCf[iBFace]
            Region.coefficients.bc[own]   -=  Region.fluxes.FluxTf[iBFace]

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
        Region.fluxes.FluxC.fill(0)
        Region.fluxes.FluxV.fill(0)
        Region.fluxes.FluxT.fill(0)
        Region.fluxes.FluxC_old.fill(0)

    def cfdZeroFaceFLUXCoefficients(self,Region):
        # print('Inside cfdZeroFaceFLUXCoefficients')
        Region.fluxes.FluxCf.fill(0)
        Region.fluxes.FluxVf.fill(0)
        Region.fluxes.FluxTf.fill(0)
        Region.fluxes.FluxFf.fill(0)

    def cfdPreAssembleContinuityEquation(self,Region,*args):
        '''初始化连续性方程的组装，系数置零，并初始化压力修正方程'''
        self.cfdPreAssembleEquation(Region)
        #   Get DU field
        theNumberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        DU0_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear',tools.cfdGetSubArrayForInterior('DU0',Region))
        DU1_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear', tools.cfdGetSubArrayForInterior('DU1',Region))
        DU2_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear', tools.cfdGetSubArrayForInterior('DU2',Region))    
        #   Assemble Coefficients
        #   assemble term I
        #       rho_f [v]_f.Sf
        # local_FluxVf += rho_f*(U_bar_f*Sf).sum(1)
        #   Assemble term II and linearize it
        #        - rho_f ([DPVOL]_f.P_grad_f).Sf
        Sf = Region.mesh.faceSf[0:theNumberOfInteriorFaces,:]
        #   Calculated info
        e = Region.mesh.faceCFn[0:theNumberOfInteriorFaces,:]
        DUSf = np.column_stack((DU0_f*Sf[:,0],DU1_f*Sf[:,1],DU2_f*Sf[:,2]))
        Region.fluid['DUSf'].phi[0:theNumberOfInteriorFaces,:]=DUSf
        magDUSf = mth.cfdMag(DUSf)
        if Region.mesh.OrthogonalCorrectionMethod=='Minimum':
            Region.fluid['DUEf'].phi[0:theNumberOfInteriorFaces,:] = (DUSf*e).sum(1)[:,None]*e
        elif Region.mesh.OrthogonalCorrectionMethod=='Orthogonal':
            Region.fluid['DUEf'].phi[0:theNumberOfInteriorFaces,:] =magDUSf[:,None]*e
        elif Region.mesh.OrthogonalCorrectionMethod=='OverRelaxed':
            eDUSf = mth.cfdUnit(DUSf)
            epsilon = 1e-10  # 定义一个小的正常数
            Region.fluid['DUEf'].phi[0:theNumberOfInteriorFaces,:] =(magDUSf/((eDUSf*e + epsilon).sum(1)))[:,None]*e
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
            DU0_b = tools.cfdGetSubArrayForBoundaryPatch('DU0', iBPatch,Region)
            DU1_b = tools.cfdGetSubArrayForBoundaryPatch('DU1', iBPatch,Region)
            DU2_b = tools.cfdGetSubArrayForBoundaryPatch('DU2', iBPatch,Region)
            DUSb = np.column_stack((DU0_b*Sf_b[:,1],DU1_b*Sf_b[:,2],DU2_b*Sf_b[:,2]))
            Region.fluid['DUSf'].phi[iBFaces,:]=DUSb
            magSUDb = mth.cfdMag(DUSb)
            if Region.mesh.OrthogonalCorrectionMethod=='Minimum':
                Region.fluid['DUEf'].phi[iBFaces,:] = (DUSb*e).sum(1)[:,None]*e
            elif Region.mesh.OrthogonalCorrectionMethod=='Orthogonal':
                Region.fluid['DUEf'].phi[iBFaces,:] =magSUDb[:,None]*e
            elif Region.mesh.OrthogonalCorrectionMethod=='OverRelaxed':
                epsilon = 1e-10  # 定义一个小的正常数
                eDUSb =  mth.cfdUnit(DUSb)
                Region.fluid['DUEf'].phi[iBFaces,:] =(magSUDb/((eDUSb*e + epsilon).sum(1)))[:,None]*e
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
            Region.fluid[self.theEquationName].phiGrad.cfdGetGradientSubArrayForInterior(Region,iComponent)
        else:
            self.iComponent=-1
            Region.fluid[self.theEquationName].phiGrad.cfdGetGradientSubArrayForInterior(Region)

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

    def cfdAssembleDiagDominance(self,Region,*args):
    # ==========================================================================
    # Enforce Diagonal Dominance as this may not be ensured
    # --------------------------------------------------------------------------
    # Get info and fields
        for iElement in range(Region.mesh.numberOfElements):
            theNumberOfNeighbours = len(Region.coefficients.theCConn[iElement])
            SumAik = 0
            #   adding all the off diagonal pressure terms
            for k in range(theNumberOfNeighbours):
                Region.coefficients.anb[iElement][k]=min(Region.coefficients.anb[iElement][k],1e-10)
                SumAik -= Region.coefficients.anb[iElement][k]     
            Region.coefficients.ac[iElement] = max(Region.coefficients.ac[iElement],SumAik)

    def cfdAssembleMassDivergenceAdvectionTerm(self,Region):
        io.cfdError('cfdCompressible solver not yet written')

    def cfdStoreMassFlowRate(self,Region):
        Region.fluid['mdot_f'].phi=Region.fluxes.FluxVf[:,None]
    
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
                    io.cfdErrorr(theBCType+'<<<< not implemented')
            elif thePhysicalType=='inlet':
                if theBCType=='inlet' or theBCType=='zeroGradient':
                    #Specified Velocity
                    self.cfdAssembleMassDivergenceTermInletZeroGradientBC(Region,iBPatch)
                elif theBCType=='fixedValue':
                    #Specified Pressure and Velocity Direction
                    self.cfdAssembleMassDivergenceTermInletFixedValueBC(Region,iBPatch)
                else:
                    #Specified Total Pressure and Velocity Direction
                    io.cfdErrorr(theBCType+'<<<< not implemented')
            
            elif thePhysicalType=='outlet':
                if theBCType=='outlet' or theBCType=='zeroGradient':
                    #Specified Mass Flow Rate
                    self.cfdAssembleMassDivergenceTermOutletZeroGradientBC(Region,iBPatch)
                elif theBCType=='fixedValue':
                    #Specified Pressure
                    self.cfdAssembleMassDivergenceTermOutletFixedValueBC(Region,iBPatch)
                else:
                    io.cfdErrorr(theBCType+'<<<< not implemented')
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
        local_FluxCb = np.zeros(theNumberOfBFaces)
        local_FluxFb = np.zeros(theNumberOfBFaces)
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
        U_b = (Sf_b*U_b).sum(1)
        local_FluxVb = rho_b*U_b
        local_FluxVb += rho_b*((p_grad_b-p_grad_C)*Region.fluid['DUSf'].phi[iBFaces,:]).sum(1)
        #   Total flux
        #   local_FluxTb = local_FluxCb.*p(owners_b) + local_FluxFb.*p_b + local_FluxVb;
        #   Update global fluxes
        Region.fluxes.FluxCf[iBFaces] =  local_FluxCb
        Region.fluxes.FluxFf[iBFaces] =  local_FluxFb
        Region.fluxes.FluxVf[iBFaces] =  local_FluxVb
        #  theFluxes.FluxTf(iBFaces) =  local_FluxTb


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
        #   Assemble Coefficients
        #   The DU field for cfdBoundary
        CF_b = Region.mesh.faceCF[iBFaces]
        geoDiff = mth.cfdMag(Region.fluid['DUEf'].phi[iBFaces,:])/mth.cfdMag(CF_b)
        #   Initialize local fluxes
        local_FluxVb =np.zeros(theNumberOfBFaces)
        #   Assemble term I
        # U_bar_b = (U_b*Sf_b).sum(1)
        local_FluxVb += rho_b*(U_b*Sf_b).sum(1)
        #   Assemble term II and linearize it
        local_FluxCb =  rho_b*geoDiff
        # local_FluxFb = -rho_b*geoDiff
        local_FluxFb = np.zeros(theNumberOfBFaces)
        #   Assemble term III
        local_FluxVb += rho_b*((p_grad_b-p_grad_C)*Region.fluid['DUSf'].phi[iBFaces,:]).sum(1)
        #   local_FluxTb = local_FluxCb.*p(owners_b) + local_FluxFb.*p_b + local_FluxVb;
        #   Update global fluxes
        Region.fluxes.FluxCf[iBFaces] = local_FluxCb
        Region.fluxes.FluxFf[iBFaces] = local_FluxFb
        Region.fluxes.FluxVf[iBFaces] = local_FluxVb
        #   theFluxes.FluxTf(iBFaces) =  local_FluxTb;



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
        # Region.fluxes.FluxVf(iBFaces) = local_FluxVb
        # #   theFluxes.FluxTf(iBFaces,1) =  local_FluxTb;


    def cfdAssembleMassDivergenceTermInterior(self,Region,*args):
        #   Get mesh info
        theNumberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        # iFaces = list(range(Region.mesh.numberOfInteriorFaces))
        # owners_f = cfdGetOwnersSubArrayForInteriorFaces
        # neighbours_f = cfdGetNeighboursSubArrayForInteriorFaces
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
            rho_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,scheme, rho, mdot_f_prev)
        else:
            rho_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,scheme, rho)

        #   Get velocity field and interpolate to faces
        U_bar_f = interp.cfdInterpolateFromElementsToInteriorFaces(Region,'linear', tools.cfdGetSubArrayForInterior('U',Region))

        #   Initialize local fluxes
        local_FluxCf = rho_f*geoDiff
        local_FluxFf = -rho_f*geoDiff
        local_FluxVf = rho_f*(U_bar_f*Sf-p_RhieChowValue*Region.fluid['DUSf'].phi[0:theNumberOfInteriorFaces]).sum(1)

        if Region.pp_nonlinear_corrected:
            Region.fluid['pprime'].phiGrad.cfdUpdateGradient(Region)
            DUTf = Region.fluid['DUSf'].phi - Region.fluid['DUEf'].phi
            pp_RhieChowValue=tools.cfdRhieChowValue('pprime',Region)
            local_FluxVf += rho_f*(pp_RhieChowValue*DUTf[0:theNumberOfInteriorFaces]).sum(1)

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
        Region.fluxes.FluxCf[0:theNumberOfInteriorFaces]  = local_FluxCf
        Region.fluxes.FluxFf[0:theNumberOfInteriorFaces]  = local_FluxFf
        Region.fluxes.FluxVf[0:theNumberOfInteriorFaces]  = local_FluxVf
        #   theFluxes.FluxTf(iFaces,1) = local_FluxTf;



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
        # if theScheme == 'steadyState':
        #     pass
        # el
        if theScheme == 'Euler':
            # if args:
            #     iComponent=args[0]
            #     self.assembleFirstOrderEulerTransientTerm(Region,iComponent)
            # else:
            self.assembleFirstOrderEulerTransientTerm(Region)
        else:
            io.cfdError(theScheme+' ddtScheme is incorrect')

    def assembleFirstOrderEulerTransientTerm(self,Region,*args):
        """Assembles first order transient euler term
        这段Python代码定义了一个名为`assembleFirstOrderEulerTransientTerm`的方法，它用于组装一阶欧拉瞬态项，这在计算流体动力学（CFD）中是常见的操作。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def assembleFirstOrderEulerTransientTerm(self, Region):` 定义了一个方法，接收一个参数`Region`，它是一个包含模拟区域信息的对象。

        2. **获取元素体积**：
        - `self.volumes = Region.mesh.elementVolumes.reshape(-1,1)` 将区域网格的元素体积转换为NumPy数组，并重塑为二维列向量。

        3. **获取场的子数组**：
        - 方法调用`Region.fluid[self.theEquationName].cfdGetSubArrayForInterior(Region)`和`cfdGetPrevTimeStepSubArrayForInterior(Region)`用于获取当前和上一时间步的内部场子数组。

        4. **初始化场变量**：
        - `self.phi` 和 `self.phi_old` 分别存储当前和上一时间步的场内部子数组。
        - `self.rho` 和 `self.rho_old` 分别存储当前和上一时间步的密度内部子数组。

        5. **获取时间步长**：
        - `self.deltaT = Region.dictionaries.controlDict['deltaT']` 获取控制字典中的`deltaT`，即时间步长。

        6. **计算通量系数**：
        - `local_FluxC` 计算当前时间步的通量系数，它是体积、密度和时间步长的函数。
        - `local_FluxC_old` 计算上一时间步的通量系数，注意符号相反。
        - `local_FluxV` 初始化为零，表示没有源项或体积力。
        - `local_FluxT` 计算总通量，它是`local_FluxC`和`local_FluxC_old`与相应场变量的乘积。

        7. **更新通量数组**：
        - 将计算得到的通量系数更新到`Region.fluxes`对象中，包括`FluxC`、`FluxC_old`、`FluxV`和`FluxT`。

        ### 注意事项：
        - 这段代码假设`Region`对象具有网格信息、流体属性、控制字典和通量对象。
        - `self.theEquationName`应该是当前正在处理的方程的名称。
        - 使用`np.asarray()`和`reshape(-1,1)`确保元素体积数组是二维的，这有助于后续的数组运算。
        - `np.squeeze()`用于去除数组中的单维度条目，确保进行正确的逐元素乘法。
        - 代码中的注释提供了对关键步骤的说明，有助于理解每个计算步骤的目的。

        这个方法是CFD模拟中数值求解过程的一部分，用于组装瞬态项，这对于求解流体动力学方程是必要的。通过这种方式，可以方便地访问和更新通量信息，以实现模拟的数值求解。
        """   
        volumes = Region.mesh.elementVolumes[:,np.newaxis]
        deltaT = Region.dictionaries.controlDict['deltaT']
        local_FluxC = np.squeeze(volumes*Region.fluid['rho'].phi[:Region.mesh.numberOfElements])/deltaT 
        local_FluxC_old = -np.squeeze(volumes*Region.fluid['rho'].phi_old[:Region.mesh.numberOfElements])/deltaT
        # local_FluxT = np.squeeze(np.multiply(local_FluxC[:, np.newaxis],np.squeeze(self.phi))) + np.multiply(local_FluxC_old[:, np.newaxis],np.squeeze(self.phi_old))
        Region.fluxes.FluxC = local_FluxC
        Region.fluxes.FluxC_old = local_FluxC_old

        # local_FluxV = np.zeros(len(local_FluxC))
        Region.fluxes.FluxV = np.zeros(len(local_FluxC),dtype=float)
        Region.fluxes.FluxT= (Region.fluxes.FluxC*Region.fluid[self.theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
                              +Region.fluxes.FluxC_old*Region.fluid[self.theEquationName].phi_old[:Region.mesh.numberOfElements,self.iComponent])

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
            print([theScheme, ' divScheme incorrect\n'])
            os.exit()

    def processDCSOUScheme(self,Region):
        #   Get mesh info
        theElementCentroids = Region.mesh.elementCentroids
        theFaceCentroids = Region.mesh.faceCentroids
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces

        #   Get fields
        theEquationName=self.theEquationName
        phi=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        gradPhi = Region.fluid[theEquationName].phiGrad.phiGradInter
        mdot_f = Region.fluid['mdot_f'].phi[:Region.mesh.numberOfInteriorFaces]
        pos = np.zeros(len(mdot_f),dtype=int)
        pos[np.squeeze(mdot_f) > 0] = 1
        iFaces = list(range(numberOfInteriorFaces))
        owners_f = Region.mesh.owners[iFaces]
        neighbours_f = Region.mesh.neighbours[iFaces]
        iUpwind = pos*owners_f + (1-pos)*neighbours_f
        #   Get the upwind gradient at the interior faces
        phiGradC = gradPhi[iUpwind,:]
        #   Interpolated gradient to interior faces
        # phiGrad_f = cfdInterpolateGradientsFromElementsToInteriorFaces('Gauss linear corrected', gradPhi, phi);     cfdInterpolateGradientsFromElementsToInteriorFaces
        phiGrad_f = interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,gradPhi,'Gauss linear corrected',phi)

        rC = theElementCentroids[iUpwind,:]
        rf = theFaceCentroids[iFaces,:]
        rCf = rf - rC
        # for i in range(numberOfInteriorFaces)
        #     dc_corr[i,:] = mdot_f[i] * np.dot(2*phiGradC[i,:]-phiGrad_f[i,:],rCf[i,:])
        dc_corr = np.squeeze(mdot_f * np.multiply(2*phiGradC-phiGrad_f,rCf).sum(1)[:,np.newaxis])

        #   Update global fluxes
        Region.fluxes.FluxTf[iFaces] +=  dc_corr

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
                    print([theBCType,'<<<< Not implemented'])
                    os.exit()
                
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
        mdot_b = Region.fluid['mdot_f'].phi[iBFaces]

        #   linear fluxes
        local_FluxCb =  max(mdot_b,0)
        local_FluxFb = -max(-mdot_b,0)# 《The SIMPLE Algorithm》 Page57 eq6.85

        #   Non-linear fluxes
        # local_FluxVb = np.zeros(len(local_FluxCb))
        #   Update global fluxes
        Region.fluxes.FluxCf[iBFaces]  = local_FluxCb
        Region.fluxes.FluxFf[iBFaces]  = local_FluxFb
        # Region.fluxes.FluxVf[iBFaces]  = local_FluxVb
        Region.fluxes.FluxTf[iBFaces]  = local_FluxCb*Ui_C + local_FluxFb*Ui_b #+ local_FluxVb

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
        mdot_b = Region.fluid['mdot_f'].phi[iBFaces]

        #   linear fluxes
        local_FluxCb = mdot_b
        local_FluxFb = np.zeros(len(local_FluxCb))
        #   Non-linear fluxes
        # local_FluxVb = np.zeros(len(local_FluxCb))
        #   Update global fluxes
        Region.fluxes.FluxCf[iBFaces]  = local_FluxCb
        Region.fluxes.FluxFf[iBFaces]  = local_FluxFb
        # Region.fluxes.FluxVf[iBFaces]  = local_FluxVb
        Region.fluxes.FluxTf[iBFaces]  = local_FluxCb*Ui_C
        # + local_FluxFb*Ui_b[iBElements] + local_FluxVb


    def cfdAssembleConvectionTermInterior(self,Region,*args):
        theEquationName=self.theEquationName
        # Region.fluid[theEquationName].cfdGetSubArrayForInterior(Region)
        phi=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        numberOfInteriorFaces = Region.mesh.numberOfInteriorFaces
        mdot_f=np.squeeze(Region.fluid['mdot_f'].phi[0:numberOfInteriorFaces])
        local_FluxCf=np.maximum(mdot_f,0)  #《The SIMPLE Algorithm》Chapter 5. Page39 公式5.66
        local_FluxFf=-np.maximum(-mdot_f,0)
        local_FluxVf=np.zeros(len(local_FluxCf))
        Region.fluxes.FluxCf[:numberOfInteriorFaces] = local_FluxCf
        Region.fluxes.FluxFf[:numberOfInteriorFaces] = local_FluxFf
        Region.fluxes.FluxVf[:numberOfInteriorFaces] = local_FluxVf
        # Region.fluxes.FluxTf[0:numberOfInteriorFaces] = np.multiply(local_FluxCf[:, np.newaxis],np.squeeze(phi[self.owners_interior_f]))+np.multiply(local_FluxFf[:, np.newaxis],np.squeeze(phi[self.neighbours_interior_f]))+ local_FluxVf[:, np.newaxis]
        # for field in Region.fluxes.FluxTf:
        # if args:
        #     iComponent=args[0]
        #     Region.fluxes.FluxTf[0:numberOfInteriorFaces] = Region.fluxes.FluxCf[0:numberOfInteriorFaces]*phi[Region.mesh.interiorFaceOwners,iComponent]+Region.fluxes.FluxFf[0:numberOfInteriorFaces]*phi[Region.mesh.interiorFaceNeighbours,iComponent]+ Region.fluxes.FluxVf[0:numberOfInteriorFaces]
        # else:
        Region.fluxes.FluxTf[0:numberOfInteriorFaces] = Region.fluxes.FluxCf[0:numberOfInteriorFaces]*np.squeeze(phi[Region.mesh.interiorFaceOwners])+Region.fluxes.FluxFf[0:numberOfInteriorFaces]*np.squeeze(phi[Region.mesh.interiorFaceNeighbours])+ Region.fluxes.FluxVf[0:numberOfInteriorFaces]


    def cfdAssembleMomentumDivergenceCorrectionTerm(self,Region,*args):
        effDiv = self.cfdComputeEffectiveDivergence(Region,self.theEquationName)
        numberOfElements = Region.mesh.numberOfElements
        # 计算 FluxC, FluxV, FluxT
        Ui=Region.fluid[self.theEquationName].phi[:numberOfElements,self.iComponent]
        max_effDiv = np.maximum(effDiv[:numberOfElements], 0.0)
        Region.fluxes.FluxC[:numberOfElements] = max_effDiv - effDiv[:numberOfElements]
        Region.fluxes.FluxV[:numberOfElements] = -max_effDiv * Ui
        Region.fluxes.FluxT[:numberOfElements] = Region.fluxes.FluxC[:numberOfElements] * Ui + Region.fluxes.FluxV[:numberOfElements]

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
        mdot_f = Region.fluid['mdot_f'].phi
        if theEquationName=='T':
            try:
                Cp = Region.fluid['Cp'].phi
                Cp_f = interp.interpolateFromElementsToFaces(Region,'linear', Cp)
                mdot_f = mdot_f * Cp_f
            except:
                print('Cp field is not available')
                # os.exit()
        #   Initialize effective divergence array
        effDiv = np.zeros(theNumberOfFaces)
        #   Interior Faces Contribution
        owners = np.array(owners)
        neighbours = np.array(neighbours)
        mdot_f = np.squeeze(mdot_f)
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
        phi=Region.fluid[theEquationName].phi[:Region.mesh.numberOfElements,self.iComponent]
        gamma_f=interp.cfdinterpolateFromElementsToFaces(Region,'harmonic mean',Region.model.equations[self.theEquationName].gamma)#调和平均值，《FVM》P226
        local_FluxCf  = np.squeeze(gamma_f[0:numberOfInteriorFaces])*Region.mesh.geoDiff_f[0:numberOfInteriorFaces]
        local_FluxFf  =-np.squeeze(gamma_f[0:numberOfInteriorFaces])*Region.mesh.geoDiff_f[0:numberOfInteriorFaces]

        # 处理非正交修正项（如果需要）
        # Interpolated gradients on interior faces
        gradPhi_f=interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,Region.fluid[theEquationName].phiGrad.phiGradInter,'linear')
        local_FluxVf  =-np.squeeze(gamma_f[:numberOfInteriorFaces])*(gradPhi_f*Region.mesh.faceTf[:numberOfInteriorFaces,:]).sum(1)
        if theEquationName=='U':
            gradPhi_f=interp.cfdInterpolateGradientsFromElementsToInteriorFaces(Region,Region.fluid[theEquationName].phiGrad.phiGradInter_TR,'linear')
            local_FluxVf += np.squeeze(gamma_f[0:numberOfInteriorFaces])*(gradPhi_f*Region.mesh.interiorFaceSf).sum(1)
            if Region.cfdIsCompressible:
                local_FluxVf += 2.0/3.0*np.squeeze(gamma_f[0:numberOfInteriorFaces])*(Region.fluid[theEquationName].phiGrad.phiGrad_Trace[0:numberOfInteriorFaces]*Region.mesh.interiorFaceSf[:,self.iComponent])

        Region.fluxes.FluxCf[0:numberOfInteriorFaces] = local_FluxCf
        Region.fluxes.FluxFf[0:numberOfInteriorFaces] = local_FluxFf
        Region.fluxes.FluxVf[0:numberOfInteriorFaces] = local_FluxVf
        Region.fluxes.FluxTf[0:numberOfInteriorFaces] = Region.fluxes.FluxCf[0:numberOfInteriorFaces]*np.squeeze(phi[Region.mesh.interiorFaceOwners])+Region.fluxes.FluxFf[0:numberOfInteriorFaces]*np.squeeze(phi[Region.mesh.interiorFaceNeighbours])+ Region.fluxes.FluxVf[0:numberOfInteriorFaces]

        
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
        # U_b =Region.fluid[theEquationName].phi[iBElements] #三个方向的都有
        #   Get required fields
        mu_b = np.squeeze(Region.model.equations[self.theEquationName].gamma[owners_b])


        u_C = phi_C[:,0]
        v_C = phi_C[:,1]
        w_C = phi_C[:,2]
        # u_b = U_b[:,0]
        # v_b = U_b[:,1]
        # w_b = U_b[:,2]
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

        Region.fluxes.FluxCf[iBFaces] = local_FluxCb
        # Region.fluxes.FluxFf[iBFaces] = local_FluxFb
        Region.fluxes.FluxVf[iBFaces] = local_FluxVb
        Region.fluxes.FluxTf[iBFaces] = local_FluxCb*phi_C[:,iComponent] + local_FluxVb#+ local_FluxFb*phi_b[:,iComponent]

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
                local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-nx2)
                local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*(u_b*(1-nx2)+(v_C-v_b)*ny*nx+(w_C-w_b)*nz*nx)
            elif iComponent==1:
                local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-ny2)
                local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*((u_C-u_b)*nx*ny+v_b*(1-ny2)+(w_C-w_b)*nz*ny)
            elif iComponent==2:
                local_FluxCb =  mu_b*Region.mesh.geoDiff_f[iBFaces]*(1-nz2)
                local_FluxVb = -mu_b*Region.mesh.geoDiff_f[iBFaces]*((u_C-u_b)*nx*nz+(v_C-v_b)*ny*nz+w_b*(1-nz2))
            local_FluxFb =np.zeros(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])
        else:
            local_FluxCb=mu_b*Region.mesh.geoDiff_f[iBFaces]
            local_FluxFb=-mu_b*Region.mesh.geoDiff_f[iBFaces]
            local_FluxVb=np.zeros(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])


        #   Update global fluxes
        # local_FluxFb =np.zeros(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])

        Region.fluxes.FluxCf[iBFaces] = local_FluxCb
        # Region.fluxes.FluxFf[iBFaces] = local_FluxFb
        Region.fluxes.FluxVf[iBFaces] = local_FluxVb
        Region.fluxes.FluxTf[iBFaces] = local_FluxCb*phi_C[:,iComponent]  + local_FluxFb*phi_b[:,iComponent]+ local_FluxVb
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

        Region.fluxes.FluxCf[iBFaces] = local_FluxCb
        # Region.fluxes.FluxFf[iBFaces] = local_FluxFb
        Region.fluxes.FluxVf[iBFaces] = local_FluxVb
        Region.fluxes.FluxTf[iBFaces] = local_FluxCb*phi_C[:,iComponent] + local_FluxVb #+ local_FluxFb*U_b[:,iComponent]


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
        local_FluxFb = -mu_b*Region.mesh.geoDiff_f[iBFaces]
        #   Update global fluxes
        local_FluxVb = np.zeros(Region.mesh.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces'])

        #   Get fields
        phi_C =Region.fluid[theEquationName].phi[owners_b,self.iComponent]
        phi_b =Region.fluid[theEquationName].phi[iBElements,self.iComponent]

        Region.fluxes.FluxCf[iBFaces] = local_FluxCb
        Region.fluxes.FluxFf[iBFaces] = local_FluxFb
        Region.fluxes.FluxVf[iBFaces] = local_FluxVb
        Region.fluxes.FluxTf[iBFaces] = local_FluxCb*phi_C+ local_FluxFb*phi_b + local_FluxVb


    def cfdAssembleStressTermZeroGradientBC(self,Region,iBPatch):
        self.cfdAssembleStressTermSpecifiedValue(Region,iBPatch)

    def cfdAssembleStressTermEmptyBC(self,*args):
        pass
                
    def cfdAssembleMomentumBuoyancyTerm(self,Region,*args):
        #   Get info
        iComponent=self.iComponent
        volumes = np.squeeze(Region.mesh.elementVolumes)
        #   Get fields
        rho = np.squeeze(Region.fluid['rho'].phi[:Region.mesh.numberOfElements])
        #   Get gravity
        gi = Region.dictionaries.g['value'][iComponent]
        #   Update and store
        Region.fluxes.FluxT = rho*volumes*gi
 
    def cfdAssemblePressureGradientTerm(self,Region,*args):
        #   Get info and fields
        volumes = np.squeeze(Region.mesh.elementVolumes)
        p_grad = np.squeeze(Region.fluid['p'].phiGrad.phiGrad[:Region.mesh.numberOfElements,self.iComponent])
        #   Update global fluxes
        Region.fluxes.FluxT = volumes*p_grad

    
