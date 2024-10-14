# import os
import numpy as np
import cfdtool.Math as mth
# import pyFVM.Coefficients as coefficients
# import pyFVM.Solve as solve
# import pyFVM.Scalar as scalar
# import pyFVM.Assemble as assemble

class Equation():
    
    def __init__(self,fieldName):
        
        # self.Region = Region
        self.name=fieldName
        self.initializeResiduals()
        
    def initializeResiduals(self):
        if self.name == 'U':
            self.rmsResidual =[1,1,1]
            self.maxResidual =[1,1,1]
            self.initResidual =[1,1,1]
            self.finalResidual =[1,1,1]
        else:
            self.rmsResidual = 1
            self.maxResidual = 1
            self.initResidual = 1
            self.finalResidual = 1
        
    def setTerms(self, terms):  
        self.terms = terms

    def cfdComputeScaledRMSResiduals(self,Region,*args):
        theEquationName=self.name
        scale = Region.fluid[theEquationName].scale

        # % Get coefficients
        ac = Region.coefficients.ac
        bc = Region.coefficients.bc

        if theEquationName=='p':
            # % Fore pressure correction equation, the divergence of the mass flow
            # % rate is the residual.          
            # % Scale with scale value (max value)
            p_scale = Region.fluid['p'].scale
            maxResidual = max(abs(bc)/(ac*p_scale))
            rmsResidual = mth.cfdResidual(abs(bc)/(ac*p_scale),'RMS')
        else:
            # % Other equations ...
            # % Get info
            theNumberOfElements = Region.mesh.numberOfElements
            volumes = Region.mesh.elementVolumes.value
            # % Another approach which takes the transient term into consideration
            if not Region.STEADY_STATE_RUN:
                rho = Region.fluid['rho'].phi.value.copy()
                if theEquationName== 'T':
                    try:
                        Cp = Region.fluid['kappa'].phi.value.copy()
                        rho *=  Cp
                    except AttributeError:
                        pass

                deltaT = Region.dictionaries.controlDict['deltaT']
                theMaxResidualSquared = 0.
                theMaxScaledResidual = 0
                for iElement in range(theNumberOfElements):
                    volume = volumes[iElement]
                    local_ac = ac[iElement]
                    if not Region.STEADY_STATE_RUN:
                        at = volume*rho[iElement]/deltaT
                        local_ac -= at
                        if local_ac < 1e-6*at:
                            local_ac = at

                    local_residual  = bc[iElement]
                    local_residual /= local_ac*scale
                    theMaxScaledResidual   = max(theMaxScaledResidual,abs(local_residual))
                    theMaxResidualSquared +=  local_residual*local_residual

                maxResidual = theMaxScaledResidual
                rmsResidual = np.sqrt(theMaxResidualSquared/theNumberOfElements)   
            else:
                theMaxResidualSquared = 0
                theMaxScaledResidual = 0
                for iElement in range(theNumberOfElements):
                    local_residual = bc[iElement]
                    local_residual = local_residual/(ac[iElement]*scale)
  
                    theMaxScaledResidual   = max(theMaxScaledResidual,abs(local_residual))
                    theMaxResidualSquared += local_residual*local_residual
   
                maxResidual = theMaxScaledResidual
                rmsResidual = np.sqrt(theMaxResidualSquared/theNumberOfElements)

        if len(args)==1:
            iComponent = args[0]
            self.maxResidual[iComponent]=maxResidual
            self.rmsResidual[iComponent]=rmsResidual
        else:
            self.maxResidual=maxResidual
            self.rmsResidual=rmsResidual
