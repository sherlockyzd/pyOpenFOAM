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
        bc = Region.coefficients.bc
        if Region.MatrixFormat == 'acnb':
            ac = Region.coefficients.ac
        elif Region.MatrixFormat == 'ldu':
            ac = Region.coefficients.Diag
        elif Region.MatrixFormat == 'csr':
            ac = Region.coefficients.csrdata[Region.coefficients._indptr[:-1]]
        elif Region.MatrixFormat == 'coo':
            ac = Region.coefficients.coodata[Region.coefficients._coodiagPositions]
        else:
            raise ValueError(f"Unsupported MatrixFormat: {Region.MatrixFormat}")

        if theEquationName=='p':
            # % Fore pressure correction equation, the divergence of the mass flow
            # % rate is the residual.          
            # % Scale with scale value (max value)
            p_scale = Region.fluid['p'].scale
            maxResidual = max(abs(bc)/(ac*p_scale))
            rmsResidual = mth.cfdResidual(abs(bc)/(ac*p_scale),'RMS')
        else:
            # % Other equations ... (向量化实现)
            theNumberOfElements = Region.mesh.numberOfElements
            volumes = Region.mesh.elementVolumes.value[:theNumberOfElements]
            if not Region.STEADY_STATE_RUN:
                rho_val = Region.fluid['rho'].phi.value[:theNumberOfElements]
                # rho 可能是 (Ne,1) 或 (Ne,) 标量场，统一展平为 1D
                rho = np.asarray(rho_val).ravel().copy()
                if theEquationName== 'T':
                    try:
                        Cp_val = Region.fluid['kappa'].phi.value[:theNumberOfElements]
                        Cp = np.asarray(Cp_val).ravel().copy()
                        rho *=  Cp
                    except AttributeError:
                        pass

                deltaT = Region.dictionaries.controlDict['deltaT']
                # 向量化：一次计算所有单元的 local_ac
                local_ac = ac.copy()
                at = volumes * rho / deltaT
                local_ac -= at
                # 防止 local_ac 过小（向量化条件赋值）
                local_ac = np.where(local_ac < 1e-6 * at, at, local_ac)

                local_residual = bc / (local_ac * scale)
                maxResidual = np.max(np.abs(local_residual))
                rmsResidual = np.sqrt(np.sum(local_residual ** 2) / theNumberOfElements)
            else:
                local_residual = bc / (ac * scale)
                maxResidual = np.max(np.abs(local_residual))
                rmsResidual = np.sqrt(np.sum(local_residual ** 2) / theNumberOfElements)

        if len(args)==1:
            iComponent = args[0]
            self.maxResidual[iComponent]=maxResidual
            self.rmsResidual[iComponent]=rmsResidual
        else:
            self.maxResidual=maxResidual
            self.rmsResidual=rmsResidual
