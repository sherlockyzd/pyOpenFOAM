import os
import pyFVM.Field as field
# import pyFVM.Coefficients as coefficients
# import pyFVM.Solve as solve
# import pyFVM.Scalar as scalar
import pyFVM.Gradient as grad
import pyFVM.Equation as equation

class Model():
    def __init__(self,Region):
        
        # self.Region = Region
        # self.name=fieldName
        # self.initializeResiduals()
        #Define transient-convection equation
        # if not os.path.isfile(gravityFilePath):
        #     print('\n\nNo g file found\n')
        #     pass
        # else:
        self.equations = {}
        # Create theEquation structure
        self.DefineMomentumEquation(Region)
        self.DefineContinuityEquation(Region)
        self.DefineEnergyEquation(Region)

    def DefineContinuityEquation(self,Region):
        initCasePath=Region.caseDirectoryPath + os.sep+'0'
        if not os.path.isfile(initCasePath+ os.sep+'p'):
            print('A file of the name p  doesn''t exist in ', initCasePath)               
        else:
            self.equations['p']=equation.Equation('p')
            # self.equations['phi']=equation.Equation('phi')
            if not Region.STEADY_STATE_RUN:
                self.equations['p'].setTerms(['Transient', 'massDivergenceTerm'])
            else:
                self.equations['p'].setTerms(['massDivergenceTerm'])
            Region.fluid['pprime']=field.Field(Region,'pprime','volScalarField')
            Region.fluid['pprime'].boundaryPatchRef=Region.fluid['p'].boundaryPatchRef
            for iBPatch, values in Region.fluid['pprime'].boundaryPatchRef.items():
                if values['type']=='fixedValue':
                    Region.fluid['pprime'].boundaryPatchRef[iBPatch]['value']=[0.0]
            Region.fluid['p'].phiGrad=grad.Gradient(Region,'p')
            Region.fluid['pprime'].phiGrad=grad.Gradient(Region,'pprime')
            for i in range(3): 
                self.DefineDUfield(Region,'DU',i)
                self.DefineDUfield(Region,'DUT',i)
            Region.fluid['DUSf']=field.Field(Region,'DUSf','surfaceVectorField')
            Region.fluid['DUEf']=field.Field(Region,'DUEf','surfaceVectorField')

    def DefineEnergyEquation(self,Region):
        initCasePath=Region.caseDirectoryPath + os.sep+'0'
        if not os.path.isfile(initCasePath+ os.sep+'T'):
            print('A file of the name T  doesn''t exist in ', initCasePath)               
        else:
            self.equations['T']=equation.Equation('T')
            try:
                self.equations['T'].gamma=Region.fluid['kappa'].phi
            except AttributeError:
                self.equations['T'].gamma=Region.fluid['k'].phi/Region.fluid['Cp'].phi
                print("self.equations['T'].gamma information doesn't exist in the FoamDictionaries object")
            
            if not Region.STEADY_STATE_RUN:
                self.equations['T'].setTerms(['Transient', 'Convection','Diffusion'])
            else:
                self.equations['T'].setTerms(['Convection','Diffusion'])

            Region.fluid['T'].phiGrad=grad.Gradient(Region,'T')

    def DefineMomentumEquation(self,Region):
        initCasePath=Region.caseDirectoryPath + os.sep+'0'
        if not os.path.isfile(initCasePath+ os.sep+'U'):
            print('A file of the name U doesn''t exist in ', initCasePath)               
        else:
            self.equations['U']=equation.Equation('U')
            self.equations['U'].gamma=Region.fluid['mu'].phi
            if not Region.STEADY_STATE_RUN:
                self.equations['U'].setTerms(['Transient', 'Convection','Diffusion'])
            else:
                self.equations['U'].setTerms(['Convection','Diffusion'])

            if os.path.isfile(initCasePath+ os.sep+'p'):
                self.equations['U'].terms.append('PressureGradient')

            try:
                Region.dictionaries.g
                self.equations['U'].terms.append('Buoyancy')
            except AttributeError:
                print("Gravity information doesn't exist in the FoamDictionaries object")
            #Define mdot_f field
            self.DefineMdot_f(Region)
            Region.fluid['U'].phiGrad=grad.Gradient(Region,'U')
            
                
    def DefineDUfield(self,Region,name,iComponent):
        Name=name+str(iComponent)
        Region.fluid[Name]=field.Field(Region,Name,'volScalarField')
        # Region.fluid[Name].dimensions=[0,0,0,0,0,0,0]
        # Region.fluid['DU'+iComponent].initializeMdotFromU(Region)

    def DefineMdot_f(self,Region):
        Region.fluid['mdot_f']=field.Field(Region,'mdot_f','surfaceScalarField')
        Region.fluid['mdot_f'].dimensions=[-2,1,-1,0,0,0,0]
        Region.fluid['mdot_f'].boundaryPatchRef=Region.fluid['U'].boundaryPatchRef
        Region.fluid['mdot_f'].initializeMdotFromU(Region)
        
            


                 
