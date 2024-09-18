import os
import pyFVM.IO as io
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
        self.rhophi_exists=False
        # Create theEquation structure
        self.DefineMomentumEquation(Region)
        self.DefineContinuityEquation(Region)
        self.DefineEnergyEquation(Region)
        self.DefineScalarTransportEquation(Region)

    def DefineContinuityEquation(self,Region):
        initCasePath=Region.caseDirectoryPath + os.sep+'0'
        if not os.path.isfile(initCasePath+ os.sep+'p'):
            print('A file of the name p  doesn''t exist in ', initCasePath)               
        else:
            self.equations['p']=equation.Equation('p')
            self.rhophi_exists=True
            self.equations['p'].setTerms(['massDivergenceTerm'])
            if not Region.STEADY_STATE_RUN:
                self.equations['p'].terms.append('Transient')
            # if not Region.STEADY_STATE_RUN:
            #     self.equations['p'].setTerms(['Transient', 'massDivergenceTerm'])
            # else:
            #     self.equations['p'].setTerms(['massDivergenceTerm'])
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
            self.equations['T'].setTerms([])
            if not Region.STEADY_STATE_RUN:
                self.equations['T'].terms.append('Transient')
                self.rhophi_exists=True
            for iterm in Region.dictionaries.fvSchemes['laplacianSchemes']:
                if io.contains_term(iterm,'T'):
                    self.equations['T'].terms.append('Diffusion')
                    parts=io.term_split(iterm)
                    terms_to_remove = ['laplacian', 'T']
                    str_gammas=io.remove_terms(parts,terms_to_remove)[0]
                    try:
                        self.equations['T'].gamma=Region.fluid[str_gammas].phi
                    except AttributeError:
                        self.equations['T'].gamma=Region.fluid['k'].phi/Region.fluid['Cp'].phi
                        print("self.equations['T'].gamma information doesn't exist in the FoamDictionaries object")
            
            for iterm in Region.dictionaries.fvSchemes['divSchemes']:
                if io.contains_term(iterm,'T'):
                    self.equations['T'].terms.append('Convection')
                    self.rhophi_exists=True  

            Region.fluid['T'].phiGrad=grad.Gradient(Region,'T')

    def DefineMomentumEquation(self,Region):
        initCasePath=Region.caseDirectoryPath + os.sep+'0'
        if not os.path.isfile(initCasePath+ os.sep+'U'):
            print('A file of the name U doesn''t exist in ', initCasePath)               
        else:
            self.equations['U']=equation.Equation('U')
            self.equations['U'].gamma=Region.fluid['mu'].phi
            self.equations['U'].setTerms([])
            if not Region.STEADY_STATE_RUN:
                self.equations['U'].terms.append('Transient')
                self.rhophi_exists=True
                # self.equations['U'].setTerms(['Transient', 'Convection','Diffusion'])
            # else:
            #     self.equations['U'].setTerms(['Convection','Diffusion'])
            for iterm in Region.dictionaries.fvSchemes['divSchemes']:
                if io.contains_term(iterm,'U'):
                    self.equations['U'].terms.append('Convection')
                    self.rhophi_exists=True
            
            for iterm in Region.dictionaries.fvSchemes['laplacianSchemes']:
                if io.contains_term(iterm,'default'):
                    self.equations['U'].terms.append('Diffusion')
                #     parts=io.term_split(iterm)
                #     terms_to_remove = ['laplacian', 'U']
                #     str_gammas=io.remove_terms(parts,terms_to_remove)
                # try:
                #     self.equations['U'].gamma=Region.fluid[str_gammas].phi
                # except AttributeError:
                #     self.equations['U'].gamma=Region.fluid['k'].phi/Region.fluid['Cp'].phi
                #     print("self.equations['U'].gamma information doesn't exist in the FoamDictionaries object")

            if os.path.isfile(initCasePath+ os.sep+'p'):
                self.equations['U'].terms.append('PressureGradient')

            try:
                Region.dictionaries.g
                self.equations['U'].terms.append('Buoyancy')
                self.rhophi_exists=True
            except AttributeError:
                print("Gravity information doesn't exist in the FoamDictionaries object")
            #Define mdot_f field
            self.DefineMdot_f(Region)
            Region.fluid['U'].phiGrad=grad.Gradient(Region,'U')
            
    def DefineScalarTransportEquation(self,Region):
        pass

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
        
            


                 
