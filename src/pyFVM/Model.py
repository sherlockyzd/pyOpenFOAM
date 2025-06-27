import os
# from numpy import array
import cfdtool.IO as io
import pyFVM.Field as field
import pyFVM.Gradient as grad
import pyFVM.Equation as equation
import pyFVM.cfdfunction as cfun
import cfdtool.dimensions as dm

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
        self.DefineResiduals(Region)

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
            Region.fluid['pprime']=field.Field(Region,'pprime','volScalarField',dm.pressure_dim)
            Region.fluid['pprime'].boundaryPatchRef=Region.fluid['p'].boundaryPatchRef
            for iBPatch, values in Region.fluid['pprime'].boundaryPatchRef.items():
                if values['type']=='fixedValue':
                    Region.fluid['pprime'].boundaryPatchRef[iBPatch]['value']=[0.0]
            Region.fluid['p'].Grad=grad.Gradient(Region,'p')
            Region.fluid['pprime'].Grad=grad.Gradient(Region,'pprime')
            self.equations['p'].CoffeDim=dm.length_dim*dm.time_dim
            self.DefineDUfield(Region,'DU')
            # self.DefineDUfield(Region,'DUT')
            self.DefineDUSffield(Region,'DUSf')
            self.DefineDUSffield(Region,'DUEf')

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
                        self.equations['T'].gamma=Region.fluid[str_gammas].phi*Region.fluid['rho'].phi
                    except AttributeError:
                        self.equations['T'].gamma=Region.fluid['k'].phi/Region.fluid['Cp'].phi*Region.fluid['rho'].phi
                        print("self.equations['T'].gamma information doesn't exist in the FoamDictionaries object")
            
            for iterm in Region.dictionaries.fvSchemes['divSchemes']:
                if io.contains_term(iterm,'T'):
                    self.equations['T'].terms.append('Convection')
                    self.rhophi_exists=True  

            Region.fluid['T'].Grad=grad.Gradient(Region,'T')
            self.equations['T'].CoffeDim=dm.flux_dim

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
            Region.fluid['U'].Grad=grad.Gradient(Region,'U')
            self.equations['U'].CoffeDim=dm.flux_dim

    def DefineScalarTransportEquation(self,Region):
        pass

    def DefineDUfield(self,Region,Name,*args):
        Dp_dim=Region.fluid['p'].Grad.phi.dimension.copy()
        DU_dim=dm.velocity_dim/Dp_dim
        Region.fluid[Name]=field.Field(Region,Name,'volVectorField',DU_dim)
        # Region.fluid[Name].dimensions=[-1,3,1,0,0,0,0]
        # Region.fluid[Name].boundaryPatchRef=Region.fluid['U'].boundaryPatchRef


    def DefineDUSffield(self,Region,Name,*args):
        DU_dim=Region.fluid['DU'].phi.dimension.copy()
        DUSf_dim=DU_dim*dm.area_dim
        Region.fluid[Name]=field.Field(Region,Name,'surfaceVectorField',DUSf_dim)
        Region.fluid[Name].dimensions=[-1,5,1,0,0,0,0]
        # Region.fluid[Name].boundaryPatchRef=Region.fluid['U'].boundaryPatchRef


    def DefineMdot_f(self,Region):
        # dimensions=[-2,1,-1,0,0,0,0]
        Region.fluid['mdot_f']=field.Field(Region,'mdot_f','surfaceScalarField',dm.flux_dim)
        Region.fluid['mdot_f'].boundaryPatchRef=Region.fluid['U'].boundaryPatchRef
        cfun.initializeMdotFromU(Region)

    def DefineResiduals(self,*args):
        """
        Initializes the residuals for the equations in the model.
        This method creates a dictionary to hold the residuals for each equation.
        """
        self.residuals = {}
        self.residuals['sumRes']=float(0.0)
        for equation_name in self.equations:
            self.residuals[equation_name] = {}
            # Initialize the residuals for each equation
            self.residuals[equation_name].setdefault('time', [])
            self.residuals[equation_name].setdefault('residuals', [])
            self.residuals[equation_name].setdefault('iterations', [])

