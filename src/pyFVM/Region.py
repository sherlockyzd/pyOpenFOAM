import numpy as np
import cfdtool.IO as io
import pyFVM.FoamDictionaries as fd
import pyFVM.Polymesh as pm
import pyFVM.Coefficients as coefficients
import pyFVM.Fluxes as fluxes
import cfdtool.Time as time
import pyFVM.Assemble as assemble
import pyFVM.Model as model
import cfdtool.Solve as solve
import cfdtool.cfdPlot as plt
import config as cfg


class Region():
    """Sets up the simulation's 'Region'.
    An instance of the Region class is required for the case to run. 
    The instance is created at the beginning of each case and is used to hold
    other class instances, such as 'polyMesh', any number of 'fluid' instances
    and a number of other attributes required for the simulation to run.
    All information related to the mesh's topology (i.e., distances of cell centers to wall, face surface areas, face normals and cell volumes) is available in the Region class. 
    """
    def __init__(self,casePath):
        self.cfdIsCompressible=cfg.cfdIsCompressible
        self.pp_nonlinear_corrected=cfg.pp_nonlinear_corrected
        self.caseDirectoryPath = casePath
        self.StartSession()
        self.ReadOpenFoamFiles()
        self.model=model.Model(self)

    def RunCase(self):
        """
        Runs the case by performing the necessary computations and iterations for each equation in the model.
        """
        io.cfdPrintHeader()
        ## Instance of Coefficients class which contains information related to the connectivity of the mesh.定义ac，anb
        self.coefficients=coefficients.Coefficients(self)
        ## Instance of Fluxes class which contains flux information  
        self.fluxes=fluxes.Fluxes(self)
        self.time = time.Time(self)
        self.assembledPhi = {} 
        for iTerm in self.model.equations:
            self.assembledPhi[iTerm]=assemble.Assemble(self,iTerm)
            self.fluid[iTerm].cfdfieldUpdate(self)
        for iTerm in self.fluid:
            self.fluid[iTerm].setPreviousTimeStep()    
        
        while(self.time.cfdDoTransientLoop()):
            #manage time
            self.time.cfdPrintCurrentTime()
            for iTerm in self.model.equations:
                self.fluid[iTerm].setPreviousTimeStep()
                self.fluid[iTerm].cfdfieldUpdateGradient(self)
            for iter in range(self.time.maxIter):
                self.time.iterationCount += 1
                self.model.residuals['sumRes']=0.0
                for iTerm in self.model.equations:
                    #sub-loop
                    if iTerm=='U':
                        self.MomentumIteration()
                    elif iTerm=='p':
                        self.ContinuityIteration()
                    else:
                        self.ScalarTransportIteration(iTerm)
                if self.model.residuals['sumRes'] < 1e-5:
                    break
            # plt.cfdPlotRes(self.caseDirectoryPath,self.model.equations)
            plt.plotResidualHistory(self)
            self.time.cfdUpdateRunTime()
            if self.time.cfdDoWriteTime():            
                io.cfdWriteOpenFoamParaViewData(self)
                pass

    def MomentumIteration(self):
        numVector=self.fluid['U'].iComponent
        for iComponent in range(numVector):
            io.MomentumPrintIteration(iComponent)
            self.assembledPhi['U'].cfdAssembleEquation(self,iComponent)
            self.cfdSolveUpdateEquation('U',iComponent)

    def ContinuityIteration(self): 
        io.ContinuityPrintIteration()
        self.assembledPhi['p'].cfdAssembleEquation(self)
        self.cfdSolveUpdateEquation('p')
        self.cfdCorrectNSSystemFields()

    def ScalarTransportIteration(self,iTerm):
        io.ScalarTransportPrintIteration()
        self.assembledPhi[iTerm].cfdAssembleEquation(self)
        self.cfdSolveUpdateEquation(iTerm)

    def cfdSolveUpdateEquation(self,theEquationName,iComponent=-1):
        solve.cfdSolveEquation(self,theEquationName,iComponent)
        self.fluid[theEquationName].cfdCorrectField(self,iComponent)
        #记录每一个方程的时间残差和迭代次数
        self.updateResiduals(theEquationName,iComponent)
        if iComponent==-1 or iComponent==2:
            self.fluid[theEquationName].cfdUpdateScale(self)

    def cfdCorrectNSSystemFields(self):
        theNumberOfElements=self.mesh.numberOfElements
        self.fluid['pprime'].phi[:theNumberOfElements].value = self.coefficients.dphi[:,None]
        self.fluid['pprime'].cfdfieldUpdate(self)#更新pprime的梯度，来计算速度增量
        self.fluid['U'].cfdCorrectNSFields(self)
        self.cfdUpdateProperty()

    def StartSession(self):
        """Initiates the class instance with the caseDirectoryPath attribute 
        and adds the 'dictionaries' and 'fluid' dictionaries. Reminder - 
        __init__ functions are run automatically when a new class instance is 
        created. 
        """
        io.cfdPrintMainHeader()
        io.cfdInitDirectories(self.caseDirectoryPath)
        print('Working case directory is %s' % self.caseDirectoryPath)

    def ReadOpenFoamFiles(self):
        """
        Reads the OpenFOAM files and initializes the necessary dictionaries and variables.
        This method performs the following steps:
        1. Initializes the 'fluid' dictionary to hold fluid properties.
        2. Initializes the 'dictionaries' dictionary to hold information from various OpenFOAM dictionaries.
        3. Checks if the simulation is steady-state or transient based on the 'ddtSchemes' entry in 'fvSchemes'.
        4. Determines the time solver type based on the 'fvSolution' entries.
        5. Initializes the 'mesh' dictionary to hold FVM mesh information.
        6. Reads the transport properties and thermophysical properties from the dictionaries.
        7. Calculates the geometric length scale of the mesh.
        8. Reads the time directory from the dictionaries.
        Note: Some steps require the mesh information and are called after initializing the 'mesh' dictionary.
        """
        ## Dictionary to hold 'fluid' properties. We are considering changing this to a more meaningful name such as 'field' because often this dictionary is used to store field and field variables which are not necessarily fluids.  
        self.fluid={}
        ## Dictionary holding information contained within the various c++ dictionaries used in OpenFOAM. For example, the contents of the './system/controlDict' file can be retrieved by calling Region.dictionaries.controlDict which return the dictionary containing all the entries in controlDict. 
        self.dictionaries=fd.FoamDictionaries(self)

        if self.dictionaries.fvSchemes['ddtSchemes']['default']=='steadyState':
            self.STEADY_STATE_RUN = True
        else:
            self.STEADY_STATE_RUN = False

        for item in self.dictionaries.fvSolution:
            if item=='PISO':
                self.Timesolver = 'PISO'
            if item=='SIMPLE':
                self.Timesolver = 'SIMPLE'
            if item=='PIMPLE':
                self.Timesolver = 'PIMPLE'
                io.cfdError('PIMPLE solver not yet implemented')

        ## Dictionary containing all the information related to the FVM mesh. 
        self.mesh=pm.Polymesh(self)
        #cfdReadTransportProperties需要用到网格信息。因此滞后计算，另外由于更新p.cfdUpdateScale需要密度信息，因此需要提前计算！！！！！
        self.dictionaries.cfdReadTransportProperties(self)
        self.dictionaries.cfdReadThermophysicalProperties(self)

        """cfdGeometricLengthScale() and self.dictionaries.cfdReadTimeDirectory() require the mesh and therefore are not included in the __init__ function of FoamDictionaries and are instead called after the self.mesh=pm.Polymesh(self) line above."""
        self.dictionaries.cfdReadTimeDirectory(self)

    def updateResiduals(self,theEquationName,iComponent=-1):
        """
        Updates the residuals for the specified equation and component.
        This method updates the residuals dictionary in the model with the current residuals for the specified equation and component.
        """
        if theEquationName =='U':
            self.model.residuals[theEquationName]['residuals'][iComponent].append(self.assembledPhi[theEquationName].theEquation.finalResidual[iComponent])
            #记录单次迭代步的总误差
            self.model.residuals['sumRes']+=self.assembledPhi[theEquationName].theEquation.finalResidual[iComponent]
        else:
            self.model.residuals[theEquationName]['residuals'].append(self.assembledPhi[theEquationName].theEquation.finalResidual)
            self.model.residuals['sumRes']+=self.assembledPhi[theEquationName].theEquation.finalResidual
        self.model.residuals[theEquationName]['time'].append(self.time.currentTime)
        self.model.residuals[theEquationName]['iterations'].append(self.time.iterationCount)
        

    # def check_iteration_Convergence(self):
        



    def cfdUpdateProperty(self):
        if self.cfdIsCompressible:
            self.dictionaries.cfdUpdateCompressibleProperties(self)
            self.dictionaries.cfdUpdateTransportProperties(self)
            self.dictionaries.cfdUpdateThermophysicalProperties(self)
            self.dictionaries.cfdUpdateTurbulenceProperties(self)
            self.dictionaries.cfdUpdatePressureReference(self)
