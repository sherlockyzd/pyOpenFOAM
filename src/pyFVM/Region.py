import pyFVM.cfdfunction as cfun
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
        #预定义矩阵组装结构 acnb, ldu, coo, csr
        #本程序依据cfd原理原创的是'acnb'格式，openfoam使用的是'ldu'格式，可选：'acnb','ldu','coo','csr'。
        self.MatrixFormat = 'acnb' 
        # 控制稀疏矩阵使用策略                                     
        # False: 只有压力泊松方程使用稀疏矩阵（原有行为）             
        # True: 所有方程都使用稀疏矩阵（新的优化选项）        
        self.sparse_always = True
        self.cfdIsCompressible=cfg.cfdIsCompressible
        self.pp_nonlinear_corrected=cfg.pp_nonlinear_corrected
        self.caseDirectoryPath = casePath
        self.StartSession()

    def readCase(self):
        self.ReadOpenFoamFiles()
        self.model=model.Model(self)

    def RunCase(self):
        """
        Runs the case by performing the necessary computations and iterations for each equation in the model.
        """
        self.readCase()
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
            self.SolveEquations() #根据SIMPLE或PISO算法求解方程
            self.cfdUpdate()
            plt.plotResidualHistory(self)
            self.time.cfdUpdateRunTime()
            if self.time.cfdDoWriteTime():            
                io.cfdWriteOpenFoamParaViewData(self)
                pass

    def cfdUpdate(self):
        for iTerm in self.model.equations:
            self.fluid[iTerm].setPreviousTimeStep()
            self.fluid[iTerm].cfdfieldUpdateGradient(self)            

    def SolveEquations(self):
        """
        Solves the equations for the current time step.
        This method performs the following steps:
        1. Iterates through the momentum equations for each component.
        2. Iterates through the continuity equation.
        3. Iterates through the scalar transport equation.
        """
        if self.Timesolver == 'PISO':
            self.PISOIteration()
        elif self.Timesolver == 'SIMPLE':
            self.SIMPLEIteration()
        else:
            io.cfdError('Unknown time solver type: %s' % self.Timesolver)

    def SIMPLEIteration(self):
        """
        Performs the SIMPLE iteration for the momentum and continuity equations.
        This method iterates through the momentum equations for each component,
        then iterates through the continuity equation, and finally updates the
        mass flow rate field.
        """
        for iter in range(self.time.maxIter):
            self.time.iterationCount += 1
            self.model.residuals['sumRes']=0.0
            self.MomentumIteration()
            self.ContinuityIteration()
            self.ScalarTransportIteration()
            if self.model.residuals['sumRes'] < 1e-5:
                break
        print('迭代 %d 次后完成收敛' % iter)
        
    def PISOIteration(self):
        self.model.residuals['sumRes']=0.0
        self.MomentumIteration()
        for iter in range(self.time.maxIter):
            self.time.iterationCount += 1
            self.ContinuityIteration()
            self.ScalarTransportIteration()
            if self.model.residuals['sumRes'] < 1e-5:
                break
            self.model.residuals['sumRes']=0.0
        print('迭代 %d 次后完成收敛' % iter)

    def MomentumIteration(self):
        """        Performs the momentum iteration for the momentum equations.
        This method iterates through the momentum equations for each component,
        assembles the equations, and updates the equations for each component.  
        """
        if 'U' not in self.model.equations:
            return
        numVector=self.fluid['U'].iComponent
        for iComponent in range(numVector):
            io.MomentumPrintIteration(iComponent)
            self.assembledPhi['U'].cfdAssembleEquation(self,iComponent)
            self.cfdSolveUpdateEquation('U',iComponent)
        self.cfdUpdateMassFlowRate()

    def ContinuityIteration(self): 
        if 'p' not in self.model.equations:
            return
        io.ContinuityPrintIteration()
        self.assembledPhi['p'].cfdAssembleEquation(self)
        self.cfdSolveUpdateEquation('p')
        self.cfdCorrectNSSystemFields()

    def ScalarTransportIteration(self):
        io.ScalarTransportPrintIteration()
        for iTerm in self.model.equations:
            if iTerm == 'U' or iTerm == 'p':
                continue
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
        self.fluid['pprime'].phi.value[:theNumberOfElements] = self.coefficients.dphi[:,None]
        self.fluid['pprime'].cfdfieldUpdate(self)
        self.fluid['pprime'].cfdfieldUpdateGradient(self)
        self.fluid['U'].cfdCorrectNSFields(self)
        self.cfdUpdateMassFlowRate()
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
        if iComponent==-1 or iComponent==2:
            self.model.residuals[theEquationName]['time'].append(self.time.currentTime)
            self.model.residuals[theEquationName]['iterations'].append(self.time.iterationCount)
        
    def cfdUpdateMassFlowRate(self):
        """
        Updates the mass flow rate field 'mdot_f' based on the current velocity field 'U' and density field 'rho'.
        This method interpolates the velocity and density fields from the cell centers to the faces and calculates the mass flow rate.
        """
        if 'U' in self.fluid and 'rho' in self.fluid:
            # Interpolate velocity and density from cell centers to faces
            cfun.initializeMdotFromU(self)


    def cfdUpdateProperty(self):
        if self.cfdIsCompressible:
            self.dictionaries.cfdUpdateCompressibleProperties(self)
            self.dictionaries.cfdUpdateTransportProperties(self)
            self.dictionaries.cfdUpdateThermophysicalProperties(self)
            self.dictionaries.cfdUpdateTurbulenceProperties(self)
            self.dictionaries.cfdUpdatePressureReference(self)
