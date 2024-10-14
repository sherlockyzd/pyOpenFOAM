import os
import sys
import cfdtool.IO as io
import pyFVM.Field as field
# import pyFVM.Math as mth
import numpy as np
# import pyFVM.Scalar as Scalar

class FoamDictionaries():

    """Functions to read and manipulate foam dictionaries.
    
    Each dictionary that is read is accessible by calling the attribute related to the dictionary in question. For example, the values contained in 'controlDict' are accessed from within the self.controlDict attribute.  
    这段Python代码定义了一个名为 `FoamDictionaries` 的类，它专门用于读取和操作OpenFOAM软件中的字典文件。OpenFOAM是一个用于计算流体动力学（CFD）的开源软件，它使用一系列的字典文件来控制模拟的各个方面。

    以下是对这个类的详细解释：

    1. **类定义**：
    - `class FoamDictionaries():` 定义了一个名为 `FoamDictionaries` 的类。

    2. **文档字符串**：
    - 类的文档字符串提供了类的简要描述，说明了类的用途：读取和操作Foam字典，并提供了如何访问特定字典的示例。

    3. **构造器**：
    - `def __init__(self, Region):` 定义了类的构造器，它接收一个参数 `Region`，这个参数预期是一个包含OpenFOAM案例区域信息的对象。

    4. **初始化操作**：
    - 构造器中调用了多个方法来读取OpenFOAM案例中的不同字典文件：
        - `self.cfdReadControlDictFile()`：读取控制字典（`controlDict`）。
        - `self.cfdReadFvSchemesFile()`：读取有限体积方案字典（`fvSchemes`）。
        - `self.cfdReadFvSolutionFile()`：读取有限体积解决方案字典（`fvSolution`）。
        - `self.cfdReadGravity()`：读取重力相关属性。
        - `self.cfdReadTurbulenceProperties()`：读取湍流属性。
        - `self.cfdGetFields()`：获取模拟中使用的场。

    5. **属性访问**：
    - 根据文档字符串中的描述，每个读取的字典都可以通过对应的类属性访问。例如，控制字典的内容可以通过 `self.controlDict` 访问。

    6. **方法实现**：
    - 虽然这段代码没有提供这些方法的具体实现，但它们定义在类中的其他部分，用于执行文件读取和解析操作。

    7. **类的作用**：
    - `FoamDictionaries` 类提供了一个集中的方式来管理和访问OpenFOAM案例的配置信息，使得对案例的设置和参数的修改变得更加方便。

    ### 注意事项：
    - 这个类预期与OpenFOAM案例一起使用，因此它的方法将依赖于OpenFOAM特定的文件结构和命名约定。
    - 类的实现细节（如文件读取和解析逻辑）没有在这段代码中给出，需要进一步的代码来完成。
    - 类的使用示例和具体操作可能需要结合OpenFOAM的文档和案例结构来理解。

    这个类是CFD案例设置和配置管理的一部分，通过提供对关键配置文件的访问和操作，有助于自动化和简化CFD模拟的设置过程。`FoamDictionaries`类通过提供读取OpenFOAM配置文件的方法，使得对案例的设置和参数的访问变得更加方便和结构化。
    """
    
    def __init__(self,Region):
        
        # self.Region=Region
        self.cfdReadControlDictFile(Region)
        self.cfdReadFvSchemesFile(Region)
        self.cfdReadFvSolutionFile(Region)
        self.cfdReadGravity(Region)
        self.cfdReadTurbulenceProperties(Region)
        self.cfdGetFields(Region)

        
    def cfdReadControlDictFile(self,Region):
        """Reads contents of controlDict file in ./system folder.
        这段Python代码定义了一个名为 `cfdReadControlDictFile` 的方法，它是 `FoamDictionaries` 类的一部分，用于读取OpenFOAM案例中 `./system` 文件夹下的 `controlDict` 文件。以下是对这个方法的中文详细解释：

        1. **方法定义**：
        - `def cfdReadControlDictFile(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **打印信息**：
        - `print('Reading controlDict file ...')` 打印一条消息，告知用户正在读取 `controlDict` 文件。

        3. **构造文件路径**：
        - `controlDictFileDirectory = r"%s/system/controlDict" % self.Region.caseDirectoryPath` 构造 `controlDict` 文件的完整路径。

        4. **文件读取操作**：
        - 使用 `try...except` 结构来处理可能发生的异常，尤其是文件未找到的情况。
        - `with open(controlDictFileDirectory, "r") as fpid:` 以只读模式打开 `controlDict` 文件，并将其文件对象赋值给 `fpid`。

        5. **初始化字典**：
        - `self.controlDict={}` 初始化一个空字典，用于存储从 `controlDict` 文件中读取的键值对。

        6. **逐行读取文件**：
        - `for linecount, tline in enumerate(fpid):` 遍历文件的每一行，并使用 `enumerate` 获取行号。

        7. **跳过空行和注释**：
        - 使用 `io.cfdSkipEmptyLines(tline)` 和 `io.cfdSkipMacroComments(tline)` 函数跳过空行和宏定义注释行。

        8. **跳过FoamFile字典**：
        - 如果行包含 `"FoamFile"`，则调用 `io.cfdReadCfdDictionary(fpid)` 读取随后的字典定义，然后继续下一行。

        9. **解析键值对**：
        - 如果行包含多个分割的元素（即 `len(tline.split()) > 1`），则调用 `io.cfdReadCfdDictionary(fpid, line=tline.split())` 尝试读取字典。

        10. **异常处理**：
            - `except FileNotFoundError:` 如果 `controlDict` 文件不存在，捕获 `FileNotFoundError` 异常，并打印错误消息。

        ### 注意事项：
        - 这段代码中使用的 `io` 模块中的函数（如 `cfdSkipEmptyLines`、`cfdSkipMacroComments`、`cfdReadCfdDictionary`）没有在代码中定义，它们可能是在类的其他部分或外部模块定义的。
        - `self.Region.caseDirectoryPath` 预期是 `Region` 对象的一个属性，包含OpenFOAM案例的根目录路径。
        - 错误处理仅针对文件未找到的情况，其他潜在的 I/O 错误未被捕获。

        `cfdReadControlDictFile` 方法的目的是读取OpenFOAM案例的控制字典文件，并将文件中的配置信息存储在 `self.controlDict` 字典中，以便后续使用和访问。
        """
        print('Reading controlDict file ...')
        
        controlDictFileDirectory = r"%s/system/controlDict" % Region.caseDirectoryPath
        
        try:
            with open(controlDictFileDirectory,"r") as fpid:

                ## Dictionary with keys and values read from the 'controlDict' file.
                self.controlDict={}
                for linecount, tline in enumerate(fpid):
                    
                    if not io.cfdSkipEmptyLines(tline):
                        continue
                    
                    if not io.cfdSkipMacroComments(tline):
                        continue
                    
                    if "FoamFile" in tline:
                        dictionary=io.cfdReadCfdDictionary(fpid)
                        continue
        
                    if len(tline.split()) > 1:
                        
                        self.controlDict=io.cfdReadCfdDictionary(fpid,line=tline.split())
                        
        except FileNotFoundError:
            print('"controlDict" file is not found!!!' )

    def cfdReadFvSchemesFile(self,Region):
        """Reads contents of fvSchemes file in system folder.
        这段Python代码定义了一个名为`cfdReadFvSchemesFile`的方法，它是`FoamDictionaries`类的一部分，用于读取OpenFOAM案例中`system`文件夹下的`fvSchemes`文件。以下是对这个方法的中文详细解释：

        1. **方法定义**：
        - `def cfdReadFvSchemesFile(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **打印信息**：
        - `print('Reading fvSchemes file ...')` 打印一条消息，告知用户正在读取`fvSchemes`文件。

        3. **构造文件路径**：
        - `fvSchemesFileDirectory = r"%s/system/fvSchemes" % self.Region.caseDirectoryPath` 构造`fvSchemes`文件的完整路径。

        4. **初始化字典**：
        - `self.fvSchemes={}` 初始化一个空字典，用于存储从`fvSchemes`文件中读取的不同方案设置。

        5. **文件读取操作**：
        - 使用`with open(fvSchemesFileDirectory, "r") as fpid:`以只读模式打开文件，并将其文件对象赋值给`fpid`。

        6. **逐行读取文件**：
        - 通过遍历文件对象`fpid`，逐行读取文件内容。

        7. **跳过空行和注释**：
        - 使用`io.cfdSkipEmptyLines(tline)`和`io.cfdSkipMacroComments(tline)`函数跳过空行和宏定义注释行。

        8. **跳过FoamFile字典**：
        - 如果行包含`"FoamFile"`，则调用`io.cfdReadCfdDictionary(fpid)`读取随后的字典定义，然后继续下一行。

        9. **解析特定方案设置**：
        - 根据行内容，使用`if`语句检查特定关键字（如`"ddtSchemes"`、`"gradSchemes"`等），并为每个找到的关键字调用`io.cfdReadCfdDictionary(fpid)`来读取相应的字典定义，存储到`self.fvSchemes`字典中。

        ### 注意事项：
        - `self.Region.caseDirectoryPath`预期是`Region`对象的一个属性，包含OpenFOAM案例的根目录路径。
        - 该方法只处理了`fvSchemes`文件中的特定几个关键字，如果`fvSchemes`文件中包含其他关键字，可能需要额外的`if`语句来处理。

        `cfdReadFvSchemesFile`方法的目的是读取OpenFOAM案例的有限体积方案文件，并将文件中的不同方案设置存储在`self.fvSchemes`字典中，便于后续使用和访问。这有助于理解和配置模拟中的数值方案，例如时间步长方案、梯度方案、散度方案、拉普拉斯方案和插值方案等。
        """

        print('Reading fvSchemes file ...')
        
        fvSchemesFileDirectory = r"%s/system/fvSchemes" % Region.caseDirectoryPath 

        ## Dictionary with keys and values read from the 'fvSchemes' file. 
        self.fvSchemes={}
        
        with open(fvSchemesFileDirectory,"r") as fpid:
            
            for linecount, tline in enumerate(fpid):
                
                if not io.cfdSkipEmptyLines(tline):
                    continue
                
                if not io.cfdSkipMacroComments(tline):
                    continue
                
                if "FoamFile" in tline:
                    dictionary=io.cfdReadCfdDictionary(fpid)
                    continue
    
                if "ddtSchemes" in tline:
                    self.fvSchemes['ddtSchemes']=io.cfdReadCfdDictionary(fpid)
                    continue
    
                if "gradSchemes" in tline:
                    self.fvSchemes['gradSchemes']=io.cfdReadCfdDictionary(fpid)
                    continue
                
                if "divSchemes" in tline:
                    self.fvSchemes['divSchemes']=io.cfdReadCfdDictionary(fpid)
                    continue
                
                if "laplacianSchemes" in tline:
                    self.fvSchemes['laplacianSchemes']=io.cfdReadCfdDictionary(fpid)
                    continue         
                
                if "interpolationSchemes" in tline:
                    self.fvSchemes['interpolationSchemes']=io.cfdReadCfdDictionary(fpid)
                    continue      
                
                if "snGradSchemes" in tline:
                    self.fvSchemes['snGradSchemes']=io.cfdReadCfdDictionary(fpid)
                    continue      
                
                if "fluxRequired" in tline:
                    self.fvSchemes['fluxRequired']=io.cfdReadCfdDictionary(fpid)
                    continue            
    
    def cfdReadFvSolutionFile(self,Region):
        """Reads contents of fvSolution file in system folder.
        这段Python代码定义了一个名为`cfdReadFvSolutionFile`的方法，它是`FoamDictionaries`类的一部分，用于读取OpenFOAM案例中`system`文件夹下的`fvSolution`文件。以下是对这个方法的中文详细解释：

        1. **方法定义**：
        - `def cfdReadFvSolutionFile(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **打印信息**：
        - `print('Reading fvSolution file ...')` 打印一条消息，告知用户正在读取`fvSolution`文件。

        3. **构造文件路径**：
        - `fvSolutionFileDirectory = r"%s/system/fvSolution" % self.Region.caseDirectoryPath` 构造`fvSolution`文件的完整路径。

        4. **初始化字典**：
        - `self.fvSolution={}` 初始化一个空字典，用于存储从`fvSolution`文件中读取的设置。

        5. **文件读取操作**：
        - 使用`with open(fvSolutionFileDirectory, "r") as fpid:`以只读模式打开文件，并将其文件对象赋值给`fpid`。

        6. **逐行读取文件**：
        - 通过遍历文件对象`fpid`，逐行读取文件内容。

        7. **跳过空行和注释**：
        - 使用`io.cfdSkipEmptyLines(tline)`和`io.cfdSkipMacroComments(tline)`函数跳过空行和宏定义注释行。

        8. **跳过FoamFile字典**：
        - 如果行包含`"FoamFile"`，则调用`io.cfdReadCfdDictionary(fpid)`读取随后的字典定义，然后继续下一行。

        9. **解析特定设置**：
        - 根据行内容，使用`if`语句检查特定关键字（如`"solvers"`、`"SIMPLE"`、`"relaxationFactors"`），并为每个找到的关键字调用`io.cfdReadCfdDictionary(fpid)`来读取相应的字典定义，存储到`self.fvSolution`字典中。

        10. **设置迭代次数**：
            - 如果读取到的`'solvers'`字典中不包含`'maxIter'`键，则为每个求解器设置默认的迭代次数`20`。

        ### 注意事项：
        - `self.Region.caseDirectoryPath`预期是`Region`对象的一个属性，包含OpenFOAM案例的根目录路径。
        - 该方法处理了`fvSolution`文件中的`'solvers'`、`'SIMPLE'`和`'relaxationFactors'`部分，如果文件中包含其他部分，可能需要额外的`if`语句来处理。

        `cfdReadFvSolutionFile`方法的目的是读取OpenFOAM案例的求解器配置文件，并将文件中的求解器设置、SIMPLE算法设置和松弛因子等信息存储在`self.fvSolution`字典中，便于后续使用和访问。这有助于理解和配置模拟中的求解器参数和算法行为。
        """        
        print('Reading fvSolution file ...')
        fvSolutionFileDirectory = r"%s/system/fvSolution" % Region.caseDirectoryPath 
        self.fvSolution={}
        with open(fvSolutionFileDirectory,"r") as fpid:
            
            for linecount, tline in enumerate(fpid):
                if not io.cfdSkipEmptyLines(tline):
                    continue
                if not io.cfdSkipMacroComments(tline):
                    continue
                if "FoamFile" in tline:
                    dictionary=io.cfdReadCfdDictionary(fpid)
                    continue
                if "solvers" in tline:
                    self.fvSolution['solvers']=io.cfdReadCfdDictionary(fpid)
                    for key in self.fvSolution['solvers']:
                        if key == '//':
                             continue      
                        if 'maxIter' in self.fvSolution['solvers'][key]:
                            continue
                        else:
                            self.fvSolution['solvers'][key]['maxIter']=20
                            continue
                if "SIMPLE" in tline:
                    self.fvSolution['SIMPLE']=io.cfdReadCfdDictionary(fpid)
                    continue
                if "PISO" in tline:
                    self.fvSolution['PISO']=io.cfdReadCfdDictionary(fpid)
                    continue
                if "relaxationFactors" in tline:
                    self.fvSolution['relaxationFactors']=io.cfdReadCfdDictionary(fpid)
                    continue
                
    def cfdReadGravity(self,Region):
        """Reads contents of g file in ./constant folder.
        这段Python代码定义了一个名为 `cfdReadGravity` 的方法，它是 `FoamDictionaries` 类的一部分，用于读取OpenFOAM案例中 `./constant` 文件夹下的 `g` 文件，该文件通常包含重力向量的定义。以下是对这个方法的中文详细解释：

        1. **方法定义**：
        - `def cfdReadGravity(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **打印信息**：
        - `print('Reading contents of g file in ./constant folder.')` 打印一条消息，告知用户正在读取 `./constant` 文件夹中的 `g` 文件。

        3. **构造文件路径**：
        - `gravityFilePath=self.Region.caseDirectoryPath + "/constant/g"` 构造 `g` 文件的完整路径。

        4. **检查文件是否存在**：
        - 使用 `os.path.isfile(gravityFilePath)` 检查 `g` 文件是否存在。如果文件不存在，则打印消息并使用 `pass` 跳过后续操作。

        5. **读取文件**：
        - 如果文件存在，打印 `print('Reading Gravity file ...')` 消息，然后调用 `io.cfdReadAllDictionaries(gravityFilePath)` 读取文件中的所有字典条目。

        6. **初始化变量**：
        - `dimensions=[]` 和 `value=[]` 分别初始化用于存储维度和重力值的列表。

        7. **解析维度和值**：
        - 遍历 `gravityDict['dimensions']` 列表，尝试将每个条目转换为浮点数并添加到 `dimensions` 列表。
        - 遍历 `gravityDict['value']` 列表，移除括号，尝试将每个条目转换为浮点数并添加到 `value` 列表。

        8. **创建重力字典**：
        - `self.g={}` 初始化一个字典，用于存储重力的维度和值。
        - `self.g['dimensions']=dimensions` 和 `self.g['value']=value` 将解析出的维度和值赋值给 `self.g` 字典。

        ### 注意事项：
        - 这段代码中使用的 `io` 模块中的函数 `cfdReadAllDictionaries` 没有在代码中定义，它可能是在类的其他部分或外部模块定义的。
        - `self.Region.caseDirectoryPath` 预期是 `Region` 对象的一个属性，包含OpenFOAM案例的根目录路径。
        - 错误处理仅在文件不存在时进行了简单的打印操作，其他潜在的I/O错误未被捕获。

        `cfdReadGravity` 方法的目的是读取OpenFOAM案例的重力定义文件，并将文件中的维度和重力值解析出来，存储在 `self.g` 字典中，便于后续使用和访问。这有助于理解和配置模拟中的重力效应。
        """    
        gravityFilePath=Region.caseDirectoryPath + os.sep+"constant"+os.sep+"g"
        if not os.path.isfile(gravityFilePath):
            print('\n\nNo g file found\n')
            pass
        else:
            print('Reading Gravity file ...')        
            gravityDict = io.cfdReadAllDictionaries(gravityFilePath)
            dimensions=[]
            for iEntry in gravityDict['dimensions']:
                try:
                    dimensions.append(float(iEntry))
                except ValueError:
                    pass
            value=[]
            for iEntry in gravityDict['value']:
                iEntry=iEntry.replace("(","")
                iEntry=iEntry.replace(")","")
                try:
                    value.append(float(iEntry))
                except ValueError:
                    pass
            self.g={}
            self.g['dimensions']=dimensions
            self.g['value']=value

    def cfdReadTurbulenceProperties(self,Region):
        """
        Reads the turbulenceProperties dictionary 
           and sets the turbulence properties in Region.foamDictionary
        If there is no turbulenceProperties file, sets the turbulence
           model to 'laminar'.
        这段Python代码定义了一个名为`cfdReadTurbulenceProperties`的方法，它是`FoamDictionaries`类的一部分，用于读取OpenFOAM案例中的湍流属性字典，并设置区域（Region）的湍流属性。以下是对这个方法的中文详细解释：

        1. **方法定义**：
        - `def cfdReadTurbulenceProperties(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **文档字符串**：
        - 方法的文档字符串说明了方法的功能：读取湍流属性字典，并在找不到文件时将湍流模型设置为‘laminar’（层流）。

        3. **初始化字典**：
        - `self.turbulenceProperties={}` 初始化一个空字典，用于存储湍流属性。

        4. **构造文件路径**：
        - `turbulencePropertiesFilePath=self.Region.caseDirectoryPath+"/constant/turbulenceProperties"` 构造湍流属性文件的完整路径。

        5. **检查文件路径**：
        - 如果`turbulencePropertiesFilePath`为空，则将湍流属性设置为关闭状态，并将湍流模型设置为‘laminar’。

        6. **打印信息**：
        - 如果文件路径存在，打印`print('Reading Turbulence Properties ...')`消息，告知用户正在读取湍流属性。

        7. **读取字典**：
        - 调用`io.cfdReadAllDictionaries(turbulencePropertiesFilePath)`读取湍流属性文件中的所有字典条目。

        8. **获取字典键**：
        - `turbulenceKeys = turbulencePropertiesDict.keys()` 获取湍流属性字典的所有键。

        9. **遍历字典**：
        - 通过两个嵌套的循环，外循环遍历`turbulenceKeys`，内循环遍历每个键对应的子字典。

        10. **过滤特定键**：
            - 如果键是`'FoamFile'`或`'simulationType'`，则跳过不处理。

        11. **设置湍流属性**：
            - 对于其他键，将子字典中的值赋值到`self.turbulenceProperties`字典中。

        ### 注意事项：
        - 这段代码中使用的`io`模块中的函数`cfdReadAllDictionaries`没有在代码中定义，它可能是在类的其他部分或外部模块定义的。
        - `self.Region.caseDirectoryPath`预期是`Region`对象的一个属性，包含OpenFOAM案例的根目录路径。
        - 如果`turbulenceProperties`文件不存在或路径为空，方法将默认设置湍流模型为关闭或层流状态。
        - 代码中的错误处理较为简单，如果文件存在但读取过程中出现问题，可能需要额外的错误捕获和处理逻辑。

        `cfdReadTurbulenceProperties`方法的目的是读取OpenFOAM案例的湍流属性配置，并将这些属性存储在`self.turbulenceProperties`字典中，便于后续使用和访问。这有助于理解和配置模拟中的湍流模型和行为。
        """
        self.turbulenceProperties={}
        turbulencePropertiesFilePath=Region.caseDirectoryPath+"/constant/turbulenceProperties"
        if not os.path.isfile(turbulencePropertiesFilePath):
            self.turbulenceProperties['turbulence'] = 'off'
            self.turbulenceProperties['RASModel'] = 'laminar'
        else:
            print('Reading Turbulence Properties ...')
            turbulencePropertiesDict = io.cfdReadAllDictionaries(turbulencePropertiesFilePath)
            turbulenceKeys = turbulencePropertiesDict.keys()
            for iDict in turbulenceKeys:
                if 'FoamFile' in iDict or 'simulationType' in iDict: 
                    pass
                else:
                    for iSubDict in turbulencePropertiesDict[iDict]:
                        self.turbulenceProperties[iSubDict]=turbulencePropertiesDict[iDict][iSubDict][0]

    def cfdGetFields(self,Region):
        """Gets field names from keys contained in Region.foamDictionary['fvSolution'].
        Attributes:
           fields (list): fields.
        """
        Region.fvSolutionfields=[]
        # 创建一个列表来存储需要删除的键
        keys_to_remove = []
        for key in self.fvSolution['solvers']:
            if key == '//'or not key.isidentifier():
                keys_to_remove.append(key)
            else:
                Region.fvSolutionfields.append(key)
        # 从 self.fvSolution['solvers'] 中删除不符合条件的键
        for key in keys_to_remove:
            del self.fvSolution['solvers'][key]
            
    def cfdReadTimeDirectory(self,Region):
        '''
        这段Python代码定义了一个名为`cfdReadTimeDirectory`的方法，它用于确定OpenFOAM案例的时间步目录，并读取该时间步目录下的所有场数据。以下是对这个方法的中文详细解释：

        1. **方法定义**：
        - `def cfdReadTimeDirectory(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **处理启动参数**：
        - 根据`self.controlDict['startFrom']`的值确定时间步目录：
            - 如果提供了`kwargs`参数，使用`kwargs['time']`作为时间步目录。
            - 如果`startFrom`是`'startTime'`，使用`controlDict`中定义的开始时间。
            - 如果`startFrom`是`'latestTime'`，调用`cfdGetTimeSteps`方法并选择最大时间步作为当前时间步目录。
            - 如果`startFrom`是`'firstTime'`，同样调用`cfdGetTimeSteps`方法并选择最小时间步作为当前时间步目录。

        3. **检查字段文件**：
        - 使用`os.walk`遍历确定的时间步目录，检查是否存在字段文件。

        4. **读取字段文件**：
        - 对于每个字段文件，使用`io.cfdGetFoamFileHeader`读取文件头部信息，并根据头部信息创建场对象。

        5. **处理边界条件**：
        - 遍历每个边界补丁，读取边界条件类型和值，根据边界条件类型设置场的边界值。

        6. **初始化场数据**：
        - 对于均匀初始化的场，直接将边界条件值赋给场数组。
        - 对于非均匀初始化的场，提示尚未实现相关功能。

        7. **异常处理**：
        - 在处理过程中，如果遇到`KeyError`或`ValueError`，打印错误消息并根据错误类型决定是否继续执行。

        8. **删除局部变量**：
        - 在边界条件处理的最后，删除一些局部变量以释放内存。

        ### 注意事项：
        - 这段代码中使用的`io`模块中的函数（如`cfdGetFoamFileHeader`、`cfdGetKeyValue`、`cfdReadAllDictionaries`等）没有在代码中定义，它们可能是在类的其他部分或外部模块定义的。
        - `self.Region`预期是一个包含OpenFOAM案例信息的对象，具有`caseDirectoryPath`、`timeSteps`、`mesh`等属性。
        - 方法中处理了均匀（uniform）和非均匀（nonuniform）边界条件，但非均匀边界条件的处理提示尚未实现。
        - 代码中存在一些潜在的逻辑问题，例如在处理边界条件时，对于`volScalarField`和`surfaceScalarField`的处理似乎有重叠，且在某些情况下可能不会正确设置边界值。

        `cfdReadTimeDirectory`方法的目的是确定OpenFOAM案例的当前时间步目录，并读取该目录下所有场的数据，包括内部场值和边界条件。这有助于CFD模拟的初始化和数据管理。
        '''
        kwargs=[]  
        if len(kwargs) > 0:
            Region.timeDirectory=kwargs['time']
            
        elif self.controlDict['startFrom']=='startTime':
            Region.timeDirectory=str(int(self.controlDict['startTime']))
            
            
        elif self.controlDict['startFrom']=='latestTime':
            self.cfdGetTimeSteps(Region)
            Region.timeDirectory=max(Region.timeDictionary)
            
        elif self.controlDict['startFrom']=='firstTime':   
            ## I think in this case, the timeDirectory should be the minimum in 
            ## the list of time directories in the working folder (analogous to
            ## the latestTime case)
            self.cfdGetTimeSteps(Region)
            Region.timeDirectory=min(self.timeDictionary)         
            
        else:
            print("Error in controlDict: startFrom is not valid!")
        
        # Ensure field files are present in the time directory
        field_dir = os.path.join(Region.caseDirectoryPath, str(Region.timeDirectory))
        for root, directory,files in os.walk(field_dir):
            if not files:
                io.cfdError('Fields are not found in the %s directory' % (Region.caseDirectoryPath + os.sep +Region.timeDirectory+"!"))

        theNumberOfElements = Region.mesh.numberOfElements                       
        
        # 假设 files 是原始字段列表
        # files = ['U', 'P', 'T']
        # 对列表进行排序，确保 'U' 字段在最前面
        priority_fields = ['U','rho']
        files_with_priority = [field for field in priority_fields if field in files]
        remaining_fields = [field for field in files if field not in priority_fields]
        # 合并列表，'U' 字段在前，其余字段在后
        files = files_with_priority + remaining_fields

        for fieldName in files:
            fieldFilePath=os.path.join(Region.caseDirectoryPath, Region.timeDirectory, fieldName)
            header=io.cfdGetFoamFileHeader(fieldFilePath)
            dimensions=np.int8(io.cfdGetKeyValue('dimensions','dimensions',fieldFilePath)[2])
            if fieldName=='p':#OpenFoam里压强单位除以了密度，这里要乘回来
                dimensions[0]+=1
                dimensions[1]-=3
            Region.fluid[fieldName]=field.Field(Region,fieldName,header['class'],dimensions)                      
            internalField = io.cfdGetKeyValue('internalField','string',fieldFilePath)
            valueType=internalField[1]
            
            if Region.fluid[fieldName].type=='surfaceScalarField':
                io.cfdError('surfaceScalarFields are not yet handled.')

            # Vectorized initialization for uniform fields
            if valueType == 'uniform':
                value_str = internalField[2]
                if Region.fluid[fieldName].type=='volScalarField':
                    Region.fluid[fieldName].phi.value[:theNumberOfElements] = value_str
                        
                elif Region.fluid[fieldName].type=='volVectorField':
                    Region.fluid[fieldName].phi.value[:theNumberOfElements] = np.array(value_str)

            elif valueType == 'nonuniform':
                io.cfdError('The function cfdReadNonuniformList() is not yet writen.')
            # del(valueType)#防止内部的值对以下边界的误判！！！
                
            for iBPatch, values in Region.mesh.cfdBoundaryPatchesArray.items():          
                iElementStart = values['iBElements'][0]
                iElementEnd = values['iBElements'][-1]
                owners_b=values['owners_b']

                boundaryFile = io.cfdReadAllDictionaries(fieldFilePath)
                boundaryType = boundaryFile['boundaryField'][iBPatch]['type'][0]

                boundaryValueDict = boundaryFile['boundaryField'][iBPatch].get('value', None)
                if boundaryValueDict is not None:
                    valueType, boundaryValue = io.cfdReadUniformVolVectorFieldValue(boundaryValueDict)
                    if valueType == 'uniform':
                        if Region.fluid[fieldName].type in ['volScalarField', 'surfaceScalarField']:
                            Region.fluid[fieldName].phi.value[iElementStart:iElementEnd+1] = boundaryValue
                        elif Region.fluid[fieldName].type in ['volVectorField', 'surfaceVectorField']:
                            Region.fluid[fieldName].phi.value[iElementStart:iElementEnd+1] = np.array(boundaryValue)
                    else:
                        io.cfdError("Error: Oops, code cannot yet handle nonuniform boundary conditions. Not continuing any further ... apply uniform b.c.'s to continue")
                else:
                    valueType = None
                    # Handle missing 'value' entry based on boundary type
                    if boundaryType in ['noSlip', 'empty']:
                        boundaryValue = 0 if Region.fluid[fieldName].type in ['volScalarField', 'surfaceScalarField'] else [0, 0, 0]
                    elif boundaryType == 'zeroGradient':
                        boundaryValue = Region.fluid[fieldName].phi.value[owners_b]
                  
                    else:
                        io.cfdError(f"Warning: {fieldName} field's {iBPatch} boundary lacks 'value' entry.")
                    if Region.fluid[fieldName].type in ['volScalarField', 'surfaceScalarField']:
                        Region.fluid[fieldName].phi.value[iElementStart:iElementEnd+1] = boundaryValue
                    elif Region.fluid[fieldName].type in ['volVectorField', 'surfaceVectorField']:
                        Region.fluid[fieldName].phi.value[iElementStart:iElementEnd+1] = np.array(boundaryValue)


                Region.fluid[fieldName].boundaryPatchRef[iBPatch] = {
                'type': boundaryType,
                'valueType': valueType,
                'value': boundaryValue
                }
                # try:
                #     boundaryValueDict = boundaryFile['boundaryField'][iBPatch]['value']
                #     valueType, boundaryValue = io.cfdReadUniformVolVectorFieldValue(boundaryValueDict)
                #     '''
                #     这段代码是一个异常处理的例子，用于处理在读取和解析边界条件值时可能发生的错误。以下是对这段代码的详细解释：
                #     1. **尝试块** (`try`):
                #     - `boundaryValue = boundaryFile['boundaryField'][iBPatch]['value']`: 尝试从`boundaryFile`字典中获取与`iBPatch`边界补丁相关的`'value'`条目。
                #     - `valueType, boundaryValue = io.cfdReadUniformVolVectorFieldValue(boundaryValue)`: 尝试读取边界值，并确定其类型（例如，是否是均匀场或非均匀场）。

                #     2. **KeyError 异常处理**:
                #     - 如果在尝试访问`'value'`条目时发生了`KeyError`（即字典中不存在该键），则执行`except KeyError`块中的代码。
                #     - 根据`boundaryType`的值，为标量场或矢量场设置默认的边界值。如果边界类型是`'zeroGradient'`或`'empty'`，则分别将边界值设置为0（标量场）或`[0, 0, 0]`（矢量场）。
                #     - 如果边界类型不是上述两种之一，打印一条警告消息，指出该字段的边界条件缺少`'value'`条目。

                #     3. **ValueError 异常处理**:
                #     - 如果在尝试解析边界值时发生了`ValueError`（例如，当值不是预期的格式或类型时），则执行`except ValueError`块中的代码。
                #     - 打印一条错误消息，指出代码目前还无法处理非均匀边界条件，并建议应用均匀边界条件以继续执行。

                #     4. **循环中断** (`break`):
                #     - 在`except ValueError`块中，使用`break`语句中断当前循环。这意味着如果遇到无法处理的非均匀边界条件，将不再处理更多的边界补丁，而是退出循环。

                #     这段代码的目的是确保在读取和设置边界条件时，能够妥善处理缺失或格式不正确的边界值，并在必要时提供清晰的错误信息。
                #     ''' 
                # except KeyError:
                #     if boundaryType == 'noSlip' or boundaryType == 'empty' : 
                #         if Region.fluid[fieldName].type=='volScalarField' or Region.fluid[fieldName].type=='surfaceScalarField':
                #             boundaryValue = 0                        
                #         elif Region.fluid[fieldName].type=='volVectorField':
                #             boundaryValue = [0,0,0]
                #     elif boundaryType == 'zeroGradient': 
                #         boundaryValue=[]
                #         if Region.fluid[fieldName].type=='volScalarField' or Region.fluid[fieldName].type=='surfaceScalarField':

                #             for index, val in enumerate(range(iElementStart, iElementEnd+1)):
                #                 # print(index, val)
                #                 Region.fluid[fieldName].phi.value[val]=Region.fluid[fieldName].phi.value[owners_b[index]]
                #                 boundaryValue.append(Region.fluid[fieldName].phi.value[val])


                #         elif Region.fluid[fieldName].type=='volVectorField':
                #             for index, val in enumerate(range(iElementStart, iElementEnd+1)):
                #                 # print(index, val)
                #                 Region.fluid[fieldName].phi.value[val]=Region.fluid[fieldName].phi.value[owners_b[index]]
                #                 boundaryValue.append(Region.fluid[fieldName].phi.value[val])
                #     else:
                #         io.cfdError('Warning: The %s field\'s %s boundary does not have a \'value\' entry' %(fieldName, iBPatch))
                #         # break
                        
                # except ValueError:                
                #         io.cfdError("Error: Oops, code cannot yet handle nonuniform boundary conditions. Not continuing any further ... apply uniform b.c.'s to continue")             
    
                # try:
                #     if valueType == 'uniform':
                #         if Region.fluid[fieldName].type=='volScalarField' or Region.fluid[fieldName].type=='surfaceScalarField':
                #             for count in range(iElementStart,iElementEnd+1):
                #                 Region.fluid[fieldName].phi.value[count]=boundaryValue[0]        
                #         if Region.fluid[fieldName].type=='volVectorField':   
                #             for count in range(iElementStart,iElementEnd+1):
                #                 Region.fluid[fieldName].phi.value[count]=boundaryValue                   
                # except NameError:
                #     Region.fluid[fieldName].boundaryPatchRef[iBPatch]={}
                #     Region.fluid[fieldName].boundaryPatchRef[iBPatch]['type']=boundaryType
                #     del(boundaryType)
                #     continue
                # del(boundaryValue)
                # del(valueType)#防止上一个边界的值对一下个边界的误判！！！
                # del(boundaryType)   


    def cfdGetTimeSteps(self,Region):
        """Finds valid time directories in case directory.
        这段Python代码定义了一个名为 `cfdGetTimeSteps` 的方法，它是 `FoamDictionaries` 类的一部分，用于在OpenFOAM案例目录中查找所有有效的时间步目录。以下是对这个方法的中文详细解释：

        1. **方法定义**：
        - `def cfdGetTimeSteps(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **文档字符串**：
        - 方法的文档字符串说明了方法的功能：在案例目录中查找有效的时间目录。

        3. **打印信息**：
        - `print("Searching for time directories ... \n")` 打印一条消息，告知用户程序正在搜索时间目录。

        4. **初始化时间步列表**：
        - `self.Region.timeSteps=[]` 初始化一个空列表，用于存储找到的时间步目录的名称。

        5. **遍历案例目录**：
        - 使用 `os.walk(self.Region.caseDirectoryPath)` 遍历案例目录及其所有子目录。

        6. **检查目录**：
        - 通过 `for root, directory, files in os.walk(...)` 循环，遍历每个子目录的名称。

        7. **判断时间目录**：
        - 对于每个文件夹名称，使用 `self.cfdIsTimeDirectory(os.path.join(root, folder))` 判断它是否是一个时间目录。这里假设 `cfdIsTimeDirectory` 是一个辅助方法，用于确定给定路径是否为时间步目录。

        8. **处理目录名称**：
        - 如果文件夹名称可以转换为浮点数，并且存在小数位，则将其转换为字符串并添加到时间步列表。
        - 如果文件夹名称的小数部分为零（即整数），则将其转换为整数形式的字符串并添加到时间步列表。

        9. **打印换行**：
        - `print("\n")` 在完成时间步目录搜索后打印一个换行符。

        ### 注意事项：
        - 这段代码中使用的 `os` 模块是Python的标准库，用于操作系统功能，如文件路径操作。
        - `self.Region.caseDirectoryPath` 预期是 `Region` 对象的一个属性，包含OpenFOAM案例的根目录路径。
        - 代码中的 `cfdIsTimeDirectory` 方法没有给出定义，它应该是一个辅助方法，用于确定目录是否为时间步目录。
        - 代码假设时间步目录的名称可以直接转换为数值，并且根据是否存在小数位来决定是使用浮点数还是整数。

        `cfdGetTimeSteps` 方法的目的是遍历OpenFOAM案例目录，找出所有代表时间步的目录，并将它们的名称存储在 `self.Region.timeSteps` 列表中，便于后续使用和访问。这有助于管理和分析不同时间步的模拟数据。
        """
        print("Searching for time directories ... \n")
    
        Region.timeDictionary=[]
        for root, directory,files in os.walk(Region.caseDirectoryPath):
            
            for folder in directory:
                if self.cfdIsTimeDirectory(os.path.join(root, folder)):
                    #check for decimal place in folder name
                    if float(folder)-int(folder) != 0:
                        Region.timeDictionary.append(str(float(folder)))
                    elif float(folder)-int(folder) ==0:
                        Region.timeDictionary.append(str(int(folder)))

    def cfdIsTimeDirectory(self,theDirectoryPath):
        """Checks input directory if it is a valid time directory.
        这段Python代码定义了一个名为`cfdIsTimeDirectory`的方法，它是用于检查给定路径是否为有效的时间步目录的方法。以下是对这个方法的中文详细解释：
        1. **方法定义**：
        - `def cfdIsTimeDirectory(self, theDirectoryPath):` 定义了一个实例方法，接收一个参数`theDirectoryPath`，表示要检查的目录路径。

        2. **文档字符串**：
        - 方法的文档字符串简要说明了方法的功能：检查输入目录是否为有效的时间目录。

        3. **分解目录路径**：
        - `root, basename = os.path.split(theDirectoryPath)` 使用`os.path.split`函数分解目录路径，获取父目录路径`root`和基本目录名`basename`。

        4. **尝试将目录名转换为浮点数**：
        - `try:` 尝试块开始，尝试将目录名`basename`转换为浮点数`check`。如果转换成功，说明目录名可能是一个数值，这是时间步目录的一个特征。

        5. **检查目录中的文件**：
        - `for file in os.listdir(theDirectoryPath):` 遍历`theDirectoryPath`目录中的所有文件。

        6. **检查文件是否为场**：
        - `if str(file) in self.Region.fields:` 如果文件名（转换为字符串）存在于`self.Region.fields`列表中，则认为该文件是一个场文件。

        7. **确认时间目录**：
        - 如果找到场文件，打印消息确认`theDirectoryPath`是一个时间步目录，并返回`True`。

        8. **异常处理**：
        - `except ValueError:` 异常处理块。如果目录名无法转换为浮点数，捕获`ValueError`异常。

        9. **目录不是时间目录**：
        - 如果捕获到异常或没有找到场文件，打印消息说明`theDirectoryPath`不是一个时间步目录，并返回`False`。

        ### 注意事项：
        - 这个方法假设`self.Region.fields`是一个包含所有场名称的列表，用于验证目录中的文件是否与场相关。
        - 方法使用`os.listdir(theDirectoryPath)`获取目录中的文件列表。
        - 如果目录名可以转换为浮点数，这通常意味着它可能是一个时间步（例如，时间步编号或时间点）。
        - 方法中的打印语句用于提供反馈，告知用户哪些目录被识别为时间步目录，哪些被跳过。

        `cfdIsTimeDirectory`方法的目的是辅助识别OpenFOAM案例中的时间步目录，这对于CFD模拟的管理和分析非常重要。通过检查目录名是否为数值以及目录中是否存在场文件，可以确定该目录是否代表一个时间步。
        """
        #root, basename = os.path.split(theDirectoryPath) 使用os.path.split函数分解目录路径，获取父目录路径root和基本目录名basename。
        root, basename=os.path.split(theDirectoryPath)
        
        try:
            #if string, throw ValueError
            check=float(basename)
            #else
            for file in os.listdir(theDirectoryPath):    
                #check if file name is a field
                if str(file) in self.fvSolution['solvers']:
                    print("%s is a time directory" % theDirectoryPath)
                    return True
            
        except ValueError:
            
            print('%s is not a time directory, skipping ...' % theDirectoryPath)
            
            return False

    def cfdReadTransportProperties(self,Region):
        """Reads the transportProperties dictionary and sets the 
           transportProperties in Region.fluid If rho, mu and Cp dictionaries 
           are not user defined, creates them with default air properties
           Same for k (thermal conductivity) if the DT dictionary is present
        Attributes:
           Region (instance of cfdSetupRegion): the cfd Region.
        Example usage:
            cfdReadTransportProperties(Region)
        这段Python代码定义了一个名为`cfdReadTransportProperties`的方法，它是用于读取OpenFOAM案例中的`transportProperties`字典文件，并设置流体区域（`Region.fluid`）的输运属性。以下是对这个方法的详细解释：
        1. **方法定义**：
        - `def cfdReadTransportProperties(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **文件路径构造**：
        - `transportPropertiesFilePath` 构造了`transportProperties`文件的完整路径。

        3. **文件存在性检查**：
        - 使用`os.path.isfile`检查`transportProperties`文件是否存在，如果不存在，则不执行任何操作。

        4. **读取字典**：
        - 如果文件存在，调用`io.cfdReadAllDictionaries`函数读取文件中的所有字典条目。

        5. **初始化输运属性字典**：
        - `self.transportProperties={}` 初始化一个空字典，用于存储输运属性。

        6. **遍历字典键**：
        - 通过遍历`transportDicts`字典的键，对每个输运属性进行处理。

        7. **检查字典长度**：
        - 如果字典的长度不是8，打印错误消息并退出循环。

        8. **创建场对象**：
        - 对于每个输运属性，创建一个`Field`对象，并设置其维度和初始值。

        9. **设置边界条件**：
        - 为每个边界补丁设置默认的边界条件（`'zeroGradient'`），并赋予初始值。

        10. **更新比例尺**：
            - 调用`cfdUpdateScale`方法更新场的比例尺。

        11. **设置默认空气属性**：
            - 如果`'rho'`（密度）、`'mu'`（动力黏性系数）、`'Cp'`（比热容）不在字典键中，创建它们并赋予默认值。

        12. **计算热导率**：
            - 如果`'k'`（热导率）不在字典键中，但存在`'DT'`（温度梯度）键，根据`DT`、`Cp`和`rho`计算`k`。

        13. **设置边界条件和比例尺**：
            - 为默认创建的输运属性设置边界条件并更新比例尺。

        14. **设置压缩性标志**：
            - `self.Region.compressible='false'` 设置压缩性标志为`'false'`。

        ### 注意事项：
        - `self.Region`预期是一个包含OpenFOAM案例信息的对象，具有`caseDirectoryPath`、`fluid`、`mesh`等属性。
        - 方法中处理了默认空气属性的设置，如果`transportProperties`文件中没有定义相应的输运属性，则使用默认值。
        - 代码中使用`fill`方法来填充NumPy数组，这意味着所有单元格被赋予相同的初始值。
        `cfdReadTransportProperties`方法的目的是读取OpenFOAM案例的输运属性配置，并将这些属性存储在`self.transportProperties`字典中，同时初始化和设置流体区域的场对象，为CFD模拟的初始化和数据管理提供支持。
        """ 
        transportPropertiesFilePath=Region.caseDirectoryPath+"/constant/transportProperties"
                        
        if not os.path.isfile(transportPropertiesFilePath):
            pass
        else:
            print('\nReading transport properties ...')
            transportDicts=io.cfdReadAllDictionaries(transportPropertiesFilePath)
            transportKeys=list(transportDicts)   
            self.transportProperties={}
            for iKey in transportKeys:
                if iKey=='FoamFile' or iKey=='cfdTransportModel':
                    pass
                elif not len(transportDicts[iKey])==8:
                    #前七个是量纲“dimension”，第八个是具体的值，比如rho和mu的具体值
                    # print('FATAL: There is a problem with entry %s in transportProperties has no value' %iKey )
                    # sys.exit()
                    io.cfdError('FATAL: There is a problem with entry %s in transportProperties has no value' %iKey)
                else:
                    dimVector=[]
                    boundaryPatch={} 
                    self.transportProperties[iKey]={}
                    for iDim in transportDicts[iKey][0:7]:
                        dimVector.append(float(iDim))
                    keyValue = float(transportDicts[iKey][7])
                    Region.fluid[iKey]=field.Field(Region,iKey,'volScalarField',transportDicts[iKey][0:7])
                    # Region.fluid[iKey].dimensions=transportDicts[iKey][0:7]  
                    Region.fluid[iKey].phi.value.fill(keyValue)
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)
                    # for iPatch in range(0,numberOfBPatches):
                    boundaryPatch['value'] = keyValue
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch(iKey,boundaryPatch,Region)
                    # Region.fluid[iKey].boundaryPatch = boundaryPatch
                    self.transportProperties[iKey]['name']=iKey
                    self.transportProperties[iKey]['propertyValue']=keyValue
                    self.transportProperties[iKey]['dimensions']=transportDicts[iKey][0:7]
                    Region.fluid[iKey].cfdUpdateScale(Region)
                    # Scalar.cfdUpdateScale(self.fluid[iKey],self)
    
            # if 'rho' in transportKeys:
            #     boundaryPatch={}
            #     Region.fluid['rho']=field.Field(Region,'rho','volScalarField')
            #     Region.fluid['rho'].dimensions=[0., 0., 0., 0., 0., 0.,0.]
            #     Region.fluid['rho'].phi.fill(1.)
            #     # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)
            #     # for iPatch in range(0,numberOfBPatches):
            #     boundaryPatch['value'] = 1
            #     boundaryPatch['type'] = 'zeroGradient'
            #     self.cfdSetBoundaryPatch('rho',boundaryPatch,Region)
            #     # Region.fluid['rho'].boundaryPatch = boundaryPatch
            #     Region.fluid['rho'].cfdUpdateScale(Region)
            # # Region.compressible='false' 
            
            if not 'mu' in transportKeys:
                if 'nu' in transportKeys and 'rho' in transportKeys:
                    dimensions_mu=Region.fluid['nu'].phi.dimension*Region.fluid['rho'].phi.dimension
                    Region.fluid['mu']=field.Field(Region,'mu','volScalarField',dimensions_mu)
                    Region.fluid['mu'].phi=Region.fluid['nu'].phi*Region.fluid['rho'].phi
                    boundaryPatch={}
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)
                    # for iPatch in range(0,numberOfBPatches):
                    # boundaryPatch['value'] = 1
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('mu',boundaryPatch,Region)
                    Region.fluid['mu'].cfdUpdateScale(Region)
                    # Scalar.cfdUpdateScale(self.fluid['mu'],self)
                # boundaryPatch={} 
                # Region.fluid['mu']=field.Field(Region,'mu','volScalarField')
                # Region.fluid['mu'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 
                # Region.fluid['mu'].phi.fill(1E-3)
                # # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)
                # # for iPatch in range(0,numberOfBPatches):
                # boundaryPatch['value'] = 1
                # boundaryPatch['type'] = 'zeroGradient'
                # self.cfdSetBoundaryPatch('mu',boundaryPatch,Region)
                # Region.fluid['mu'].cfdUpdateScale(Region)
                # Scalar.cfdUpdateScale(self.fluid['mu'],self)
            
            # if 'Cp' in transportKeys:
            #     boundaryPatch={} 
            #     Region.fluid['Cp']=field.Field(Region,'Cp','volScalarField')
            #     Region.fluid['Cp'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 
            #     Region.fluid['Cp'].phi.fill(1004.)
            #     # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)
            #     # for iPatch in range(0,numberOfBPatches):
            #     boundaryPatch['value'] = 1
            #     boundaryPatch['type'] = 'zeroGradient'
            #     self.cfdSetBoundaryPatch('Cp',boundaryPatch,Region)
            #     Region.fluid['Cp'].cfdUpdateScale(Region)
            #     # Scalar.cfdUpdateScale(self.fluid['Cp'],self)

            # if 'k' in transportKeys:
            #     if 'DT' in transportKeys:
               
            #         boundaryPatch={} 
                    
            #         DTField = Region.fluid['DT'].phi
            #         CpField = Region.fluid['Cp'].phi
            #         rhoField = Region.fluid['rho'].phi            

            #         Region.fluid['k']=field.Field(Region,'k','volScalarField')
            #         Region.fluid['k'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 
            #         Region.fluid['k'].phi= DTField*CpField*rhoField  
    
            #         # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)
                    
            #         # for iPatch in range(0,numberOfBPatches):
            #         boundaryPatch['value'] = 1
            #         boundaryPatch['type'] = 'zeroGradient'
            #         self.cfdSetBoundaryPatch('k',boundaryPatch,Region)
            #         Region.fluid['k'].cfdUpdateScale()
            #         # Scalar.cfdUpdateScale(self.fluid['k'],self)


    def cfdReadThermophysicalProperties(self,Region):
        '''
        这段Python代码定义了一个名为`cfdReadThermophysicalProperties`的方法，用于读取OpenFOAM案例中的`thermophysicalProperties`字典文件，并根据文件内容设置流体区域（`Region.fluid`）的热物理属性。以下是对这个方法的详细解释：

        1. **文件路径构造**：
        - `thermophysicalPropertiesFilePath`构造了`thermophysicalProperties`文件的完整路径。

        2. **文件存在性检查**：
        - 使用`os.path.isfile`检查`thermophysicalProperties`文件是否存在，如果不存在，则打印文件路径并跳过后续操作。

        3. **读取字典**：
        - 如果文件存在，调用`io.cfdReadAllDictionaries`函数读取文件中的所有字典条目。

        4. **初始化热物理属性字典**：
        - `self.thermophysicalProperties={}`初始化一个空字典，用于存储热物理属性。

        5. **处理混合物属性**：
        - 根据`thermoType`字典中的`mixture`和`specie`等键，读取和存储混合物的组成和热力学数据。

        6. **处理输运模型**：
        - 根据`transport`键的值，选择合适的输运模型（如`const`或`sutherland`），并根据模型计算动力黏性系数`mu`和其他输运属性。

        7. **处理热力学模型**：
        - 根据`thermo`键的值，选择合适的热力学模型（如`hConst`），并根据模型设置比热容`Cp`和热导率`k`。

        8. **处理状态方程**：
        - 根据`equationOfState`键的值，选择合适的状态方程模型（如`perfectGas`或`Boussinesq`），并根据模型计算密度`rho`。

        9. **设置边界条件**：
        - 对于每个输运属性和热物理属性，设置边界条件，通常使用`'zeroGradient'`类型。

        10. **更新比例尺**：
            - 对于某些场（如`rho`），调用`cfdUpdateScale`方法更新比例尺。

        11. **错误处理**：
            - 如果在`thermophysicalProperties`字典中未识别出输运模型、热力学模型或状态方程模型，打印错误消息并退出程序。

        ### 注意事项：
        - 这段代码中使用的`io`模块中的函数`cfdReadAllDictionaries`没有在代码中定义，它可能是在类的其他部分或外部模块定义的。
        - `self.Region`预期是一个包含OpenFOAM案例信息的对象，具有`caseDirectoryPath`、`fluid`、`mesh`等属性。
        - 方法中处理了多种热物理属性和模型，包括混合物组成、输运模型、热力学模型和状态方程。
        - 代码中使用`vars`函数来访问对象属性的值，这在Python 3中可能不是最佳实践，更推荐直接访问属性。

        `cfdReadThermophysicalProperties`方法的目的是读取OpenFOAM案例的热物理属性配置，并将这些属性存储在`self.thermophysicalProperties`字典中，同时初始化和设置流体区域的场对象，为CFD模拟的初始化和数据管理提供支持。
        '''
        thermophysicalPropertiesFilePath=Region.caseDirectoryPath+"/constant/thermophysicalProperties"
                        
        if not os.path.isfile(thermophysicalPropertiesFilePath):
            print('%s not exist\n' %thermophysicalPropertiesFilePath)
            pass
    
        else:
            # print('\n')
            print('Reading thermophysical properties ...')
    
            thermophysicalDicts=io.cfdReadAllDictionaries(thermophysicalPropertiesFilePath)
            # thermophysicalKeys=list(thermophysicalDicts)   

            self.thermophysicalProperties={}
            self.thermophysicalProperties['thermoType'] = thermophysicalDicts['thermoType']
        
            
            if self.thermophysicalProperties['thermoType']['mixture']==['pureMixture']:
                # specieBlock = self.thermophysicalProperties['thermoType']['specie']

                # Read and store the specie subdict
                specieList = thermophysicalDicts['mixture']['specie']
                
                for i in specieList: 
                    specieList[i]=float(specieList[i][0])
                
                self.thermophysicalProperties['mixture']={}
                self.thermophysicalProperties['mixture']['specie']={}
                self.thermophysicalProperties['mixture']['specie'].update(specieList)
                
                
                # Read and store the thermodynamics subdict    
                thermoList = thermophysicalDicts['mixture']['thermodynamics']    
                
                for i in thermoList: 
                    thermoList[i]=float(thermoList[i][0])
                    
                self.thermophysicalProperties['mixture']['thermodynamics']={}
                self.thermophysicalProperties['mixture']['thermodynamics'].update(thermoList)
                
                
                # Read and store the transport properties subdict    
                transportList = thermophysicalDicts['mixture']['transport']    
                
                for i in transportList: 
                    transportList[i]=float(transportList[i][0])
                    
                self.thermophysicalProperties['mixture']['transport']={}
                self.thermophysicalProperties['mixture']['transport'].update(transportList)
                
                
                # Read and store the transport model 
                if self.thermophysicalProperties['thermoType']['transport']==['const']:
        
                    print('\n Using transport model: const')
                    # Update mu 
                    muValue = self.thermophysicalProperties['mixture']['transport']['mu']
 
                    Region.fluid['mu']=field.Field(Region,'mu','volScalarField')
                    Region.fluid['mu'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 
                    Region.fluid['mu'].phi= [[muValue] for i in range(Region.mesh.numberOfElements+Region.mesh.numberOfBElements)]
    
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)

                    boundaryPatch={}                     
                    # for iPatch in range(0,numberOfBPatches):
                    boundaryPatch['value'] = muValue
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('mu',boundaryPatch,Region)
                        

                    # Update Pr
                    PrValue = self.thermophysicalProperties['mixture']['transport']['Pr']
 
                    Region.fluid['Pr']=field.Field(Region,'Pr','volScalarField')
                    Region.fluid['Pr'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 
                    Region.fluid['Pr'].phi= [[PrValue] for i in range(Region.mesh.numberOfElements+Region.mesh.numberOfBElements)]
    
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)

                    boundaryPatch={}                     
                    # for iPatch in range(0,numberOfBPatches):
                    boundaryPatch['value'] = PrValue
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('Pr',boundaryPatch,Region)

                elif self.thermophysicalProperties['thermoType']['transport']==['sutherland']:
                    print('\n Using transport model: sutherland')    

                    if not 'T' in Region.fluid.keys():
                        print('Sutherland model requires T, which is not there \n')
                        
                    else:
                        AsValue = self.thermophysicalProperties['mixture']['transport']['As'] # No pun intended
                        TsValue = self.thermophysicalProperties['mixture']['transport']['Ts']
                        TField = vars(Region.fluid['T'])
                        TField = np.array(TField['phi'])
                        # Update mu according to the sutherland law
     
                        Region.fluid['mu']=field.Field(Region,'mu','volScalarField')
                        Region.fluid['mu'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 

                        Region.fluid['mu'].phi= AsValue*np.sqrt(TField)/(1+np.divide(TsValue,TField))
        
                        # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)
    
                        boundaryPatch={}                     
                        # for iPatch in range(0,numberOfBPatches):
                        boundaryPatch['value'] = muValue
                        boundaryPatch['type'] = 'zeroGradient'
                        self.cfdSetBoundaryPatch('mu',boundaryPatch,Region)
                        
                elif self.thermophysicalProperties['thermoType']['transport']==['polynomial']:
                    # print('polynomial transport model not yet implemented, sorry\n')
                    # sys.exit()
                    io.cfdError('polynomial transport model not yet implemented, sorry\n')

                else:
                    print('\nERROR: transport model in thermophysicalProperties not recognized. Valid entries are:')
                    print('const')
                    print('sutherland')
                    sys.exit()

####             Read and store the thermodynamics model 
                if self.thermophysicalProperties['thermoType']['thermo']==['hConst']:
                    print('\n Using thermodynamics model: hConst')                        
#               Update Cp 
                    CpValue = self.thermophysicalProperties['mixture']['thermodynamics']['Cp']
 
                    Region.fluid['Cp']=field.Field(Region,'Cp','volScalarField')
                    Region.fluid['Cp'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 
                    Region.fluid['Cp'].phi= [[CpValue] for i in range(Region.mesh.numberOfElements+Region.mesh.numberOfBElements)]
    
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)

                    boundaryPatch={}                     
                    # for iPatch in range(0,numberOfBPatches):
                    boundaryPatch['value'] = CpValue
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('Cp',boundaryPatch,Region)

#               Update k, thermal conductivity
                    PrValue = self.thermophysicalProperties['mixture']['transport']['Pr'] 

                    muField = vars(Region.fluid['mu'])
                    muField = np.array(muField['phi'])

                    CpField = vars(Region.fluid['Cp'])
                    CpField = np.array(CpField['phi'])
 
                    Region.fluid['k']=field.Field(Region,'k','volScalarField')
                    Region.fluid['k'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 

                    Region.fluid['k'].phi= np.multiply(muField,CpField)/PrValue
    
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)

                    boundaryPatch={}                     
                    # for iPatch in range(0,numberOfBPatches):
                        # boundaryPatch['value'] = kValue;
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('k',boundaryPatch,Region)

                elif self.thermophysicalProperties['thermoType']['thermo']==['hPolynomial']:
                    # print('hPolynomial transport model not yet implemented, sorry\n')
                    # sys.exit()
                    io.cfdError('hPolynomial transport model not yet implemented, sorry\n')

                else:
                    print('\nERROR: thermodynamics model in thermophysicalProperties not recognized. Valid entries are:')
                    print('hConst')
                    sys.exit()


####             Read and store the Equation of State model 
                if self.thermophysicalProperties['thermoType']['equationOfState']==['perfectGas']:               
                    print('\n Using equationOfState model: perfectGas') 

                    Region.compressible=True

#               Update rho, density

                    TField = vars(Region.fluid['T'])
                    TField = np.array(TField['phi'])

                    PField = vars(Region.fluid['P'])
                    PField = np.array(PField['phi'])

                    molWeightValue = self.thermophysicalProperties['mixture']['specie']['molWeight'] 
                    RuValue = 8.314e3 # Universal gas constant in SI 
                    RbarValue = RuValue / molWeightValue # Gas constant in SI 

                    Region.fluid['rho']=field.Field(Region,'rho','volScalarField')
                    Region.fluid['rho'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 

                    Region.fluid['rho'].phi= np.divide(PField,RbarValue*TField)
    
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)

                    boundaryPatch={}                     
                    # for iPatch in range(0,numberOfBPatches):
                        # boundaryPatch['value'] = rhoValue;
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('rho',boundaryPatch,Region)

                    Region.fluid['rho'].cfdUpdateScale(Region)

#                   Update drhodp, (1/RT)
                    Region.fluid['drhodp']=field.Field(Region,'drhodp','volScalarField')
                    Region.fluid['drhodp'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 

                    Region.fluid['drhodp'].phi= np.divide(1,RbarValue*TField)
    
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)

                    boundaryPatch={}                     
                    # for iPatch in range(0,numberOfBPatches):
                        # boundaryPatch['value'] = drhodpValue;
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('drhodp',boundaryPatch,Region)


                elif self.thermophysicalProperties['thermoType']['equationOfState']==['Boussinesq']: 
                    print('\n Using equationOfState model: Boussinesq') 

                    Region.compressible=True

#               Update rho, density

                    TField = vars(Region.fluid['T'])
                    TField = np.array(TField['phi'])

                    TRefValue = self.thermophysicalProperties['mixture']['thermodynamics']['TRef'] 
                    betaValue = self.thermophysicalProperties['mixture']['thermodynamics']['beta']                     
                    rhoRefValue = self.thermophysicalProperties['mixture']['thermodynamics']['rhoRef']                     


                    Region.fluid['rho']=field.Field(Region,'rho','volScalarField')
                    Region.fluid['rho'].dimensions=[0., 0., 0., 0., 0., 0.,0.] 

                    auxTerm = 1-betaValue*(TField-TRefValue)

                    Region.fluid['rho'].phi= np.multiply(rhoRefValue,auxTerm)
    
                    # numberOfBPatches=int(self.Region.mesh.numberOfBoundaryPatches)

                    boundaryPatch={}                     
                    # for iPatch in range(0,numberOfBPatches):
                        # boundaryPatch['value'] = rhoValue;
                    boundaryPatch['type'] = 'zeroGradient'
                    self.cfdSetBoundaryPatch('rho',boundaryPatch,Region)
                    Region.fluid['rho'].cfdUpdateScale(Region)

                elif self.thermophysicalProperties['thermoType']['equationOfState']==['incompressiblePerfectGas']: 
                    print('incompressiblePerfectGas equationOfState model not yet implemented, sorry\n')

                else:
                    print('\nERROR: equationOfState model in thermophysicalProperties not recognized. Valid entries are:')
                    print('Boussinesq')
                    print('perfectGas')

    def cfdSetBoundaryPatch(self,fieldName,boundaryPatch,Region):
        for iBPatch, theBCInfo in Region.mesh.cfdBoundaryPatchesArray.items():
            Region.fluid[fieldName].boundaryPatchRef[iBPatch] = boundaryPatch                            