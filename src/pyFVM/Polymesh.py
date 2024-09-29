import numpy as np
import cfdtool.IO as io
import cfdtool.Math as mth
from cfdtool.quantities import Quantity as Q_
import cfdtool.dimensions as dm

class Polymesh():
    """ Handles all mesh related methods.
        
        The Polymesh class takes care of all mesh related operations: 
        Read the polymesh directory and instance the mesh elements (points, faces, neighbours, owners... ) in memory
        Process topology and connectivity
        这段Python代码定义了一个名为`Polymesh`的类，它用于处理与网格（mesh）相关的所有操作，特别是在计算流体动力学（CFD）的上下文中。以下是对类和其构造器的详细解释：

        ### 类：Polymesh
        - **目的**：管理所有与网格相关的方法，包括读取网格文件，实例化网格元素（点、面、邻居、所有者等），处理拓扑结构和连接性。
        
    """    
    
    def __init__(self, Region):
        """Sets paths of mesh files, reads in mesh file data and calls numerous functions to process the mesh toplogy.

        ### 构造器：`__init__`
        - **参数**：
        - `Region`：包含案例目录路径和其他区域相关信息的对象。

        - **功能**：
        - 设置网格文件的路径。
        - 读取网格文件数据。
        - 调用多个函数来处理网格拓扑结构。

        - **步骤**：
        1. 初始化实例变量，包括网格文件的路径。
        2. 打印信息，提示开始读取`./constant/polyMesh`文件夹中的内容。
        3. 调用以下方法来读取网格的不同组成部分：
            - `cfdReadPointsFile`：读取点文件。
            - `cfdReadFacesFile`：读取面文件。
            - `cfdReadOwnerFile`：读取所有者文件。
            - `cfdReadNeighbourFile`：读取邻居文件。
        4. 计算边界面的数量、元素的数量、边界元素的数量。
        5. 调用`cfdReadBoundaryFile`来读取边界文件。
        6. 调用`cfdCheckIfCavity`来检查是否有腔体存在。
        7. 打印信息，提示正在处理网格。
        8. 调用以下方法来处理网格的拓扑、节点和几何：
            - `cfdProcessElementTopology`
            - `cfdProcessNodeTopology`
            - `cfdProcessGeometry`
        9. 调用一系列方法来获取边界补丁的子数组：
            - `cfdGetBoundaryElementsSubArrayForBoundaryPatch`
            - `cfdGetOwnersSubArrayForBoundaryPatch`
            - `cfdGetFaceSfSubArrayForBoundaryPatch`
            - `cfdGetFaceCentroidsSubArrayForBoundaryPatch`
        10. 初始化多个实例变量以存储内部面的所有者、邻居、权重、法向量和CF向量，以及边界面的相关信息。

        ### 重要属性：
        - `self.pointsFile`、`self.facesFile`等：存储网格文件的路径。
        - `self.numberOfBFaces`、`self.numberOfElements`等：存储网格统计信息。
        - `self.interiorFaceOwners`、`self.interiorFaceNeighbours`等：存储内部面和边界面的相关信息。

        ### 注意事项：
        - 构造器中的方法调用（如`cfdReadPointsFile`）在类的其他部分定义的。
        - 构造器执行了大量操作，包括文件读取、数据解析和网格处理，这些操作可能需要较长时间来完成，尤其是在处理大型网格时。
        - 类的文档字符串提供了类的概述和构造器的功能描述，但缺少具体的使用方法示例。

        `Polymesh`类是CFD模拟中网格处理的基础，负责读取和处理网格数据，为后续的模拟步骤提供必要的网格信息。
        """
        
        # self.Region=Region
        
        ## Path to points files
        self.pointsFile = r"%s/constant/polyMesh/points" % Region.caseDirectoryPath

        ## Path to faces file
        self.facesFile = r"%s/constant/polyMesh/faces" % Region.caseDirectoryPath

        ## Path to owner file
        self.ownerFile = r"%s/constant/polyMesh/owner" % Region.caseDirectoryPath

        ## Path to neighbour file
        self.neighbourFile = r"%s/constant/polyMesh/neighbour" % Region.caseDirectoryPath

        ## Path to boundary file
        self.boundaryFile = r"%s/constant/polyMesh/boundary" % Region.caseDirectoryPath 
        
        print('\n')
        print('Reading contents of ./constant/polyMesh folder ...')
        
        self.cfdReadPointsFile()
        self.cfdReadFacesFile()
        self.cfdReadOwnerFile()
        self.cfdReadNeighbourFile()

        #maybe these should go in a function?
        self.numberOfBFaces=self.numberOfFaces-self.numberOfInteriorFaces
        self.numberOfElements = int(np.max(self.neighbours)+1) #because of zero indexing in Python
        self.numberOfBElements=self.numberOfFaces-self.numberOfInteriorFaces #seems strange that subtracting faces gives elements ...

        self.cfdReadBoundaryFile()  
        self.cfdCheckIfCavity()
        
        print('Processing mesh ... please wait ....')

        self.OrthogonalCorrectionMethod='OverRelaxed'
        
        self.cfdProcessElementTopology()
        self.cfdProcessNodeTopology()
        self.cfdProcessGeometry()
        
        self.cfdGetBoundaryElementsSubArrayForBoundaryPatch()
        self.cfdGetFaceSfSubArrayForBoundaryPatch()
        self.cfdGetOwnersSubArrayForBoundaryPatch()
        self.cfdGetFaceCentroidsSubArrayForBoundaryPatch()
        
        ## (list) 1D, indices refer to an interior face, list value is the face's owner
        self.interiorFaceOwners = self.owners[:self.numberOfInteriorFaces]

        ## (list) 1D, indices refer to an interior face, list value is the face's neighbor cell
        self.interiorFaceNeighbours = self.neighbours[:self.numberOfInteriorFaces]

        ## (list) 1D, face weighting factors. Values near 0.5 mean the face's centroid is approximately halfway between the center of the owner and neighbour cell centers, values less than 0.5 mean the face centroid is closer to the owner and those greater than 0.5 are closer to the neighbour cell).
        self.interiorFaceWeights = self.faceWeights[:self.numberOfInteriorFaces]

        ## (array) 2D, normal vectors (Sf) of the interior faces (indices refer to face index)
        self.interiorFaceSf = self.faceSf[:self.numberOfInteriorFaces]
        
        ## (array) 2D, CF vectors of the interior faces (indices refer to face index)
        self.interiorFaceCF = self.faceCF[:self.numberOfInteriorFaces]
        
        ## (list) 1D, indices refer to an boundary face, list value refers to the face's owner
        self.owners_b = self.owners[self.numberOfInteriorFaces:self.numberOfFaces]

        ## (list) 1D, normal vectors (Sf) of the boundary faces (indices refer to face index). Boundary face normals always point out of the domain. 
        self.Sf_b=self.faceSf[self.numberOfInteriorFaces:self.numberOfFaces]

        self.iBElements = np.arange(self.numberOfElements, self.numberOfElements+self.numberOfBFaces, dtype=int)

        self.cfdGeometricLengthScale()

    def cfdGeometricLengthScale(self):
        # Calculates the geometric length scale of the mesh. 
        # Length scale = [sum(element volume)]^(1/3)
        self.totalVolume = Q_(np.sum(self.elementVolumes.value),dm.volume_dim)
        self.lengthScale = self.totalVolume**(1/3)
        
    def cfdReadPointsFile(self):
        """ Reads the constant/polyMesh/points file in polymesh directory and stores the points coordinates
        into region.mesh.nodeCentroids
        这段Python代码定义了一个名为`cfdReadPointsFile`的方法，它用于读取OpenFOAM案例中`constant/polyMesh/points`文件，并存储点的坐标到`region.mesh.nodeCentroids`。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdReadPointsFile(self):` 定义了一个实例方法，没有接收额外的参数，而是使用实例属性。

        2. **打开文件**：
        - 使用`with open(self.pointsFile,"r") as fpid:`以只读模式打开`pointsFile`路径指定的文件，并将其文件对象赋值给`fpid`。

        3. **读取文件**：
        - 打印信息，提示开始读取点文件。
        - 初始化空列表`points_x`、`points_y`、`points_z`，用于存储点的x、y、z坐标。

        4. **逐行读取**：
        - 使用`for linecount, tline in enumerate(fpid):`遍历文件的每一行，并使用`enumerate`获取行号和行内容。

        5. **跳过空行和宏定义注释**：
        - 使用`io.cfdSkipEmptyLines`和`io.cfdSkipMacroComments`函数跳过空行和宏定义注释行。

        6. **跳过FoamFile字典**：
        - 如果行包含`"FoamFile"`，则使用`io.cfdReadCfdDictionary`读取随后的字典定义，然后继续下一行。

        7. **处理点的数量行**：
        - 如果行只包含一个数字，并且该行包含`"("`或`")"`，则跳过该行。
        - 如果该行包含点的数量，将其转换为整数并存储在`self.numberOfNodes`。

        8. **解析点坐标**：
        - 清理每行的内容，移除圆括号。
        - 分割行内容并转换为浮点数，然后将x、y、z坐标分别添加到相应的列表。

        9. **转换为NumPy数组**：
        - 使用`np.array`将点坐标的列表转换为NumPy数组，并使用`.transpose()`方法转置数组，使得每一列代表一个坐标轴的所有点的坐标。

        10. **存储节点坐标**：
            - 将得到的NumPy数组赋值给`self.nodeCentroids`，作为区域（Region）的节点坐标存储。

        ### 注意事项：
        - 这段代码使用了`io`模块中的一些函数，如`cfdSkipEmptyLines`、`cfdSkipMacroComments`和`cfdReadCfdDictionary`，这些函数在代码中没有给出定义，可能是在类的其他部分或外部模块定义的。
        - `self.pointsFile`是类实例的属性，它在构造器中被设置为`constant/polyMesh/points`文件的路径。
        - `self.numberOfNodes`是类实例的属性，用于存储网格中节点的数量。
        - `self.nodeCentroids`是类实例的属性，用于存储转换后的节点坐标NumPy数组。

        这个方法是读取和解析OpenFOAM网格点文件的关键步骤，为后续的网格处理和CFD模拟提供了节点坐标数据。
        """

        with open(self.pointsFile,"r") as fpid:
            
            print('Reading points file ...')
            points = []
            # points_x=[]
            # points_y=[]
            # points_z=[]
            
            for linecount, tline in enumerate(fpid):
                
                if not io.cfdSkipEmptyLines(tline):
                    continue
                
                if not io.cfdSkipMacroComments(tline):
                    continue
                
                if "FoamFile" in tline:
                    dictionary=io.cfdReadCfdDictionary(fpid)
                    continue
    
                if len(tline.split()) ==1:
                    if "(" in tline:
                        continue
                    if ")" in tline:
                        continue
                    else:
                        self.numberOfNodes = int(tline.split()[0])
                        continue
                
                tline=tline.replace("(","")
                tline=tline.replace(")","")
                tline=tline.split()
                
                points.append(list(map(float, tline)))
                # points_x.append(float(tline[0]))
                # points_y.append(float(tline[1]))
                # points_z.append(float(tline[2]))
        
        ## (array) with the mesh point coordinates 
        # self.nodeCentroids = np.array((points_x, points_y, points_z), dtype=float).transpose()
        self.nodeCentroids = np.array(points, dtype=float)

    def cfdReadFacesFile(self):
        """ Reads the constant/polyMesh/faces file and stores the nodes pertaining to each face
            in region.mesh.faceNodes
            Starts with interior faces and then goes through the boundary faces. 
            这段Python代码定义了一个名为`cfdReadFacesFile`的方法，它用于读取OpenFOAM案例中`constant/polyMesh/faces`文件，并存储每个面所关联的节点到`region.mesh.faceNodes`。以下是对这个方法的详细解释：

            1. **方法定义**：
            - `def cfdReadFacesFile(self):` 定义了一个实例方法，没有接收额外的参数，而是使用实例属性。

            2. **打开文件**：
            - 使用`with open(self.facesFile,"r") as fpid:`以只读模式打开`facesFile`路径指定的文件，并将其文件对象赋值给`fpid`。

            3. **读取文件**：
            - 打印信息，提示开始读取面文件。
            - 初始化空列表`self.faceNodes`，用于存储每个面的节点列表。

            4. **逐行读取**：
            - 使用`for linecount, tline in enumerate(fpid):`遍历文件的每一行，并使用`enumerate`获取行号和行内容。

            5. **跳过空行和宏定义注释**：
            - 使用`io.cfdSkipEmptyLines`和`io.cfdSkipMacroComments`函数跳过空行和宏定义注释行。

            6. **跳过FoamFile字典**：
            - 如果行包含`"FoamFile"`，则使用`io.cfdReadCfdDictionary`读取随后的字典定义，然后继续下一行。

            7. **处理面的数量行**：
            - 如果行只包含一个数字，并且该行包含`"("`或`")"`，则跳过该行。
            - 如果该行包含面的数量，将其转换为整数并存储在`self.numberOfFaces`。

            8. **解析面节点**：
            - 清理每行的内容，移除圆括号，并将剩余内容按空格分割。
            - 初始化空列表`faceNodesi`，用于存储当前面的节点索引。
            - 遍历分割后的行内容，跳过第一个元素（通常是面的数量），然后将剩余节点索引转换为浮点数并添加到`faceNodesi`。

            9. **存储面节点信息**：
            - 将`faceNodesi`添加到`self.faceNodes`列表。

            10. **转换为NumPy数组**：
                - 使用`np.asarray`将`self.faceNodes`列表转换为NumPy数组。

            11. **打印信息**：
                - 打印出转换后的NumPy数组，以便于调试和验证。

            ### 注意事项：
            - 这段代码使用了`io`模块中的一些函数，如`cfdSkipEmptyLines`、`cfdSkipMacroComments`和`cfdReadCfdDictionary`，这些函数在代码中没有给出定义，可能是在类的其他部分或外部模块定义的。
            - `self.facesFile`是类实例的属性，它在构造器中被设置为`constant/polyMesh/faces`文件的路径。
            - `self.numberOfFaces`是类实例的属性，用于存储网格中面的数量。
            - `self.faceNodes`是类实例的属性，用于存储转换后的每个面的节点索引NumPy数组。

            这个方法是读取和解析OpenFOAM网格面文件的关键步骤，为后续的网格处理和CFD模拟提供了面的节点信息。代码中存在一些不一致之处，例如注释掉的`faceNodesi.append(int(node))`和紧接着的`else`子句中的`float`转换，这可能是代码维护过程中的遗留问题。在实际使用中，应确保节点索引是整数，因此应使用`int`而不是`float`进行转换。
        """ 

        with open(self.facesFile,"r") as fpid:
            print('Reading faces file ...')
            # self.faceNodes=[]
            faces = []
            for linecount, tline in enumerate(fpid):
                
                if not io.cfdSkipEmptyLines(tline):
                    continue
                
                if not io.cfdSkipMacroComments(tline):
                    continue
                
                if "FoamFile" in tline:
                    dictionary=io.cfdReadCfdDictionary(fpid)
                    continue
    
                if len(tline.split()) ==1:
                    if "(" in tline:
                        continue
                    if ")" in tline:
                        continue
                    else:
                        
                        self.numberOfFaces = int(tline.split()[0])
                        continue
                
                tline=tline.replace("("," ")
                tline=tline.replace(")","")
                faceNodesi=[]
                for count, node in enumerate(tline.split()):
                    if count == 0:
                        continue
                        #faceNodesi.append(int(node))
                    else:
                        faceNodesi.append(int(node))
                faces.append(faceNodesi)
                # self.faceNodes.append(faceNodesi)

        # max_length = max(len(face) for face in faces)
        # self.faceNodes = np.array([face + [None] * (max_length - len(face)) for face in faces], dtype=object)
        self.faceNodes = np.array(faces, dtype=object)
            
                
        # ## (array) with the nodes for each face
        # max_length = max(len(item) for item in self.faceNodes)  # 找到最长序列的长度
        # # 使用 None 或其他适当的值填充较短的序列
        # padded_list = [item + [None] * (max_length - len(item)) for item in self.faceNodes]
        # self.faceNodes=np.asarray(padded_list,dtype=object)
        # print(self.faceNodes)

    def cfdReadOwnerFile(self):
        """ Reads the polyMesh/constant/owner file and returns a list 
        where the indexes are the faces and the corresponding element value is the owner cell
        这段Python代码定义了一个名为`cfdReadOwnerFile`的方法，它用于读取OpenFOAM案例中的`polyMesh/constant/owner`文件，并返回一个列表，其中索引是面，对应的元素值是拥有该面的单元格（即“所有者”细胞）。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdReadOwnerFile(self):` 定义了一个实例方法，它是`Polymesh`类的一部分，没有接收额外的参数。

        2. **打开文件**：
        - 使用`with open(self.ownerFile,"r") as fpid:`以只读模式打开`ownerFile`路径指定的文件，并将其文件对象赋值给`fpid`。

        3. **读取文件**：
        - 打印信息，提示开始读取所有者文件。
        - 初始化空列表`self.owners`，用于存储每个面的拥有者单元格的索引。

        4. **逐行读取**：
        - 使用`for linecount, tline in enumerate(fpid):`遍历文件的每一行，并使用`enumerate`获取行号和行内容。

        5. **跳过空行和宏定义注释**：
        - 使用`io.cfdSkipEmptyLines`和`io.cfdSkipMacroComments`函数跳过空行和宏定义注释行。

        6. **跳过FoamFile字典**：
        - 如果行包含`"FoamFile"`，则使用`io.cfdReadCfdDictionary`读取随后的字典定义，然后继续下一行。

        7. **处理开始标记**：
        - 如果行只包含一个数字，并且之前没有设置`start`标志，则认为这是拥有者数量的声明，并设置`start`标志。

        8. **读取所有者数据**：
        - 当设置`start`标志后，如果行包含`"("`，则跳过该行。
        - 如果行包含`")"`，则表示所有者数据的结束，使用`break`退出循环。
        - 对于其他行，将行内容分割并转换为整数，然后将拥有者单元格的索引添加到`self.owners`列表。

        9. **存储所有者列表**：
        - 完成读取后，`self.owners`列表包含了每个面的拥有者单元格的索引。

        ### 注意事项：
        - 这段代码使用了`io`模块中的一些函数，如`cfdSkipEmptyLines`、`cfdSkipMacroComments`和`cfdReadCfdDictionary`，这些函数在代码中没有给出定义，可能是在类的其他部分或外部模块定义的。
        - `self.ownerFile`是类实例的属性，它在构造器中被设置为`polyMesh/constant/owner`文件的路径。
        - `nbrOwner`变量用于存储拥有者的数量，但在这段代码中没有使用这个变量。
        - `start`标志用于确定何时开始读取所有者数据。
        - 代码中有一个逻辑判断错误，`if not start:` 后面应该直接读取拥有者数量，而不是等到设置`start`标志后才读取。

        这个方法是读取和解析OpenFOAM网格所有者文件的关键步骤，为后续的网格处理和CFD模拟提供了面的拥有者信息。
        """ 

        with open(self.ownerFile,"r") as fpid:
            print('Reading owner file ...')

            ## (list) 1D, indices refer to faces, list value is the face's owner cell
            owners=[]
            start=False
            
            for linecount, tline in enumerate(fpid):
                
                if not io.cfdSkipEmptyLines(tline):
                    continue
                
                if not io.cfdSkipMacroComments(tline):
                    continue
                
                if "FoamFile" in tline:
                    dictionary=io.cfdReadCfdDictionary(fpid)
                    continue
        
                if len(tline.split()) ==1:
                   
                    #load and skip number of owners
                    if not start:
                        nbrOwner=tline
                        start=True
                        continue
        
                    if "(" in tline:
                        continue
                    if ")" in tline:
                        break
                    else:
                        owners.append(int(tline.split()[0]))

        self.owners= np.array(owners)

    def cfdReadNeighbourFile(self):
        """ Reads the polyMesh/constant/neighbour file and returns a list 
        where the indexes are the faces and the corresponding element value is the neighbour cell
        这段Python代码定义了一个名为`cfdReadNeighbourFile`的方法，它用于读取OpenFOAM案例中的`polyMesh/constant/neighbour`文件，并返回一个列表，其中索引是面，对应的元素值是相邻单元格（即“邻居”细胞）。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdReadNeighbourFile(self):` 定义了一个实例方法，它是`Polymesh`类的一部分，没有接收额外的参数。

        2. **打开文件**：
        - 使用`with open(self.neighbourFile,"r") as fpid:`以只读模式打开`neighbourFile`路径指定的文件，并将其文件对象赋值给`fpid`。

        3. **读取文件**：
        - 打印信息，提示开始读取邻居文件。
        - 初始化空列表`self.neighbours`，用于存储每个面的邻居单元格的索引。

        4. **逐行读取**：
        - 使用`for linecount, tline in enumerate(fpid):`遍历文件的每一行，并使用`enumerate`获取行号和行内容。

        5. **跳过空行和宏定义注释**：
        - 使用`io.cfdSkipEmptyLines`和`io.cfdSkipMacroComments`函数跳过空行和宏定义注释行。

        6. **跳过FoamFile字典**：
        - 如果行包含`"FoamFile"`，则使用`io.cfdReadCfdDictionary`读取随后的字典定义，然后继续下一行。

        7. **处理开始标记**：
        - 如果行只包含一个数字，并且之前没有设置`start`标志，则认为这是内部面的数量，并设置`start`标志。

        8. **读取邻居数据**：
        - 当设置`start`标志后，如果行包含`"("`，则跳过该行。
        - 如果行包含`")"`，则表示邻居数据的结束，使用`break`退出循环。
        - 对于其他行，将行内容分割并转换为整数，然后将邻居单元格的索引添加到`self.neighbours`列表。

        9. **存储邻居列表**：
        - 完成读取后，`self.neighbours`列表包含了每个面的邻居单元格的索引。

        ### 注意事项：
        - 这段代码使用了`io`模块中的一些函数，如`cfdSkipEmptyLines`、`cfdSkipMacroComments`和`cfdReadCfdDictionary`，这些函数在代码中没有给出定义，可能是在类的其他部分或外部模块定义的。
        - `self.neighbourFile`是类实例的属性，它在构造器中被设置为`polyMesh/constant/neighbour`文件的路径。
        - `self.numberOfInteriorFaces`是类实例的属性，用于存储网格内部面的数量。
        - `start`标志用于确定何时开始读取邻居数据。
        - 代码中假设邻居文件的格式与所有者文件相同，并且在读取完内部面的数量后立即开始读取邻居数据。

        这个方法是读取和解析OpenFOAM网格邻居文件的关键步骤，为后续的网格处理和CFD模拟提供了面的邻居信息。这有助于理解网格的拓扑结构，特别是在处理非结构化网格时。
        """ 
        with open(self.neighbourFile,"r") as fpid:
            print('Reading neighbour file ...')

            ## (list) 1D, indices refer to faces, list value is the face's neighbour cell
            neighbours=[]
            start=False
            
            for linecount, tline in enumerate(fpid):
                
                if not io.cfdSkipEmptyLines(tline):
                    continue
                
                if not io.cfdSkipMacroComments(tline):
                    continue
                
                if "FoamFile" in tline:
                    dictionary=io.cfdReadCfdDictionary(fpid)
                    continue
    
                if len(tline.split()) ==1:
                   
                    #load and skip number of owners
                    if not start:
                        self.numberOfInteriorFaces=int(tline)
                        start=True
                        continue
    
                    if "(" in tline:
                        continue
                    if ")" in tline:
                        break
                    else:
                        neighbours.append(int(tline.split()[0]))

        self.neighbours = np.array(neighbours)               
    
    def cfdReadBoundaryFile(self):
        """Reads the polyMesh/boundary file and reads its contents in a dictionary (self.cfdBoundary)
        这段Python代码定义了一个名为`cfdReadBoundaryFile`的方法，它用于读取OpenFOAM案例中的`polyMesh/boundary`文件，并将文件内容存储到一个字典`self.cfdBoundaryPatchesArray`中。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdReadBoundaryFile(self):` 定义了一个实例方法，它是`Polymesh`类的一部分，没有接收额外的参数。

        2. **打开文件**：
        - 使用`with open(self.boundaryFile,"r") as fpid:`以只读模式打开`boundaryFile`路径指定的文件，并将其文件对象赋值给`fpid`。

        3. **读取文件**：
        - 打印信息，提示开始读取边界文件。
        - 初始化空字典`self.cfdBoundaryPatchesArray`，用于存储边界补丁的信息。

        4. **逐行读取**：
        - 使用`for linecount, tline in enumerate(fpid):`遍历文件的每一行，并使用`enumerate`获取行号和行内容。

        5. **跳过空行和宏定义注释**：
        - 使用`io.cfdSkipEmptyLines`和`io.cfdSkipMacroComments`函数跳过空行和宏定义注释行。

        6. **跳过FoamFile字典**：
        - 如果行包含`"FoamFile"`，则使用`io.cfdReadCfdDictionary`读取随后的字典定义，然后继续下一行。

        7. **处理边界补丁**：
        - 对于每行，如果只包含一个元素且该元素是数字，则认为是边界补丁的数量，并存储在`self.numberOfBoundaryPatches`。
        - 如果行包含边界补丁的名称，则调用`io.cfdReadCfdDictionary`读取该补丁的属性，并存储到`self.cfdBoundaryPatchesArray`字典中。

        8. **提取边界补丁属性**：
        - 对于每个边界补丁，从其属性字典中提取`'nFaces'`（补丁的面数量）和`'startFace'`（补丁在`self.faceNodes`中的起始面索引）。
        - 将提取的属性添加到相应的边界补丁字典中，并更新`'index'`属性作为边界补丁的索引。

        9. **存储边界补丁信息**：
        - 完成读取后，`self.cfdBoundaryPatchesArray`字典包含了所有边界补丁的详细信息。

        ### 注意事项：
        - 这段代码使用了`io`模块中的一些函数，如`cfdSkipEmptyLines`、`cfdSkipMacroComments`和`cfdReadCfdDictionary`，这些函数在代码中没有给出定义，可能是在类的其他部分或外部模块定义的。
        - `self.boundaryFile`是类实例的属性，它在构造器中被设置为`polyMesh/constant/boundary`文件的路径。
        - `self.numberOfBoundaryPatches`是类实例的属性，用于存储网格边界补丁的数量。
        - 代码中使用`strip()`和`isdigit()`方法来检查行内容是否为数字，并据此确定是否处理边界补丁的数量或名称。

        这个方法是读取和解析OpenFOAM网格边界文件的关键步骤，为后续的网格处理和CFD模拟提供了边界补丁的详细信息。这有助于理解网格的边界条件和拓扑结构。
        """
       
        with open(self.boundaryFile,"r") as fpid:
            print('Reading boundary file ...')
            
            ## (dict) key for each boundary patch
            self.cfdBoundaryPatchesArray={}
            for linecount, tline in enumerate(fpid):
                
                if not io.cfdSkipEmptyLines(tline):
                    continue
                
                if not io.cfdSkipMacroComments(tline):
                    continue
                
                if "FoamFile" in tline:
                    dictionary=io.cfdReadCfdDictionary(fpid)
                    continue
    
                count=0
                if len(tline.split()) ==1:
                    if "(" in tline:
                        continue
                    if ")" in tline:
                        continue
                    
                    if tline.strip().isdigit():
                        '''
                        if tline.strip().isdigit(): 是一个Python条件语句，用于检查字符串tline在去除空白字符后是否只包含数字。下面是对这个条件语句的详细解释：isdigit()：这也是Python字符串的一个方法，用于判断字符串中的所有字符是否都是数字。如果字符串至少有一个字符，并且所有字符都是数字，则返回True；否则返回False。
                        '''
                        self.numberOfBoundaryPatches = int(tline.split()[0])
                        continue
                   
                    boundaryName=tline.split()[0]
                    
                    self.cfdBoundaryPatchesArray[boundaryName]=io.cfdReadCfdDictionary(fpid)
                    ## number of faces for the boundary patch
                    self.cfdBoundaryPatchesArray[boundaryName]['numberOfBFaces']= int(self.cfdBoundaryPatchesArray[boundaryName].pop('nFaces'))
                    '''
                    self.cfdBoundaryPatchesArray[boundaryName]['numberOfBFaces']= int(self.cfdBoundaryPatchesArray[boundaryName].pop('nFaces'))
                    这里pop()函数被用来从字典self.cfdBoundaryPatchesArray[boundaryName]中移除键'nFaces'，并将其值赋给numberOfBFaces。这通常表示在处理边界补丁的字典时，一旦读取了'nFaces'（可能表示该补丁的面的数量），就不再需要这个键，因此将其从字典中删除。通过使用pop()，代码可以确保'nFaces'键只会被处理一次，并且在处理后不会再次出现。同时，使用int()函数确保了取得的值是整数类型。
                    '''
                    
                    ## start face index of the boundary patch in the self.faceNodes
                    self.cfdBoundaryPatchesArray[boundaryName]['startFaceIndex']= int(self.cfdBoundaryPatchesArray[boundaryName].pop('startFace'))
                    count=count+1

                    ## index for boundary face, used for reference
                    self.cfdBoundaryPatchesArray[boundaryName]['index']= int(count)
    
                    
    def cfdCheckIfCavity(self):
        """Checks if there are any inlets or outlets in the domain, based 
        on the boundary types found in system/boundary
        
        if any of the boundaries is an inlet or outlet returns
        self.cfdIsClosedCavity = False
        为了判断是否需要再压力边界给出初始参考压力值!pressure Ref
        """
               
        self.cfdIsClosedCavity=True
        
        for patch, value in self.cfdBoundaryPatchesArray.items():
            
            if value['type'] == 'inlet' or value['type'] =='outlet':
                self.cfdIsClosedCavity =False
                break

    def cfdProcessElementTopology(self):

        """Populates self.elementNeighbours and self.elementFaces also populates self.upperAnbCoeffIndex self.lowerAnbCoeffIndex 
        这段Python代码定义了一个名为`cfdProcessElementTopology`的方法，它是`Polymesh`类的一部分，用于处理计算流体动力学（CFD）网格的元素拓扑结构。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdProcessElementTopology(self):` 定义了一个实例方法，用于在CFD模拟中建立网格元素之间的拓扑关系。

        2. **初始化元素邻居和面列表**：
        - `self.elementNeighbours`：一个列表的列表，其中每个元素代表网格中的一个单元格，包含与之共享面的邻近单元格的索引。
        - `self.elementFaces`：一个列表的列表，其中每个元素包含形成每个单元格的面索引。

        3. **填充元素邻居和面**：
        - 通过遍历内部面（`self.numberOfInteriorFaces`），为每个面填充其拥有者单元格和邻居单元格的索引，以及这些面在`self.elementFaces`中的索引。

        4. **添加边界面**：
        - 对于超出内部面范围的面索引，将它们添加到相应的`self.elementFaces`列表中，这些面是边界面。

        5. **初始化元素节点列表**：
        - `self.elementNodes`：一个列表的列表，每个元素包含形成每个单元格的节点。

        6. **填充元素节点**：
        - 通过遍历每个单元格的所有面，收集面的所有节点，然后使用集合去除重复项，最后更新`self.elementNodes`。

        7. **初始化系数索引列表**：
        - `self.upperAnbCoeffIndex`和`self.lowerAnbCoeffIndex`：分别为每个内部面初始化两个列表，用于存储与面相关的系数索引。

        8. **填充系数索引**：
        - 通过遍历每个单元格的所有面，确定面的所有者和邻居单元格，并根据单元格编号填充`self.upperAnbCoeffIndex`和`self.lowerAnbCoeffIndex`。

        ### 注意事项：
        - 这段代码假设`self.owners`和`self.neighbours`分别存储了面的所有者和邻居单元格的索引。
        - `self.faceNodes`存储了每个面的节点列表。
        - `self.numberOfElements`、`self.numberOfInteriorFaces`和`self.numberOfFaces`分别代表网格中的单元格总数、内部面的数量和总面的数量。
        - 代码中使用了列表推导式和集合来去除重复的节点。
        - 代码中可能存在一些逻辑错误或不一致之处，例如在填充系数索引时，`iNb`的递增似乎与单元格编号无关，这可能需要根据实际的拓扑结构进行调整。

        这个方法是CFD网格处理的关键步骤，为模拟提供了网格单元格之间的拓扑信息，这对于求解流体动力学方程和网格通信是必要的。

        """
        ## (list of lists) List where each index represents an element in the domain. Each index has an associated list which contains the elements for which is shares a face (i.e. the neighouring elements). Do not confuse a faces 'neighbour cell', which refers to a face's neighbour element, with the neighbouring elements of a cell. 
        # self.elementNeighbours = [[] for _ in range(0,self.numberOfElements)]
        self.elementNeighbours = np.empty(self.numberOfElements, dtype=object)

        ## (list of lists) list of face indices forming each element
        # self.elementFaces = [[] for _ in range(0,self.numberOfElements)]
        self.elementFaces = np.empty(self.numberOfElements, dtype=object)

        for i in range(self.numberOfElements):
            self.elementNeighbours[i] = []
            self.elementFaces[i] = []
        
        #populates self.elementNeighbours
        for iFace in range(self.numberOfInteriorFaces):
            own=self.owners[iFace]
            nei=self.neighbours[iFace]
            
            #adds indices of neighbour cells
            self.elementNeighbours[own].append(nei)
            self.elementNeighbours[nei].append(own)
            
            #adds interior faces
            self.elementFaces[own].append(iFace)
            self.elementFaces[nei].append(iFace)
        
        #adds boundary faces ('patches')
        for iFace in range(self.numberOfInteriorFaces,self.numberOfFaces):
            own=self.owners[iFace]
            self.elementFaces[own].append(iFace)
        
        ## List of lists containing points forming each element
        # self.elementNodes = [[] for i in range(0,self.numberOfElements)]
        self.elementNodes = np.empty(self.numberOfElements, dtype=object)
        for i in range(self.numberOfElements):
            self.elementNodes[i] = []
        
        for iElement in range(self.numberOfElements):
            for faceIndex in self.elementFaces[iElement]:
                self.elementNodes[iElement].extend(self.faceNodes[faceIndex])
                # self.elementNodes[iElement].append(self.faceNodes[faceIndex])
            self.elementNodes[iElement] = list(set(self.elementNodes[iElement]))
            # self.elementNodes[iElement] = list(set([item for sublist in self.elementNodes[iElement] for item in sublist]))
            '''
            这行代码是用于处理 `self.elementNodes` 列表中的特定元素 `iElement` 的。它的作用是将一个由多个子列表组成的嵌套列表（可能包含重复项）转换为一个没有重复项的扁平列表（flat list）。下面是对这个操作的详细解释：

            1. **列表推导式**：
            - `[item for sublist in self.elementNodes[iElement] for item in sublist]`：这是一个嵌套的列表推导式，用于将 `self.elementNodes[iElement]` 中的每个子列表 `sublist` 展开，生成一个包含所有子列表中所有项的临时列表。

            2. **集合转换**：
            - `set(...)`：将上述列表推导式的结果转换为一个集合（set）。由于集合是一个无序的不重复元素集，这一步将去除所有重复的项。

            3. **列表转换**：
            - `list(...)`：将上一步得到的集合再转换回一个列表。这一步是可选的，但通常这样做是为了保持数据的列表形式，以便于后续处理。

            4. **更新 `self.elementNodes[iElement]`**：
            - `self.elementNodes[iElement] = ...`：最后，将去重后的列表赋值回 `self.elementNodes` 列表的对应元素 `iElement`。

            ### 示例：
            假设 `self.elementNodes[iElement]` 的初始状态如下：

            ```python
            self.elementNodes[iElement] = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            ```

            这行代码将执行以下步骤：

            1. 通过列表推导式展开为 `[1, 2, 3, 2, 3, 4, 3, 4, 5]`。
            2. 转换为集合去除重复项，变为 `{1, 2, 3, 4, 5}`。
            3. 将集合转换回列表，结果可能是 `[1, 2, 3, 4, 5]`（注意列表中的元素顺序可能不同，因为集合是无序的）。

            最终，`self.elementNodes[iElement]` 更新为去重后的新列表。

            这种操作在处理网格单元格节点时非常有用，特别是当你需要确保每个单元格的节点列表中不包含重复节点时。
            '''
        ## Upper coefficient indices (owners)
        # # self.upperAnbCoeffIndex=[[] for i in range(0,self.numberOfInteriorFaces)]
        # self.upperAnbCoeffIndex = np.empty(self.numberOfInteriorFaces, dtype=object)
        # ## Lower coefficient indices (owners)
        # # self.lowerAnbCoeffIndex=[[] for i in range(0,self.numberOfInteriorFaces)]
        # self.lowerAnbCoeffIndex = np.empty(self.numberOfInteriorFaces, dtype=object)
        # for i in range(self.numberOfInteriorFaces):
        #     self.upperAnbCoeffIndex[i] = []
        #     self.lowerAnbCoeffIndex[i] = []

        # for iElement in range(self.numberOfElements):
        #     ## Element number from 1 to numberOfElements + 1
        #     iNb=0
        #     for faceIndex in self.elementFaces[iElement]:
        #         #skip if it is a boundary face
        #         if faceIndex > self.numberOfInteriorFaces-1:
        #             continue
        #         own = self.owners[faceIndex]
        #         nei = self.neighbours[faceIndex]
        #         if iElement == own:
        #             self.upperAnbCoeffIndex[faceIndex] = iNb
        #         elif iElement == nei:
        #             self.lowerAnbCoeffIndex[faceIndex] = iNb
        #         iNb += 1
        """
        初始化并填充 upperAnbCoeffIndex 和 lowerAnbCoeffIndex 数组。

        功能：
        该段代码用于为每个内部面（interior face）计算其对应于所有者单元（owner cell）和邻居单元（neighbor cell）的 anb 系数索引，并将这些索引存储在 upperAnbCoeffIndex 和 lowerAnbCoeffIndex 数组中。

        具体过程：
        1. 初始化 upperAnbCoeffIndex 和 lowerAnbCoeffIndex：
        - 创建长度为内部面数量（numberOfInteriorFaces）的整数数组，初始值为0。
        2. 对于每个内部面 iFace：
        a. 获取该面的所有者单元索引 own 和邻居单元索引 nei。
        b. 处理所有者单元（own）：
            - 获取所有者单元的邻居单元列表 own_neighbours。
            - 在 own_neighbours 中找到邻居单元 nei 的索引 nei_in_own_anb_index。
            - 将该索引存储在 upperAnbCoeffIndex 的第 iFace 个位置。
        c. 处理邻居单元（nei）：
            - 获取邻居单元的邻居单元列表 nei_neighbours。
            - 在 nei_neighbours 中找到所有者单元 own 的索引 own_in_nei_anb_index。
            - 将该索引存储在 lowerAnbCoeffIndex 的第 iFace 个位置。

        变量说明：
        - self.upperAnbCoeffIndex：存储每个内部面在所有者单元的 anb 系数数组中的索引。
        - self.lowerAnbCoeffIndex：存储每个内部面在邻居单元的 anb 系数数组中的索引。
        - self.numberOfInteriorFaces：内部面的总数量。
        - self.owners：长度为总面数的数组，存储每个面的所有者单元索引。
        - self.neighbours：长度为内部面数的数组，存储每个内部面的邻居单元索引。
        - self.elementNeighbours：列表的列表，存储每个单元的邻居单元索引。

        注意：
        - 该过程确保在组装全球矩阵时，可以通过 upperAnbCoeffIndex 和 lowerAnbCoeffIndex 快速定位每个内部面对应的 anb 系数在所有者和邻居单元的 anb 数组中的位置。
        - 这对于非结构化网格尤为重要，因为每个单元的邻居数量可能不同，anb 数组的长度也不同。
        """
        self.upperAnbCoeffIndex = np.zeros(self.numberOfInteriorFaces, dtype=np.int32)
        self.lowerAnbCoeffIndex = np.zeros(self.numberOfInteriorFaces, dtype=np.int32)
        for iFace in range(self.numberOfInteriorFaces):
            own = self.owners[iFace]
            nei = self.neighbours[iFace]
            
            # For owner cell
            own_neighbours = self.elementNeighbours[own]
            nei_in_own_anb_index = own_neighbours.index(nei)
            self.upperAnbCoeffIndex[iFace] = nei_in_own_anb_index
            
            # For neighbour cell
            nei_neighbours = self.elementNeighbours[nei]
            own_in_nei_anb_index = nei_neighbours.index(own)
            self.lowerAnbCoeffIndex[iFace] = own_in_nei_anb_index



    def cfdProcessNodeTopology(self):
        """Invokes the cfdInvertConnectivity method on self.elementNodes and self.faceNodes
        这段Python代码定义了两个方法，`cfdProcessNodeTopology` 和 `cfdInvertConnectivity`，它们用于处理和反转网格的拓扑连接性。以下是对这两个方法的详细解释：
        ### 方法：cfdProcessNodeTopology
        这个方法的目的是调用 `cfdInvertConnectivity` 方法来反转 `self.elementNodes` 和 `self.faceNodes` 的连接性，从而得到每个节点或面所连接的元素或面的索引。

        - `self.nodeElements`：反转 `self.elementNodes` 后得到的连接性数组，表示每个节点连接的元素。
        - `self.nodeFaces`：反转 `self.faceNodes` 后得到的连接性数组，表示每个节点连接的面。

        ### 方法：cfdInvertConnectivity
        这个方法接受一个连接性数组 `theConnectivityArray` 作为输入，并返回一个反转后的连接性数组。

        - `theInvertedSize`：用于确定反转连接性数组的大小，它是输入数组中的最大元素值加一。
        - `theInvertedConnectivityArray`：初始化为一个空列表的列表，其长度为 `theInvertedSize+1`。

        #### 反转连接性的过程：
        1. 首先，通过两层循环遍历 `theConnectivityArray`，确定数组中的最大元素值，以确定反转数组的大小。
        2. 然后，再次遍历 `theConnectivityArray`，对于每个子列表中的每个元素，将其索引添加到 `theInvertedConnectivityArray` 对应元素的列表中。这样，原始数组中的每个元素都指向了包含它的索引。

        #### 返回值：
        - 返回 `theInvertedConnectivityArray`，这是一个反转后的连接性数组，其中每个索引对应一个列表，列表中包含了原始数组中该索引所属的所有元素的索引。

        ### 示例：
        假设 `self.elementNodes` 如下所示：
        ```python
        self.elementNodes = [[1, 2], [2, 3], [4, 5]]
        ```
        调用 `self.nodeElements = self.cfdInvertConnectivity(self.elementNodes)` 将执行以下操作：
        - 计算 `theInvertedSize` 为 5（元素1、2、3、4、5中最大的是5）。
        - 填充 `theInvertedConnectivityArray`，使得结果为：
        ```python
        [
        [], [0],  # 节点1连接的元素索引是0
        [0, 1],  # 节点2连接的元素索引是0和1
        [], [1],  # 节点3连接的元素索引是1
        [2],     # 节点4连接的元素索引是2
        [2]      # 节点5连接的元素索引是2
        ]
        ```
        这样，`self.nodeElements` 就包含了每个节点连接的元素索引。

        ### 注意事项：
        - 这些方法在CFD网格处理中非常有用，因为它们帮助建立了网格的逆拓扑关系，这对于求解器构建矩阵和向量以及进行数据交换非常重要。
        - 代码中的 `cfdProcessNodeTopology` 方法假设 `self.elementNodes` 和 `self.faceNodes` 已经被正确初始化和填充。
        - `cfdInvertConnectivity` 方法是一个通用方法，可以应用于任何类型的连接性数组，不仅限于网格节点或元素。
        """
        self.nodeElements = self.cfdInvertConnectivity(self.elementNodes)
        self.nodeFaces = self.cfdInvertConnectivity(self.faceNodes) 
        
        
    def cfdInvertConnectivity(self,theConnectivityArray):

        """
        Returns an array with the inverted connectivty of an input array. 
        
        For example self.cfdInvertConnectivty(self.elementNodes) takes the elementNodes connectivity
        array and returns an inverted array with the elements belonging to each node. 

        """        
        
        theInvertedSize=0
        
        for i in range(len(theConnectivityArray)):
            for j in range(len(theConnectivityArray[i])):     
                theInvertedSize=max(theInvertedSize, int(theConnectivityArray[i][j]))
        
        theInvertedConnectivityArray = [[] for i in range(theInvertedSize+1)]
        
        for i in range(len(theConnectivityArray)):
            for j in range(len(theConnectivityArray[i])):
                theInvertedConnectivityArray[int(theConnectivityArray[i][j])].append(i)
    
        return theInvertedConnectivityArray        
        
        
    def cfdProcessGeometry(self):
        """This function processes the mesh geometry
        Calculate:
            -face centroids (faceCentroids)
            -face normal (Sf)
            -face areas (faceAreas)
        这段Python代码定义了一个名为`cfdProcessGeometry`的方法，用于处理网格的几何属性，包括计算面的质心、法向量、面积，以及单元的质心和体积。此外，还计算了一些与面和单元相关联的其他几何属性。以下是对这个方法的详细解释：

        1. **初始化几何属性数组**：
        - 方法开始时，初始化了一系列用于存储面和单元几何属性的数组，例如`self.faceWeights`、`self.faceCF`、`self.faceCf`、`self.faceFf`、`self.wallDist`、`self.wallDistLimited`、`self.elementCentroids`和`self.elementVolumes`。

        2. **计算面质心（faceCentroids）**：
        - 对于每个面，通过对其节点的质心进行平均来计算面的质心。

        3. **计算面法向量（Sf）和面面积（faceAreas）**：
        - 通过计算面的虚拟三角形（由面的节点和面的质心组成）来累加面法向量和面积。法向量通过叉积计算，面积通过法向量的模长计算。

        4. **优化计算**：
        - 使用`maxPoints`来确定最大的面节点数，以便为计算过程中的临时数组分配足够的空间。

        5. **计算单元质心（elementCentroids）和单元体积（elementVolumes）**：
        - 对于每个单元，通过对其面的质心进行加权平均来计算单元的质心。单元体积通过累加每个面的贡献来计算。

        6. **计算面与单元的相对几何属性**：
        - 计算面质心相对于单元质心的向量`self.faceCf`和`self.faceFf`，以及单元质心相对于面质心的向量`self.faceCF`。

        7. **计算面权重（faceWeights）**：
        - 面权重是根据面质心与单元质心之间的相对位置计算的，用于在有限体积法中进行插值和权重分配。

        8. **处理边界面的几何属性**：
        - 对于边界面，计算面质心相对于单元质心的距离，并将面权重设置为1，因为边界面没有邻居单元。

        9. **限制壁面距离**：
        - 对于边界面，计算并限制壁面距离，以确保在靠近壁面的地方使用适当的距离值。

        ### 注意事项：
        - 这段代码中使用了NumPy库进行矩阵运算和向量计算，以提高计算效率。
        - 方法中包含了两种计算面法向量和面积的方式：一种是使用优化的方法，另一种是纯Python的迭代方法（已注释）。优化的方法通过减少NumPy的`np.cross()`函数的调用次数来提高性能。
        - 代码中的计算考虑了内部面和边界面的不同处理方式，特别是在计算面权重和壁面距离时。

        `cfdProcessGeometry`方法在CFD网格处理中非常重要，因为它为后续的数值求解和网格分析提供了必要的几何信息。
        """

        ## Linear weight of distance from cell center to face
        self.faceWeights = np.zeros(self.numberOfFaces,dtype=float)
        self.faceCF = Q_(np.zeros((self.numberOfFaces, 3),dtype=float),dm.length_dim)
        self.faceCFn = np.zeros((self.numberOfFaces, 3),dtype=float)
        self.faceCf = Q_(np.zeros((self.numberOfFaces, 3),dtype=float),dm.length_dim)
        self.faceFf = Q_(np.zeros((self.numberOfFaces, 3),dtype=float),dm.length_dim)
        self.faceDist = Q_(np.zeros(self.numberOfFaces,dtype=float),dm.length_dim)
        self.wallDistLimited = Q_(np.zeros(self.numberOfFaces,dtype=float),dm.length_dim)
        self.geoDiff_f = Q_(np.zeros(self.numberOfFaces,dtype=float),dm.length_dim)

        self.faceEf = Q_(np.zeros((self.numberOfInteriorFaces, 3),dtype=float),dm.area_dim)
        self.faceTf = Q_(np.zeros((self.numberOfInteriorFaces, 3),dtype=float),dm.area_dim)

        self.elementCentroids = Q_(np.zeros((self.numberOfElements, 3),dtype=float),dm.length_dim)
        self.elementVolumes = Q_(np.zeros(self.numberOfElements,dtype=float),dm.volume_dim)

        self.faceCentroids = Q_(np.zeros((self.numberOfFaces, 3),dtype=float),dm.length_dim)  # 假设每个面心是一个3D向量
        self.faceSf = Q_(np.zeros((self.numberOfFaces, 3),dtype=float),dm.area_dim)         # 假设每个面法向量是一个3D向量
        self.faceAreas = Q_(np.zeros(self.numberOfFaces,dtype=float),dm.area_dim)           # 假设每个面面积是一个标量    
        
        #find cell with largest number of points
        # maxPoints=len(max(self.faceNodes, key=len))
        '''
        `maxPoints = len(max(self.faceNodes, key=len))` 这行代码执行以下操作：
        1. `self.faceNodes`：这是一个列表，其中每个元素本身也是一个列表，代表一个网格面的节点索引。

        2. `key=len`：这是一个函数，指定了 `max` 函数用来比较每个元素的标准，这里使用 `len` 函数获取每个子列表的长度。

        3. `max(self.faceNodes, key=len)`：`max` 函数找出 `self.faceNodes` 中长度最长的子列表。由于 `key=len` 的指定，`max` 函数基于子列表的长度来确定最大值。

        4. `len(max(self.faceNodes, key=len))`：对上一步得到的最长子列表使用 `len` 函数，得到这个子列表的长度，即包含节点索引最多的面的节点数量。

        5. `maxPoints`：将得到的最长子列表的长度赋值给变量 `maxPoints`。

        ### 示例：
        假设 `self.faceNodes` 如下所示：
        ```python
        self.faceNodes = [
            [0, 1, 2],       # 3个节点
            [3, 4],           # 2个节点
            [5, 6, 7, 8],     # 4个节点
            ...
        ]
        ```
        在这个例子中，有一个面的节点索引列表长度为4，这是最长的。因此，`max(self.faceNodes, key=len)` 将返回 `[5, 6, 7, 8]` 这个列表，然后 `len(...)` 将返回4。最终，`maxPoints` 将被赋值为4。

        这行代码的用途通常是确定网格中面节点数量最多的情况，以便为相关的计算过程分配适当的空间或初始化数据结构。在处理网格几何属性时，这有助于确保算法可以处理网格中任意形状的面。
        '''
        # forCross1 = [[] for i in range(maxPoints)]
        # forCross2 = [[] for i in range(maxPoints)]
        # local_faceCentroid=[[] for i in range(maxPoints)]
        
        # for iFace in range(self.numberOfFaces):
        #     theNodeIndices = self.faceNodes[iFace]
        #     theNumberOfFaceNodes = len(theNodeIndices)
            
        #     #compute a rough centre of the face
        #     local_centre = [0,0,0]
            
        #     for iNode in theNodeIndices:
        #         local_centre = local_centre + self.nodeCentroids[int(iNode)]
        
        #     local_centre = local_centre/theNumberOfFaceNodes
        
        #     for iTriangle in range(theNumberOfFaceNodes):
                
        #         point1 = local_centre
        #         point2 = self.nodeCentroids[int(theNodeIndices[iTriangle])]
                
        #         if iTriangle < theNumberOfFaceNodes-1:
        #             point3 = self.nodeCentroids[int(theNodeIndices[iTriangle+1])]
        #         else:
        #             point3 = self.nodeCentroids[int(theNodeIndices[0])]
                
        #         local_faceCentroid[iTriangle].append((point1+point2+point3)/3)
                
        #         left=point2-point1
        #         right=point3-point1
                
        #         forCross1[iTriangle].append(left)
        #         forCross2[iTriangle].append(right)
        
        # local_Sf=[np.zeros([self.numberOfFaces,3]) for i in range(maxPoints)]
        # local_area=[np.zeros([self.numberOfFaces,3]) for i in range(maxPoints)]
        # centroid=np.zeros([self.numberOfFaces,3])
        # area=np.zeros([self.numberOfFaces])
        # Sf=np.zeros([self.numberOfFaces,3])
        #cells with fewer faces than others are full of zeros
        # for i in range(maxPoints):  
        #     forCrossLeft=np.vstack(np.array(forCross1[i]))
        #     forCrossRight=np.vstack(np.array(forCross2[i]))
        #     '''
        #     在您提供的代码片段中，`np.vstack` 用于垂直堆叠数组，这对于后续的叉积计算是必要的。然而，如果 `forCross1[i]` 和 `forCross2[i]` 已经是二维数组（或列表的列表），且每行代表一个向量，那么在这种情况下，可能不需要使用 `np.vstack`。

        #     以下是两种情况的对比：

        #     1. **需要 `np.vstack`**：
        #     如果 `forCross1[i]` 和 `forCross2[i]` 是一维列表的列表，即每个元素是一个向量的组件，那么需要将它们转换为二维数组，以便每行代表一个向量。这时，`np.vstack` 是必要的，因为它将这些列表转换为适合进行 NumPy 操作的数组形式。

        #     ```python
        #     forCrossLeft = np.vstack(forCross1[i])
        #     forCrossRight = np.vstack(forCross2[i])
        #     ```

        #     2. **不需要 `np.vstack`**：
        #     如果 `forCross1[i]` 和 `forCross2[i]` 已经是二维数组，且每行是一个向量，那么可以直接使用 `np.cross` 计算叉积，不需要使用 `np.vstack`。

        #     ```python
        #     local_Sf[i] = 0.5 * np.cross(forCrossLeft, forCrossRight)
        #     ```

        #     在您的代码中，如果 `forCross1[i]` 和 `forCross2[i]` 已经具有正确的形状（即，每个元素是一个向量），那么可以省略 `np.vstack`。但是，如果它们是一维数组或列表，那么需要使用 `np.vstack` 或 `np.array` 来确保它们转换为正确的形状。

        #     请注意，如果选择省略 `np.vstack`，您可能需要确保其他相关的代码部分能够正确处理数组的形状和维度。如果其他部分代码依赖于 `np.vstack` 来确保数组的正确形状，那么简单地删除 `np.vstack` 可能会导致错误或不正确的结果。
        #     '''
            
        #     local_Sf[i]=0.5*np.cross(forCrossLeft,forCrossRight)
        #     local_area[i]=np.linalg.norm(local_Sf[i],axis=1)
        #     centroid = centroid + np.array(local_faceCentroid[i])*local_area[i][:,None]#用[:,None]也可
        #     Sf=Sf+local_Sf[i]
        #     area=area+local_area[i]
            
        # self.faceCentroids=centroid/area[:,None]
        # self.faceSf=Sf
        # self.faceAreas=area   
        
        
        """
        Pure python version - causes slowness due to iterative np.cross()
        """
        # self.faceCentroids= [[] for i in range(self.numberOfFaces)]
        # self.faceSf= [[] for i in range(self.numberOfFaces)]
        # self.faceAreas= [[] for i in range(self.numberOfFaces)]
        # self.faceCentroids = np.zeros((self.numberOfFaces, 3))  # 假设每个面心是一个3D向量
        # self.faceSf = np.zeros((self.numberOfFaces, 3))         # 假设每个面法向量是一个3D向量
        # self.faceAreas = np.zeros(self.numberOfFaces)           # 假设每个面面积是一个标量
        

        for iFace in range(self.numberOfFaces):
            theNodeIndices = self.faceNodes[iFace]
            theNumberOfFaceNodes = len(theNodeIndices)
            #compute a rough centre of the face
            local_centre = np.zeros(3)
            
            for iNode in theNodeIndices:
                local_centre = local_centre + self.nodeCentroids[int(iNode)]
        
            local_centre = local_centre/theNumberOfFaceNodes
            centroid = np.zeros(3)
            Sf = np.zeros(3)
            area = 0.0
            #finds area of virtual triangles and adds them to the find to find face area
            #and direction (Sf)
            for iTriangle in range(theNumberOfFaceNodes):
                point1 = local_centre
                point2 = self.nodeCentroids[int(theNodeIndices[iTriangle])]
                
                if iTriangle < theNumberOfFaceNodes-1:
                    point3 = self.nodeCentroids[int(theNodeIndices[iTriangle+1])]
                else:
                    point3 = self.nodeCentroids[int(theNodeIndices[0])]           
                local_centroid = (point1 + point2 + point3)/3
                
                left=point2-point1
                right=point3-point1
                # x = 0.5*((left[1] * right[2]) - (left[2] * right[1]))
                # y = 0.5*((left[2] * right[0]) - (left[0] * right[2]))
                # z = 0.5*((left[0] * right[1]) - (left[1] * right[0]))
                # local_Sf=np.array([x,y,z])
                local_Sf=0.5*np.cross(left, right)#右手系，x 叉乘 y=z，y轴朝上，复制的面在上层，见ParaView，法向量向外，由owner指向neighbour!!!
                local_area = np.linalg.norm(local_Sf)
                centroid +=  local_area*local_centroid
                Sf +=  local_Sf
                area +=  local_area
            centroid /= area
            self.faceCentroids.value[iFace]=centroid
            self.faceSf.value[iFace]=Sf
            self.faceAreas.value[iFace]=area
        
        self.facen=mth.cfdUnit(self.faceSf.value)
        """
        Calculate:
            -element centroids (elementCentroids)
            -element volumes (elementVolumes)
        """
        for iElement in range(self.numberOfElements):
            
            theElementFaces = self.elementFaces[iElement]
            
            #compute a rough centre of the element
            local_centre = np.zeros(3)
            
            for iFace in range(len(theElementFaces)):
                faceIndex = theElementFaces[iFace]
                local_centre += self.faceCentroids.value[faceIndex]
            
            local_centre /= len(theElementFaces)
            
            localVolumeCentroidSum = np.zeros(3)
            localVolumeSum = 0.0
            
            for iFace in range(len(theElementFaces)):
                faceIndex = theElementFaces[iFace]
                
                Cf = self.faceCentroids.value[faceIndex]-local_centre
                
                faceSign = -1
                if iElement == self.owners[faceIndex]:
                    faceSign = 1
                    
                local_Sf = faceSign*self.faceSf.value[faceIndex]
                
                localVolume = np.dot(local_Sf,Cf)/3
                
                localCentroid = 0.75*self.faceCentroids.value[faceIndex]+0.25*local_centre
                
                localVolumeCentroidSum +=  localCentroid*localVolume
                
                localVolumeSum +=  localVolume
                
            self.elementCentroids.value[iElement]=localVolumeCentroidSum/localVolumeSum
            self.elementVolumes.value[iElement]=localVolumeSum
        
        #计算内部面与单元的相对几何属性
        n=self.facen[:self.numberOfInteriorFaces]
        own=self.owners[:self.numberOfInteriorFaces]
        nei=self.neighbours[:self.numberOfInteriorFaces]
        self.faceCF[:self.numberOfInteriorFaces]=self.elementCentroids[nei]-self.elementCentroids[own]
        self.faceDist.value[:self.numberOfInteriorFaces]=np.linalg.norm(self.faceCF.value[:self.numberOfInteriorFaces],axis=1)
        nE= self.faceCF.value[:self.numberOfInteriorFaces]/self.faceDist.value[:self.numberOfInteriorFaces][:, np.newaxis]
        self.faceCFn[:self.numberOfInteriorFaces]=nE
        if self.OrthogonalCorrectionMethod=='Minimum':
            facemagEf=np.dot(self.faceSf.value[:self.numberOfInteriorFaces],nE)
            self.geoDiff_f.value[:self.numberOfInteriorFaces]=facemagEf/self.faceDist.value[:self.numberOfInteriorFaces]
            self.faceEf.value[:self.numberOfInteriorFaces]=facemagEf[:, np.newaxis]*nE
        elif self.OrthogonalCorrectionMethod=='Orthogonal':
            self.faceEf.value[:self.numberOfInteriorFaces]=self.faceAreas.value[:self.numberOfInteriorFaces, np.newaxis]*nE
            self.geoDiff_f.value[:self.numberOfInteriorFaces]=self.faceAreas.value[:self.numberOfInteriorFaces]/self.faceDist.value[:self.numberOfInteriorFaces]
        elif self.OrthogonalCorrectionMethod=='OverRelaxed':
            facemagEf=self.faceAreas.value[:self.numberOfInteriorFaces]*self.faceAreas.value[:self.numberOfInteriorFaces]/mth.cfdDot(self.faceSf.value[:self.numberOfInteriorFaces],nE)
            self.geoDiff_f.value[:self.numberOfInteriorFaces]=facemagEf/self.faceDist.value[:self.numberOfInteriorFaces]
            self.faceEf.value[:self.numberOfInteriorFaces]=facemagEf[:, np.newaxis]*nE
        else:
            io.cfdError('Region.mesh.OrthogonalCorrectionMethod not exist')
        self.faceTf[:self.numberOfInteriorFaces]=self.faceSf[:self.numberOfInteriorFaces]-self.faceEf[:self.numberOfInteriorFaces]
        self.faceCf[:self.numberOfInteriorFaces]=self.faceCentroids[:self.numberOfInteriorFaces]-self.elementCentroids[own]
        self.faceFf[:self.numberOfInteriorFaces]=self.faceCentroids[:self.numberOfInteriorFaces]-self.elementCentroids[nei]
        self.faceWeights[:self.numberOfInteriorFaces]=(-mth.cfdDot(self.faceFf.value[:self.numberOfInteriorFaces],n))/(-mth.cfdDot(self.faceFf.value[:self.numberOfInteriorFaces],n)+mth.cfdDot(self.faceCf.value[:self.numberOfInteriorFaces],n))

        #计算外部面与单元的相对几何属性
        n=self.facen[self.numberOfInteriorFaces:]
        self.faceCFn[self.numberOfInteriorFaces:]=n
        own=self.owners[self.numberOfInteriorFaces:]
        self.faceCF[self.numberOfInteriorFaces:]=self.faceCentroids[self.numberOfInteriorFaces:]-self.elementCentroids[own]
        self.faceCf[self.numberOfInteriorFaces:]=self.faceCF[self.numberOfInteriorFaces:]
        self.faceWeights[self.numberOfInteriorFaces:]=1
        self.faceDist.value[self.numberOfInteriorFaces:]=mth.cfdDot(self.faceCf.value[self.numberOfInteriorFaces:], n)
        self.geoDiff_f[self.numberOfInteriorFaces:]=self.faceAreas[self.numberOfInteriorFaces:]/self.faceDist[self.numberOfInteriorFaces:]
        self.wallDistLimited.value[self.numberOfInteriorFaces:]=np.maximum(self.faceDist.value[self.numberOfInteriorFaces:], 0.05*np.linalg.norm(self.faceCf.value[self.numberOfInteriorFaces:],axis=1))

        # 确保权重在合理范围内
        if not np.all((self.faceWeights >= 0) & (self.faceWeights <= 1)):
            io.cfdError('Interpolation weights out of bounds: some g_f values are not between 0 and 1')

    def cfdGetBoundaryElementsSubArrayForBoundaryPatch(self):
        """
        Creates a list of the boundary elements pertaining to a patch in self.cfdBoundaryPatchesArray
        这段Python代码定义了一个名为`cfdGetBoundaryElementsSubArrayForBoundaryPatch`的方法，它用于创建一个列表，列出了属于某个边界补丁的所有边界元素的索引。这个方法是`Polymesh`类的一部分，用于处理网格的边界补丁信息。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdGetBoundaryElementsSubArrayForBoundaryPatch(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **遍历边界补丁**：
        - `for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():` 遍历`self.cfdBoundaryPatchesArray`字典，这个字典包含了所有边界补丁的信息。

        3. **计算边界元素的起始索引**：
        - `startBElement = self.numberOfElements + self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex'] - self.numberOfInteriorFaces` 这行代码计算了边界元素的起始索引。这里假设`self.numberOfElements`是网格中单元的总数，`self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']`是边界补丁在面列表中的起始索引，`self.numberOfInteriorFaces`是内部面的数量。

        4. **计算边界元素的结束索引**：
        - `endBElement = startBElement + self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']` 这行代码计算了边界元素的结束索引，通过将起始索引与该补丁的边界面数量相加。

        5. **创建边界元素索引列表**：
        - `self.cfdBoundaryPatchesArray[iBPatch]['iBElements'] = list(range(int(startBElement), int(endBElement)))` 这行代码使用`range`函数创建了一个从`startBElement`到`endBElement`的整数序列，并将这个序列转换为一个列表，赋值给对应边界补丁的`'iBElements'`键。

        ### 注意事项：
        - 这个方法假设`self.cfdBoundaryPatchesArray`字典已经被正确初始化，并且包含了每个边界补丁的`'startFaceIndex'`和`'numberOfBFaces'`信息。
        - `startBElement`的计算可能看起来有些复杂，因为它考虑了从内部面到边界面的转换。这种转换取决于网格数据的具体组织方式。
        - `range`函数在Python 3中返回一个迭代器，而不是列表。因此，通过使用`list()`函数，我们可以将这个迭代器转换为列表。
        - 代码中的`int()`函数确保了索引是整数类型，这对于列表索引和数组索引是必要的。

        这个方法是CFD网格处理的一部分，用于确定哪些网格元素位于边界上，这对于应用边界条件和分析边界效应非常重要。
        """
        for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():
            
            startBElement=self.numberOfElements+self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']-self.numberOfInteriorFaces
            endBElement=startBElement+self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        
            self.cfdBoundaryPatchesArray[iBPatch]['iBElements']=np.arange(int(startBElement), int(endBElement))
            # list(range(int(startBElement),int(endBElement)))

    def cfdGetFaceCentroidsSubArrayForBoundaryPatch(self):
        """
        Creates a list with the centroids for each face of the patch in self.cfdBoundaryPatchesArray[patch]['faceCentroids']
        这段Python代码定义了一个名为`cfdGetFaceCentroidsSubArrayForBoundaryPatch`的方法，它是`Polymesh`类的一部分，用于为每个边界补丁创建面质心的子数组。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdGetFaceCentroidsSubArrayForBoundaryPatch(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **遍历边界补丁**：
        - `for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():` 遍历`self.cfdBoundaryPatchesArray`字典，这个字典包含了所有边界补丁的信息。

        3. **计算边界面的起始和结束索引**：
        - `startBFace = self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']` 获取边界补丁的起始面索引。
        - `endBFace = startBFace + self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']` 计算边界补丁的结束面索引。

        4. **创建边界面索引列表**：
        - `iBFaces = list(range(int(startBFace), int(endBFace)))` 使用`range`函数创建一个从`startBFace`到`endBFace`的整数序列，并将这个序列转换为列表。

        5. **创建面质心子数组**：
        - `self.cfdBoundaryPatchesArray[iBPatch]['faceCentroids'] = [self.faceCentroids[i] for i in iBFaces]` 使用列表推导式，通过遍历`iBFaces`列表中的每个索引`i`，从`self.faceCentroids`中获取对应的面质心，并创建一个新的列表。

        6. **存储结果**：
        - 将创建的面质心列表赋值给`self.cfdBoundaryPatchesArray[iBPatch]['faceCentroids']`，这样每个边界补丁就有了自己的面质心列表。

        ### 注意事项：
        - 这个方法假设`self.cfdBoundaryPatchesArray`字典已经被正确初始化，并且包含了每个边界补丁的`'startFaceIndex'`和`'numberOfBFaces'`信息。
        - `self.faceCentroids`是一个存储所有面质心的列表或数组，其中每个元素是一个包含三维坐标的元组或列表。
        - 使用`int()`函数确保了索引是整数类型，这对于列表索引和数组索引是必要的。
        - 列表推导式是一种Pythonic的方式，用于根据现有列表中的元素创建新列表。

        这个方法是CFD网格处理的一部分，用于确定边界补丁上每个面的质心，这对于应用边界条件和分析边界效应非常重要。通过为每个边界补丁创建面质心的子数组，可以方便地访问和操作与特定边界补丁相关的几何信息。
        """
        
        for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():
            # startBFace=self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']
            # endBFace=startBFace+self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
            iBFaces=self.cfdBoundaryPatchesArray[iBPatch]['iBFaces'] 
            self.cfdBoundaryPatchesArray[iBPatch]['faceCentroids']=self.faceCentroids[iBFaces]
            # [self.faceCentroids[i] for i in iBFaces]

    def cfdGetOwnersSubArrayForBoundaryPatch(self):
        """
        Creates a list with the element owners of each boundary patch face in self.cfdBoundaryPatchesArray[patch]['owners_b']
        这段Python代码定义了一个名为 `cfdGetOwnersSubArrayForBoundaryPatch` 的方法，它是 `Polymesh` 类的一部分，用于为每个边界补丁创建面的所有者（即拥有该面的单元）的子数组。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdGetOwnersSubArrayForBoundaryPatch(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **遍历边界补丁**：
        - `for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():` 遍历 `self.cfdBoundaryPatchesArray` 字典，这个字典包含了所有边界补丁的信息。

        3. **计算边界面的起始和结束索引**：
        - `startBFace = self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']` 获取边界补丁的起始面索引。
        - `endBFace = startBFace + self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']` 计算边界补丁的结束面索引。

        4. **创建边界面索引列表**：
        - `iBFaces = list(range(int(startBFace), int(endBFace)))` 使用 `range` 函数创建一个从 `startBFace` 到 `endBFace` 的整数序列，并将这个序列转换为列表。

        5. **创建面的所有者子数组**：
        - `self.cfdBoundaryPatchesArray[iBPatch]['owners_b'] = [self.owners[i] for i in iBFaces]` 使用列表推导式，通过遍历 `iBFaces` 列表中的每个索引 `i`，从 `self.owners` 中获取对应的面的所有者，并创建一个新的列表。

        6. **存储结果**：
        - 将创建的面的所有者列表赋值给 `self.cfdBoundaryPatchesArray[iBPatch]['owners_b']`，这样每个边界补丁就有了自己的面的所有者列表。

        ### 注意事项：
        - 这个方法假设 `self.cfdBoundaryPatchesArray` 字典已经被正确初始化，并且包含了每个边界补丁的 `'startFaceIndex'` 和 `'numberOfBFaces'` 信息。
        - `self.owners` 是一个存储所有面的所有者单元索引的列表或数组。
        - 使用 `int()` 函数确保了索引是整数类型，这对于列表索引和数组索引是必要的。
        - 列表推导式是一种Pythonic的方式，用于根据现有列表中的元素创建新列表。

        这个方法是CFD网格处理的一部分，用于确定边界补丁上每个面的所有者单元，这对于应用边界条件和分析边界效应非常重要。通过为每个边界补丁创建面的所有者单元的子数组，可以方便地访问和操作与特定边界补丁相关的拓扑信息。
        """
        
        for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():
            
            # startBFace=self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']
            
            # endBFace=startBFace+self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        
            iBFaces=self.cfdBoundaryPatchesArray[iBPatch]['iBFaces']   
            
            self.cfdBoundaryPatchesArray[iBPatch]['owners_b']=self.owners[iBFaces]
            # [self.owners[i] for i in iBFaces]

    def cfdGetFaceSfSubArrayForBoundaryPatch(self):
        """
        Creates a list with the surface vectors (Sf) of each face belonging to each boundary patch in self.cfdBoundaryPatchesArray[patch]['facesSf']
        这段Python代码定义了一个名为`cfdGetFaceSfSubArrayForBoundaryPatch`的方法，它是`Polymesh`类的一部分，用于为每个边界补丁创建面法向量（通常表示为`Sf`）的子数组。以下是对这个方法的详细解释：

        1. **方法定义**：
        - `def cfdGetFaceSfSubArrayForBoundaryPatch(self):` 定义了一个实例方法，没有接收额外的参数。

        2. **遍历边界补丁**：
        - `for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():` 遍历`self.cfdBoundaryPatchesArray`字典，这个字典包含了所有边界补丁的信息。

        3. **计算边界面的起始和结束索引**：
        - `startBFace = self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']` 获取边界补丁的起始面索引。
        - `endBFace = startBFace + self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']` 计算边界补丁的结束面索引。

        4. **创建边界面索引列表**：
        - `iBFaces = list(range(int(startBFace), int(endBFace)))` 使用`range`函数创建一个从`startBFace`到`endBFace`的整数序列，并将这个序列转换为列表。

        5. **创建面法向量子数组**：
        - `self.cfdBoundaryPatchesArray[iBPatch]['facesSf'] = [self.faceSf[i] for i in iBFaces]` 使用列表推导式，通过遍历`iBFaces`列表中的每个索引`i`，从`self.faceSf`中获取对应的面法向量，并创建一个新的列表。

        6. **将列表转换为NumPy数组**：
        - `self.cfdBoundaryPatchesArray[iBPatch]['facesSf'] = np.asarray(self.cfdBoundaryPatchesArray[iBPatch]['facesSf'])` 使用`np.asarray`将创建的面法向量列表转换为NumPy数组。

        7. **存储结果**：
        - 将转换后的NumPy数组赋值给`self.cfdBoundaryPatchesArray[iBPatch]['facesSf']`，这样每个边界补丁就有了自己的面法向量NumPy数组。

        ### 注意事项：
        - 这个方法假设`self.cfdBoundaryPatchesArray`字典已经被正确初始化，并且包含了每个边界补丁的`'startFaceIndex'`和`'numberOfBFaces'`信息。
        - `self.faceSf`是一个存储所有面法向量的列表或数组，其中每个元素通常是一个包含三维坐标的向量。
        - 使用`int()`函数确保了索引是整数类型，这对于列表索引和数组索引是必要的。
        - `np.asarray`用于确保数据结构是NumPy数组，这通常在进行科学计算时提供更好的性能和更多的功能。

        这个方法是CFD网格处理的一部分，用于确定边界补丁上每个面的法向量，这对于应用边界条件和分析边界效应非常重要。通过为每个边界补丁创建面法向量的NumPy数组，可以方便地进行后续的数值计算和处理。
        """

        for iBPatch, theBCInfo in self.cfdBoundaryPatchesArray.items():
            
            startBFace=self.cfdBoundaryPatchesArray[iBPatch]['startFaceIndex']
            
            endBFace=startBFace+self.cfdBoundaryPatchesArray[iBPatch]['numberOfBFaces']
        
            iBFaces=np.arange(int(startBFace),int(endBFace))

            self.cfdBoundaryPatchesArray[iBPatch]['iBFaces']=iBFaces 
            
            self.cfdBoundaryPatchesArray[iBPatch]['facesSf']=self.faceSf[iBFaces]# np.asarray([self.faceSf[i] for i in iBFaces])

            self.cfdBoundaryPatchesArray[iBPatch]['facen']=self.facen[iBFaces] #np.asarray([self.facen[i] for i in iBFaces])

            self.cfdBoundaryPatchesArray[iBPatch]['normSb']=self.faceAreas[iBFaces] #np.asarray([self.faceAreas[i] for i in iBFaces])


#    def cfdGetOwnersSubArrayForInteriorFaces(self):
#        
#        """Returns owners sub-array for interior faces
#        """
#
#        self.owners_f=self.owners[0:self.numberOfInteriorFaces]         
#        return self.owners_f
#    
#    def cfdGetNeighboursSubArrayForInteriorFaces(self):
#        
#        """Returns neighbours sub-array for interior faces
#        """
#
#        self.neighbours_f=self.neighbours[0:self.numberOfInteriorFaces]         
#        return self.neighbours_f
    
    
