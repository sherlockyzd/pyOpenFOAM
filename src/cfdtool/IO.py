import os
import re
import sys
#import os.path

def cfdError(*args):
    if args:
        str=args[0]
        print(str)
    else:
        print('--------------------Error!!--------------------------\n')
    sys.exit()

def cfdPrintMainHeader():
    print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-* pyFVM *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')
    print('|| A python finite volume code based heavily on the uFVM code written in    ||\n')
    print('|| the Matlab language.                                                     ||\n')
    print('|| uFVM was written by the CFD Group at the American University of Beirut.  ||\n')
    print('|| This is an academic CFD package developed for learning purposes to serve || \n')
    print('|| the student community.                                                   ||\n')
    print('----------------------------------------------------------------------\n')
    print(' Credits:\n \tMarwan Darwish, Mhamad Mahdi Alloush for uFVM code\n')
    print('\tcfd@aub.edu.lb\n')
    print('\tAmerican University of Beirut\n')
    print('\tuFVM v1.5, 2018\n')
    print('Python version credits:\n \tYuZhengdong  for Python translation\n')
    print('\tAuthor is PHD_doctor at Tsinghua University\n')
    print('\tBeijing, China')
    print('\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\n')


def cfdPrintHeader():
    print('\n\n')
    print('||*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* pyFVM *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*||')
    print('||                                                                             ||')
    print('||                             Starting simulation                             ||')
    print('||                                                                             ||')
    print('||*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*||\n')

def MomentumPrintIteration(iComponent):
    print('|==========================================================================|\n')
    print('                     Momentum Iteration Component part %d \n'%iComponent)
    print('|--------------------------------------------------------------------------|\n')

def ContinuityPrintIteration():
    print('|==========================================================================|\n')
    print('                     Continuity Iteration\n')
    print('|--------------------------------------------------------------------------|\n')

def ScalarTransportPrintIteration():
    print('|==========================================================================|\n')
    print('                     ScalarTransport Iteration \n' )
    print('|--------------------------------------------------------------------------|\n')

def cfdPrintIteration(theEquationName,iterationNumber,*args):
    # print('|==========================================================================|\n')
    if args:
        iComponent=args[0]
        print('       The %s equation Component part %d Iteration %d\n' %(theEquationName,iComponent,iterationNumber))
    else:
        print('       The %s equation Iteration %d\n' %(theEquationName,iterationNumber))
    # print('|--------------------------------------------------------------------------|\n')

def cfdPrintResidualsHeader(theEquationName,tolerance,maxIter,initRes,finalRes):
    print('|--------------------------------------------------------------------------|\n')
    print('|  Equation  |     Tolerance     |     Maxiter     | initialResidual | finalResidual |\n')
    print('|--------------------------------------------------------------------------|\n')
    print('|---The %s equation---Tol:%e---Max:%d---init:%e---final:%e----|\n' % (theEquationName,tolerance,maxIter,initRes,finalRes))


def cfdGetFoamFileHeader(fieldFilePath):
    """
    这段Python代码定义了一个名为`cfdGetFoamFileHeader`的函数，它用于读取和解析OpenFOAM格式的文件头信息。下面是对这段代码的详细解释：
    1. `def cfdGetFoamFileHeader(fieldFilePath):`：定义了一个函数`cfdGetFoamFileHeader`，它接收一个参数`fieldFilePath`，这个参数是文件的路径。
    2. `with open(fieldFilePath, "r") as fpid:`：使用`with`语句打开文件，文件路径由`fieldFilePath`提供，以只读模式（"r"）打开。文件对象被赋予变量名`fpid`。
    3. `print('Reading %s file ...' % (fieldFilePath))`：打印一条消息，告知用户正在读取文件。
    4. `header={}`：初始化一个空字典`header`，用来存储解析出的文件头信息。
    5. `for linecount, tline in enumerate(fpid):`：使用`enumerate`函数遍历文件对象`fpid`的每一行，同时获取行号`linecount`和行内容`tline`。
    6. `if not cfdSkipEmptyLines(tline):`：调用`cfdSkipEmptyLines`函数检查当前行`tline`是否为空行或只包含空白字符。如果是，则`not cfdSkipEmptyLines`返回`True`，执行`continue`跳过当前循环的剩余部分。
    7. `if not cfdSkipMacroComments(tline):`：调用`cfdSkipMacroComments`函数检查当前行`tline`是否是宏定义注释。如果是，则`not cfdSkipMacroComments`返回`True`，同样执行`continue`跳过当前循环的剩余部分。
    8. `if "FoamFile" in tline:`：检查当前行`tline`中是否包含字符串"FoamFile"。如果包含，说明找到了文件的开始部分。
    9. `if not header:`：如果此时`header`字典仍然为空，说明还没有读取到任何头信息。
    10. `header=cfdReadCfdDictionary(fpid)`：调用`cfdReadCfdDictionary`函数，传入文件对象`fpid`，从文件中读取并解析OpenFOAM格式的字典信息，并将解析结果赋值给`header`字典。
    11. `return header`：返回解析得到的头信息字典`header`。
    总结来说，这个函数的目的是打开一个指定路径的OpenFOAM文件，跳过空行和宏定义注释，找到包含"FoamFile"的行，并从该行开始读取和解析文件的头信息，最后返回这个头信息的字典。注意，`cfdSkipEmptyLines`、`cfdSkipMacroComments`和`cfdReadCfdDictionary`这些函数在其他地方定义的辅助函数，用于处理特定的文件解析任务。
    """
    with open(fieldFilePath,"r") as fpid:
        print('Reading %s file ...' %(fieldFilePath))
       
        header={}
        for linecount, tline in enumerate(fpid):
            
            if not cfdSkipEmptyLines(tline):
                continue
            
            if not cfdSkipMacroComments(tline):
                continue
            
            if "FoamFile" in tline:
                if not header:
                    header=cfdReadCfdDictionary(fpid)
                    return header

def cfdSkipEmptyLines(tline):
    """
    这段Python代码定义了一个名为`cfdSkipEmptyLines`的函数，其作用是检查传入的字符串`tline`是否为空行或者只包含空白字符（如空格、制表符等）。下面是对这段代码的详细解释：
    1. `def cfdSkipEmptyLines(tline):`：定义了一个函数`cfdSkipEmptyLines`，它接收一个参数`tline`，这个参数应该是一个字符串，代表文件中的一行文本。
    2. `if not tline.strip():`：使用`strip()`方法移除字符串`tline`两端的空白字符（包括空格、制表符、换行符等），然后使用`not`操作符检查结果是否为空字符串。如果`tline`原本是空行或只包含空白字符，`strip()`方法会返回一个空字符串，`if`语句的条件为真。
    3. `tline = False`：如果条件为真，即`tline`是空行或只包含空白字符，将`tline`的值设置为`False`。这里的`False`用于表示这行应该被跳过，不进行进一步的处理。
    4. `else:`：如果`if`条件不满足，即`tline`不是空行或只包含空白字符，执行`else`块中的代码。
    5. `tline = tline`：在`else`块中，将`tline`的值重新赋给它自己。这一步实际上是多余的，因为`tline`的值在进入`else`块之前并没有改变。它可能是为了代码的可读性或与后续逻辑保持一致性。
    6. `return tline`：函数返回`tline`的值。如果`tline`是空行或只包含空白字符，返回`False`；否则返回原始的`tline`字符串。
    总结来说，`cfdSkipEmptyLines`函数的目的是判断传入的字符串是否为空行或只包含空白字符，如果是，则返回`False`，表示这行应该被跳过；如果不是，则返回原始字符串，表示这行应该被进一步处理。在实际使用中，这个函数通常与文件读取循环结合使用，以过滤掉文件中的空行和只包含空白字符的行。
    """
    if not tline.strip():
        tline = False
    else:
        tline = tline
    return tline

def cfdSkipMacroComments(tline):
    """
    这段Python代码定义了一个名为`cfdSkipMacroComments`的函数，其目的是检查并跳过文件中的宏定义注释行。下面是对这段代码的详细解释：
    1. `def cfdSkipMacroComments(tline):`：定义了一个函数`cfdSkipMacroComments`，接收一个参数`tline`，这个参数是一个字符串，代表文件中的一行文本。
    2. `trimmedTline = tline.strip()`：使用`strip()`方法移除字符串`tline`两端的空白字符，并将结果保存在变量`trimmedTline`中。
    3. `if "/*" in trimmedTline:`：检查经过`strip()`处理后的字符串`trimmedTline`中是否包含`"/*"`。这通常是C语言或类似语言中宏定义的开始标记。如果是，则将`tline`的值设置为`False`，表示这行是宏定义的开始，应该被跳过。
    4. `elif "|" in trimmedTline:`：检查`trimmedTline`中是否包含`"|"`。这个条件的具体意义不太明确，因为`"|"`通常不是宏定义的一部分。可能是特定上下文中的特定要求。
    5. `elif "\*" in trimmedTline:`：检查`trimmedTline`中是否包含`"\*"`（即转义的星号字符）。这个条件同样不太明确，因为通常星号`"*"`用于表示乘法操作或注释的结束，而不是注释本身。
    6. `elif "*" in trimmedTline:`：检查`trimmedTline`中是否包含未转义的星号`"*"`。这可能是为了跳过包含星号的注释行，但通常星号注释应该是以`"/*"`开始和`"*/"`结束的块注释。
    7. `else:`：如果以上条件都不满足，即`trimmedTline`既不是以`"/*"`开始的宏定义，也不包含`"|"`或星号`"*"`，则执行`else`块中的代码。
    8. `tline = tline`：在`else`块中，将`tline`的值重新赋给它自己。这一步实际上是多余的，因为`tline`的值在进入`else`块之前并没有改变。
    9. `return tline`：函数返回`tline`的值。如果`tline`是宏定义注释行或包含某些特定字符，返回`False`；否则返回原始的`tline`字符串。
    总结来说，`cfdSkipMacroComments`函数的目的是检查传入的字符串是否是宏定义注释行或包含某些特定字符，如果是，则返回`False`，表示这行应该被跳过；如果不是，则返回原始字符串，表示这行应该被进一步处理。这个函数通常与文件读取循环结合使用，以过滤掉文件中的宏定义注释行和其他特定行。
    """
    trimmedTline = tline.strip()
    
    if "/*" in trimmedTline:
        tline = False
    elif "|" in trimmedTline:
        tline = False
    elif "\*" in trimmedTline:
        tline = False
    elif "*" in trimmedTline: 
        tline = False
    else:
        tline = tline
    return tline

def cfdReadCfdDictionary(fpid,**kwargs):
    """
    这段Python代码定义了一个名为`cfdReadCfdDictionary`的函数，用于解析OpenFOAM字典文件。OpenFOAM是一个用于计算流体动力学（CFD）的开源软件。这个函数读取文件对象`fpid`中的字典内容，并将其存储在一个Python字典`dictionary`中。下面是对这段代码的详细解释：
    1. `def cfdReadCfdDictionary(fpid, **kwargs):`：定义了函数`cfdReadCfdDictionary`，它接收两个参数：文件对象`fpid`和一个关键字参数列表`kwargs`。
    2. `subDictionary=False`：初始化一个布尔变量`subDictionary`，用来标记是否正在解析子字典。
    3. `dictionary={}`：初始化一个空字典`dictionary`，用来存储解析出的字典内容。
    4. `if 'line' in kwargs:`：检查`kwargs`中是否包含关键字参数`line`。
    - `dictionary[kwargs.get("line")[0]]=kwargs.get("line")[1]`：如果存在，将`kwargs["line"]`的第一个元素作为键，第二个元素作为值，添加到`dictionary`中。
    5. `for line, tline in enumerate(fpid):`：遍历文件对象`fpid`的每一行，同时获取行号`line`和行内容`tline`。
    6. `if not cfdSkipEmptyLines(tline):` 和 `if not cfdSkipMacroComments(tline):`：调用之前定义的函数`cfdSkipEmptyLines`和`cfdSkipMacroComments`来跳过空行和宏定义注释行。如果这些函数返回`True`，则使用`continue`跳过当前循环的剩余部分。
    7. `if "{" in tline:`：如果行中包含`"{"`，表示开始一个新的字典或子字典，但这里选择`continue`跳过，可能是因为这个函数不处理嵌套字典。
    8. `if "}" in tline and subDictionary == True:`：如果行中包含`"}"`并且当前正在解析子字典，则将`subDictionary`设置为`False`，表示子字典结束。
    9. `if "}" in tline and subDictionary == False:`：如果行中包含`"}"`并且当前不在解析子字典，则使用`break`退出循环，因为这意味着整个字典解析完成。
    10. `tline = tline.replace(";", "")`：移除行中的分号，因为OpenFOAM字典中的分号通常用于注释，这里将其从行内容中删除。
    11. `if len(tline.split()) == 1 and subDictionary == False:`：如果行只包含一个单词并且不在解析子字典，则认为这行是一个新的子字典的开始。
        - `subDictionary=True`：设置`subDictionary`为`True`。
        - `dictionary[tline.split()[0]]={}`：在`dictionary`中为这个子字典创建一个新的空字典。
        - `currentSubDictKey=tline.split()[0]`：将当前子字典的键保存在`currentSubDictKey`中。
    12. `if subDictionary == True:`：如果当前正在解析子字典，则解析当前行作为子字典中的键值对。
        - 尝试将值转换为浮点数，如果转换失败，则保持原样。
    13. `else:`：如果不在解析子字典，则将当前行解析为主字典中的键值对，同样尝试转换值为浮点数，如果失败则保持原样。
    14. `return dictionary`：函数返回解析得到的字典。

    总结来说，`cfdReadCfdDictionary`函数通过逐行读取文件对象`fpid`，跳过空行和注释，解析OpenFOAM字典文件中的键值对，并将它们存储在Python字典`dictionary`中。如果遇到子字典，它会递归地解析子字典内容。最后，函数返回这个字典。
    """
    subDictionary=False
    dictionary={}
    
    if 'line' in kwargs:
        dictionary[kwargs.get("line")[0]]=kwargs.get("line")[1]
        
    for line, tline in enumerate(fpid):
        
        if not cfdSkipEmptyLines(tline):
            continue
        
        if not cfdSkipMacroComments(tline):
            continue            
        
        if "{" in tline:
            continue

        #check for end of subDictionary
        if "}" in tline and subDictionary == True:
            subDictionary = False
            continue
        
        if "}" in tline and subDictionary == False:
            break

        '''
        这段代码是Python中的一个条件语句，用于处理特定的字典合并逻辑。以下是对这段代码的详细解释：

        1. **条件判断**：
        - `if len(tline.split()) == 1`：检查经过`split()`处理后的列表长度是否为1，这意味着`tline`可能只包含一个单词或数值。
        - `'$' in tline`：检查处理后的`tline`字符串中是否包含字符`'$'`。
        - `subDictionary == True`：检查变量`subDictionary`是否为`True`，表示当前是否在处理一个子字典。

        2. **处理字典键**：
        - 如果上述条件满足，从`tline`中移除`'$'`字符、换行符`'\n'`、分号`';'`，然后使用`strip()`方法去除可能的首尾空白字符，得到字典键`DictKey`。

        3. **合并字典**：
        - `dictionary[currentSubDictKey]={**dictionary[currentSubDictKey], **dictionary[DictKey]}`：使用字典解包（`**`）来合并`dictionary[currentSubDictKey]`和`dictionary[DictKey]`的键值对。这意味着`DictKey`对应的字典内容将被合并到`currentSubDictKey`对应的字典中。

        4. **注释掉的循环**：
        - 被注释掉的循环代码提供了另一种合并字典的方法，使用`setdefault()`方法来确保`currentSubDictKey`字典中包含`DictKey`字典的所有键值对。

        5. **继续语句**：
        - `continue`：在处理完当前的合并逻辑后，跳过当前循环的剩余部分，开始处理下一行。

        这段代码的目的是在一个字典中合并由特定标记（如包含`'$'`）指示的子字典。它通过检查特定条件来确定何时进行合并，并使用Python的字典解包特性来实现键值对的合并。这种方法在处理配置文件、数据合并等场景中非常有用。

        '''
        if len(tline.split()) == 1 and '$' in tline and subDictionary == True:
            DictKey=tline.replace("$", "").replace("\n", "").replace(";", "").strip()
            dictionary[currentSubDictKey]={**dictionary[currentSubDictKey], **dictionary[DictKey]}
            # for key in dictionary[DictKey]:
            #     dictionary[currentSubDictKey].setdefault(key, dictionary[DictKey][key])
            continue

        tline = tline.replace(";", "")
        
        if len(tline.split()) == 1 and subDictionary == False:
            
            subDictionary=True
            dictionary[tline.split()[0]]={}
            currentSubDictKey=tline.split()[0]
            continue
        
        if subDictionary == True:
            try:
                dictionary[currentSubDictKey][tline.split()[0]]=float(tline.split()[1])

            except ValueError:
                lenth=len(tline.split())
                dictionary[currentSubDictKey][tline.split()[0]]=' '.join(tline.split()[1:lenth])
            continue
        else:
            try:
                dictionary[tline.split()[0]]=float(tline.split()[1])
            except ValueError:
                lenth=len(tline.split())
                dictionary[tline.split()[0]]=' '.join(tline.split()[1:lenth])
            '''
            合并`tline.split()[1:lenth]`中的所有字符串，可以使用Python的`join()`方法。这个方法会将一个列表或元组中的所有元素连接成一个字符串，元素之间用指定的分隔符隔开。如果不提供分隔符，默认使用空字符串。
            以下是如何使用`join()`方法合并字符串的示例：
            ```python
            tline = "key value1 value2 value3"
            # 分割字符串
            parts = tline.split()
            # 计算长度并获取除第一个元素之外的所有元素
            lenth = len(parts)
            values = parts[1:lenth]
            # 使用空字符串作为分隔符合并元素
            merged_string = ''.join(values)
            # 将合并后的字符串赋值给字典中的键
            dictionary[parts[0]] = merged_string
            ```
            在这个例子中，`tline.split()`将字符串`tline`分割成一个列表`parts`。然后，我们使用切片`parts[1:lenth]`获取除键之外的所有值。最后，使用`''.join(values)`将这些值合并成一个单独的字符串，并将其存储在字典`dictionary`中，其中`parts[0]`是键。
            如果你想要使用特定的分隔符（例如逗号加空格`", "`），你可以这样写：
            ```python
            merged_string = ", ".join(values)
            ```
            这将把`values`列表中的元素用逗号和空格连接起来。

            '''
    return dictionary

def cfdReadAllDictionaries(filePath):
    """Returns all dictionary entries inside an OpenFOAM file.   
    This function does not directly mimic any function in uFVM. It was added to 
    make accessing a file's dictionary keywords and associated values easily
    within one function. The user can then extract the 'raw' values by 
    navigating the returned dictionary.    
    Attributes:
       filePath (str): path to file to read.
    Example usage: 
        Region = cfdReadAllDictionaries(filePath)
    这段Python代码定义了一个名为`cfdReadAllDictionaries`的函数，它用于读取OpenFOAM文件中的所有字典条目。函数的目的是方便地访问文件中字典关键字及其关联的值。下面是对这段代码的详细解释：

    1. **函数定义和文档字符串**：
    - `def cfdReadAllDictionaries(filePath):` 定义了函数，接收一个参数`filePath`，表示要读取的文件路径。
    - 函数的文档字符串（docstring）说明了函数的功能、属性、使用示例和目的。

    2. **异常处理**：
    - `try...except` 结构用于处理可能发生的异常，特别是文件未找到的情况。

    3. **文件操作**：
    - 使用`with open(filePath, "r") as fpid:`以只读模式打开指定路径的文件，并将其文件对象赋值给`fpid`。

    4. **初始化变量**：
    - `isDictionary` 和 `isSubDictionary` 布尔变量，用于跟踪是否正在解析字典或子字典。
    - `dictionaries` 字典，用于存储解析出的字典数据。

    5. **逐行读取文件**：
    - `for line, tline in enumerate(fpid):` 遍历文件的每一行，并使用`enumerate`获取行号。

    6. **跳过空行和注释**：
    - 使用`cfdSkipEmptyLines`和`cfdSkipMacroComments`函数跳过空行和宏定义注释。

    7. **字典开始和结束的标记**：
    - 根据行中是否包含`"{"`和`"}"`来确定字典和子字典的开始和结束。

    8. **处理字典元素**：
    - 根据当前的解析状态（是否在字典中、是否在子字典中等），将行内容分割并添加到`dictionaries`字典中。
    - 特别处理了包含方括号的字典元素（例如`'dimensions [0 1 -1 0 0 0 0]'`），移除方括号并分割。

    9. **返回结果**：
    - 函数返回填充好的`dictionaries`字典，其中包含了文件中所有字典的条目。

    10. **异常处理**：
        - 如果文件未找到，捕获`FileNotFoundError`异常，并打印警告信息。

    总结来说，`cfdReadAllDictionaries`函数通过读取指定文件路径的OpenFOAM文件，解析文件中的字典结构，并将这些结构存储在一个嵌套字典中返回。它能够处理主字典和子字典，并能够跳过注释和空行。如果文件不存在，它会打印一条警告信息。这个函数可以用于快速访问和分析OpenFOAM文件中的字典数据。
        
    """    
    try:
        with open(filePath,"r") as fpid:
        
            isDictionary=False
            isSubDictionary=False
            dictionaries={}
            
            for line, tline in enumerate(fpid):
                
                if not cfdSkipEmptyLines(tline):
                    continue
                
                if not cfdSkipMacroComments(tline):
                    continue            
                
                if "{" in tline and isDictionary == False:
                    isDictionary=True
                    continue
        
                if "{" in tline and isDictionary == True:
                    isSubDictionary=True
                    continue
        
                if "}" in tline and isSubDictionary == True:
                    isSubDictionary = False
                    continue
        
                if "}" in tline and isDictionary == True:
                    isDictionary = False
                    continue
                
                tline = tline.replace(";", "")
        
                #read one line dictionaries elements (e.g. 'dimensions [0 1 -1 0 0 0 0]')
                if len(tline.split()) > 1 and isDictionary == False:
                    tline = tline.replace("[", "")
                    tline = tline.replace("]", "")
                    dictionaries[tline.split()[0]]=tline.split()[1:]
                    continue
                
                #read dictionaries    
                if len(tline.split()) == 1 and isDictionary == False:
                    
                    currentDictionaryName=tline.split()[0]
                    dictionaries[currentDictionaryName]={}
                    continue
        
                if len(tline.split()) == 1 and isDictionary == True:
                    
                    currentSubDictionaryName=tline.split()[0]
                    dictionaries[currentDictionaryName][currentSubDictionaryName]={}
                    continue
        
                if len(tline.split()) > 1 and isDictionary == True and isSubDictionary == False:
        
                    dictionaries[currentDictionaryName][tline.split()[0]]=tline.split()[1:]
                    continue
               
                if len(tline.split()) > 1 and isSubDictionary == True:
        
                    dictionaries[currentDictionaryName][currentSubDictionaryName][tline.split()[0]]=tline.split()[1:]
                    continue     
            
            return dictionaries
        
    except FileNotFoundError:
            
        print('Warning: %s file is not found!!!' % os.path.split(filePath)[1])


def cfdGetKeyValue(key, valueType, fileID):

    """Returns the value of the 'key' entry in an OpenFOAM file.
    Attributes:       
       key (str): keyword to look for in line.
       valueType (str):
       fileID (str): path to initial file to look through. 
    Example usage:
        values = cfdGetKeyValue(key, valueType, fileID)
    DO:
        Add functionality for valueTypes 'scalar', 'cfdLabelList' and
        'cfdScalarList'. We can add these as we encounter a need for them. 
    这段Python代码定义了一个名为`cfdGetKeyValue`的函数，用于从OpenFOAM文件中提取特定关键字（`key`）的值。下面是对这段代码的详细解释：

    1. **函数定义和文档字符串**：
    - `def cfdGetKeyValue(key, valueType, fileID):` 定义了函数，接收三个参数：`key`是要在文件中查找的关键字字符串，`valueType`是值的类型（尽管在当前代码中这个参数似乎未使用），`fileID`是要读取的文件路径。
    - 函数的文档字符串说明了函数的功能、属性和使用示例，并指出了未来可能添加的功能（TODO部分）。

    2. **文件操作**：
    - 使用`with open(fileID, "r") as fpid:`以只读模式打开指定路径的文件，并将其文件对象赋值给`fpid`。

    3. **逐行读取文件**：
    - `for linecount, tline in enumerate(fpid):` 遍历文件的每一行，并使用`enumerate`获取行号和行内容。

    4. **跳过空行**：
    - 使用`cfdSkipEmptyLines`函数跳过空行。

    5. **查找关键字**：
    - 如果当前行`tline`包含关键字`key`，则进行处理。

    6. **清理行内容**：
    - 移除行中的分号、方括号、圆括号，以清理可能的注释或列表标记。

    7. **分割行内容**：
    - 使用`split()`方法将清理后的行内容分割成单词列表`splittedTline`。
        
    8. **检测分布类型**：
    - 检查分割后的行内容中是否包含`'uniform'`或`'nonuniform'`，以确定值的分布类型，并将其存储在`distribution`变量中。

    9. **提取值**：
    - 初始化一个空列表`value`，用于存储提取的数值。
    - 遍历`splittedTline`列表，尝试将每个条目转换为浮点数并添加到`value`列表中。如果转换失败（例如，条目不是数字），则忽略该条目。

    10. **返回结果**：
        - 函数返回一个列表，包含关键字`key`、分布类型`distribution`和提取的值列表`value`。

    注意，尽管文档字符串中提到了`valueType`参数，但当前代码实现中并未使用这个参数来影响函数的行为。此外，TODO部分提到了未来可能添加对`'scalar'`、`'cfdLabelList'`和`'cfdScalarList'`类型的支持，但这些功能在当前代码中尚未实现。

    函数的返回值是一个列表，其中包含三个元素：关键字、分布类型和值列表。这种设计允许函数返回关于关键字的详细信息，而不仅仅是值本身。这在处理OpenFOAM文件时可能非常有用，因为这些文件可能包含不同类型的数据和分布信息。
    """
    
    with open(fileID,"r") as fpid:

        for linecount, tline in enumerate(fpid):
            
            if not cfdSkipEmptyLines(tline):
                continue
            
            if key in tline:
                
                tline=tline.replace(";","")
                tline=tline.replace("[","")
                tline=tline.replace("]","")
                tline=tline.replace("(","")
                tline=tline.replace(")","")

                splittedTline=tline.split()
                splittedTline.remove(key)
                
                print(key,':',splittedTline)
                
                if 'uniform' in splittedTline:
                    
                    distribution='uniform'
                    
                elif 'nonuniform' in splittedTline:
                    distribution='nonuniform'
                    
                else:
                    distribution = None
                    
                value=[]
                    
                for iEntry in splittedTline:
                    
                    try:
                        value.append(float(iEntry))
                    except ValueError:
                        pass
                        
    return [key, distribution, value]

def contains_term(equation, term):
    # 使用正则表达式分解字符串
    parts = term_split(equation)
    # 检查分解后的部分是否包含特定的子字符串
    return term in parts

def term_split(equation):
    # 使用正则表达式分解字符串
    parts = re.split(r'[(),]', equation)
    # 移除空字符串
    parts = [part for part in parts if part.strip()]
    return  parts

def remove_terms(parts, terms_to_remove):
    # 删除特定项
    return [part for part in parts if part not in terms_to_remove]

def cfdReadUniformVolVectorFieldValue(volVectorFieldEntry):

    """Returns [u,v,w] type list from a 'value uniform (u v w)' dictionary entry. 
    
    Basically strips off '(' and ')' and returns a python list object, e.g. 
    [0, 1.2, 5]
    
    Attributes:
        
       volVectorFieldEntry (list): list containing ['uniform', '(u', 'v','w)']
       
    Example usage:
        
        Region = cfdReadUniformVolVectorFieldValue(volVectorFieldEntry)
    这段代码定义了一个函数`cfdReadUniformVolVectorFieldValue`，它从OpenFOAM字典条目中提取并解析表示均匀体积矢量场的值。这个函数特别用于处理形如`'value uniform (u v w)'`的条目，其中`u`、`v`和`w`是向量的三个分量。下面是对这段代码的详细解释：

    1. **函数定义和文档字符串**：
    - 函数接收一个参数`volVectorFieldEntry`，这是一个列表，包含了字典条目的组成部分，例如`['uniform', '(u', 'v', 'w)']`。

    2. **初始化列表和变量**：
    - `vector=[]`：初始化一个空列表`vector`，用来存储解析后的数值。
    - `uniform='Notuniform'`：初始化一个变量`uniform`并设置为`'Notuniform'`，用来标记向量场是否是均匀分布的。如果找到`'uniform'`关键字，则将其更改为`'uniform'`。

    3. **遍历输入列表**：
    - 使用`for item in volVectorFieldEntry:`遍历输入的列表。

    4. **处理`'uniform'`关键字**：
    - 如果当前项是`'uniform'`，则将`uniform`变量设置为`'uniform'`，并使用`continue`跳过当前循环的剩余部分。

    5. **清理和转换数值**：
    - 对于列表中的每个项（除了`'uniform'`），使用`replace`方法移除可能存在的圆括号`"("`和`")"`。
    - 将清理后的字符串项转换为浮点数，并使用`append`方法添加到`vector`列表中。

    6. **返回结果**：
    - 函数返回一个包含两个元素的元组。第一个元素是字符串`'uniform'`或`'Notuniform'`，表示向量场的分布类型；第二个元素是解析后的浮点数列表`vector`，包含了向量的三个分量。

    函数的使用示例在文档字符串中给出，但示例中的变量名`Region`可能是一个占位符，实际使用时应该使用更有意义的变量名来接收返回值。例如：

    ```python
    distribution, vector_values = cfdReadUniformVolVectorFieldValue(volVectorFieldEntry)
    ```

    这样，`distribution`将是一个字符串，表示向量场是均匀分布还是非均匀分布；`vector_values`将是包含实际向量分量的列表，例如`[0, 1.2, 5]`。
            
    """    
    vector=[]
    uniform='Notuniform'
    
    for item in volVectorFieldEntry:
        
        if item == 'uniform':
            uniform='uniform'
            continue
    
        item=item.replace("(","")
        item=item.replace(")","")
        
        vector.append(float(item))
        
    return uniform, vector

def cfdInitDirectories(caseDirectoryPath, *args, **kargs):
    '''
    这段Python代码定义了一个名为`cfdInitDirectories`的方法，它属于某个类的实例方法（由`self`参数推断）。这个方法的目的是初始化CFD（计算流体动力学）案例的目录结构，并准备用于存储收敛性数据的文件。下面是对这段代码的详细解释：

    1. **方法定义**：
    - `def cfdInitDirectories(self, *args, **kargs):` 定义了一个方法，它接收不定数量的位置参数（`*args`）和关键字参数（`**kargs`），以及一个隐含的`self`参数，表示类的实例。

    2. **获取当前工作目录**：
    - `cwd = self.caseDirectoryPath` 获取类实例的`caseDirectoryPath`属性，这通常是CFD案例的根目录路径。

    3. **定义收敛性目录路径**：
    - `convergencePath = cwd+os.sep+'convergence'` 构造收敛性数据目录的路径。

    4. **检查并创建收敛性目录**：
    - 使用`os.path.exists(convergencePath)`检查该目录是否存在。
    - 如果目录不存在，使用`os.makedirs(convergencePath)`创建目录。

    5. **处理位置参数**：
    - 如果提供了位置参数（`if args:`），则执行以下操作：
        - 构造收敛性输出文件的路径。
        - 从`args`中获取第一个参数，这通常是一个方程的名称。
        - 打开（创建）文件用于写入，并写入表头，包括迭代次数、时间以及特定方程的初始残差。

    6. **处理没有位置参数的情况**：
    - 如果没有提供位置参数（`else:`），则执行以下操作：
        - 构造另一个收敛性输出文件的路径，这个文件可能用于存储更详细的收敛性数据。
        - 打开（创建）文件用于写入，并写入表头，包括迭代次数、时间和多个不同的残差初始化值。

    7. **写入文件并关闭**：
    - 在两种情况下，都使用`open(fileName, 'w')`以写入模式打开文件，使用`write`方法写入数据，然后使用`close`方法关闭文件。

    这段代码是CFD案例设置的一部分，用于初始化必要的文件和目录结构，以便在模拟过程中记录和监控收敛性。代码中使用`os.sep`来确保路径在不同操作系统中都能正确工作，`os.path.exists`和`os.makedirs`是Python标准库`os`模块提供的函数，用于文件和目录操作。
    '''
    cwd = caseDirectoryPath
    
    convergencePath = cwd+os.sep+'convergence'
    if not os.path.exists(convergencePath):
        os.makedirs(convergencePath)
    
    if args:
        fileName = cwd+os.sep+'convergence'+os.sep+'convergence.out'
        theEquationName = args[0]
        tFile = open(fileName,'w')
        tFile.write('%s\t%s\t%s' % ('noIter', 'Time[s]', theEquationName+'ResInit'))
        tFile.close()
      
    else:
        fileName = cwd+os.sep+'convergence'+os.sep+'convergenceUp.out'
        tFile = open(fileName,'w')
        tFile.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('noIter','Time[s]','UxResInit','UyResInit','UzResInit','pResInit','kResInit','epsilonResInit','omegaResInit','TResInit'))
        tFile.close()      

def cfdWriteOpenFoamParaViewDatavtk(Region):
    '''
    这段Python代码定义了一个名为`cfdWriteOpenFoamParaViewData`的方法，它属于某个类的实例方法（由`self`参数推断）。这个方法的目的是将CFD（计算流体动力学）模拟结果流场速度、压强输出为ParaView可读取的OpenFOAM格式文件。
    OpenFOAM data output for ParaView
    Write the velocity field
    Write the Pressure field
    '''
    # 创建输出目录
    output_dir = os.path.join(Region.caseDirectoryPath, 'VTK')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取当前时间步
    current_time = Region.time.currentTime
    output_file = os.path.join(output_dir, f'{current_time}.vtk')

    with open(output_file, 'w') as vtk_file:
        # 写入VTK文件头
        vtk_file.write('# vtk DataFile Version 3.0\n')
        vtk_file.write('CFD simulation data\n')
        vtk_file.write('ASCII\n')
        vtk_file.write('DATASET UNSTRUCTURED_GRID\n')

        # 写入点数据
        points = Region.mesh.nodeCentroids
        vtk_file.write(f'POINTS {len(points)} float\n')
        for point in points:
            vtk_file.write(f'{point[0]} {point[1]} {point[2]}\n')

        # 写入单元数据
        cells = Region.mesh.faceNodes
        vtk_file.write(f'CELLS {len(cells)} {len(cells) * (len(cells[0]) + 1)}\n')
        for cell in cells:
            vtk_file.write(f'{len(cell)} {" ".join(map(str, cell))}\n')

        # 写入单元类型
        vtk_file.write(f'CELL_TYPES {len(cells)}\n')
        for _ in cells:
            vtk_file.write('10\n')  # VTK_TETRA

        # 写入速度场数据
        velocity_field = Region.fluid['U'].phi
        vtk_file.write(f'POINT_DATA {len(velocity_field)}\n')
        vtk_file.write('VECTORS velocity float\n')
        for velocity in velocity_field:
            vtk_file.write(f'{velocity[0]} {velocity[1]} {velocity[2]}\n')

        # 写入压强场数据
        pressure_field = Region.fluid['p'].phi
        vtk_file.write('SCALARS pressure float 1\n')
        vtk_file.write('LOOKUP_TABLE default\n')
        for pressure in pressure_field:
            vtk_file.write(f'{pressure}\n')

    print(f'VTK file written to {output_file}')

import os

def cfdWriteOpenFoamParaViewData(Region):
    '''
    这段Python代码定义了一个名为`cfdWriteOpenFoamParaViewData`的方法，它属于某个类的实例方法（由`self`参数推断）。这个方法的目的是将CFD（计算流体动力学）模拟结果流场速度、压强输出为ParaView可读取的OpenFOAM格式文件。
    OpenFOAM data output for ParaView
    Write the velocity field
    Write the Pressure field
    Write the Temperature field
    '''
    output_dir = os.path.join(Region.caseDirectoryPath, f"{Region.time.currentTime:.4g}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writelenth = Region.mesh.numberOfElements
    location = Region.time.currentTime
    for iTerm in Region.model.equations:
        field = Region.fluid[iTerm].phi[:writelenth,:]
        file_path = os.path.join(output_dir, iTerm)
        dimensions = str([int(dim) for dim in Region.fluid[iTerm].dimensions])
        boundary_conditions = generate_boundary_conditions(iTerm, Region)
        write_field(location, file_path, iTerm, field, Region.fluid[iTerm].type, dimensions, boundary_conditions)
        # if iTerm=='U':
        #     # 写入速度场数据
        #     velocity_field = Region.fluid[iTerm].phi[:writelenth,:]
        #     velocity_file_path = os.path.join(output_dir, iTerm)
        #     velocity_dimensions = str([int(dim) for dim in Region.fluid[iTerm].dimensions])
        #     velocity_boundary_conditions = generate_boundary_conditions(iTerm, Region)
        #     write_field(location, velocity_file_path, iTerm, velocity_field, Region.fluid[iTerm].type, velocity_dimensions, velocity_boundary_conditions)
        # elif iTerm=='p':
        #     # 写入压强场数据
        #     pressure_field = Region.fluid['p'].phi[:writelenth,:]
        #     pressure_file_path = os.path.join(output_dir, 'p')
        #     pressure_dimensions = "[0 2 -2 0 0 0 0]"
        #     pressure_boundary_conditions = generate_boundary_conditions('p', Region)
        #     write_field(location, pressure_file_path, 'p', pressure_field, 'volScalarField', pressure_dimensions, pressure_boundary_conditions)
        # elif iTerm=='T':
        #     # 写入温度场数据
        #     temperature_field = Region.fluid['T'].phi[:writelenth,:]
        #     temperature_file_path = os.path.join(output_dir, 'T')
        #     temperature_dimensions = "[0 0 0 1 0 0 0]"
        #     temperature_boundary_conditions = generate_boundary_conditions('T', Region)
        #     write_field(location, temperature_file_path, 'T', temperature_field, 'volScalarField', temperature_dimensions, temperature_boundary_conditions)

    print(f'OpenFOAM files written to {output_dir}')
    
# 动态生成边界条件
def generate_boundary_conditions(field_name, Region):
    boundary_conditions = {}
    for boundary in Region.fluid[field_name].boundaryPatchRef:
        boundary_conditions[boundary] = Region.fluid[field_name].boundaryPatchRef[boundary]
    return boundary_conditions


def write_field(location, file_path, field_name, field_data, field_type, dimensions, boundary_conditions):
    with open(file_path, 'w') as file:
        file.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        file.write("| =========                 |                                                 |\n")
        file.write("| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n")
        file.write("|  \\    /   O peration     | Version:  5.x                                   |\n")
        file.write("|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n")
        file.write("|    \\/     M anipulation  |                                                 |\n")
        file.write("\\*---------------------------------------------------------------------------*/\n")
        file.write("FoamFile\n")
        file.write("{\n")
        file.write("    version     2.0;\n")
        file.write("    format      ascii;\n")
        file.write(f"    class       {field_type};\n")
        file.write(f"    location    \"{location}\";\n")
        file.write(f"    object      {field_name};\n")
        file.write("}\n")
        file.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
        file.write(f"dimensions      {dimensions};\n\n")
        file.write(f"internalField   nonuniform List<{field_type.split('vol')[1].split('Field')[0].lower()}> \n")
        file.write(f"{len(field_data)}\n")
        file.write("(\n")
        for value in field_data:
            if len(value) == 3:
                file.write(f"({value[0]} {value[1]} {value[2]})\n")
            else:
                file.write(f"{value[0]}\n")
        file.write(")\n;\n\n")
        file.write("boundaryField\n")
        file.write("{\n")
        for boundary, condition in boundary_conditions.items():
            file.write(f"    {boundary}\n")
            file.write("    {\n")
            for key, val in condition.items():
                if key == "value" :
                    if len(val) == 3:
                        file.write(f"        {key}           uniform ({val[0]} {val[1]} {val[2]});\n")
                    elif len(val) == 1:
                        file.write(f"        {key}           uniform {val[0]};\n")
                elif key =='valueType':
                    continue
                else:
                    file.write(f"        {key}            {val};\n")
            file.write("    }\n")
        file.write("}\n")
        file.write("\n// ************************************************************************* //\n")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            