import numpy as np
from cfdtool.quantities import Quantity as Q_

def cfdMag(valueVector):
    if isinstance(valueVector,Q_) :
        return Q_(cfdMag_np(valueVector.value),valueVector.dimension)
    elif isinstance(valueVector,np.ndarray):
        return cfdMag_np(valueVector)
    else:
        raise TypeError("valueVector must be a Quantity or a numpy array")

def cfdMag_np(valueVector):

    """Returns the magnitude of a vector or list of vectors
    
    Attributes:
    
       valueVector
    这段代码是一个Python程序片段，使用NumPy库来计算向量的点积和模（或长度）。下面是对这段代码的详细解释：
    1. `try`块：程序首先尝试执行`try`块中的代码。如果`try`块中的代码执行过程中没有出现异常，那么`except`块将被跳过。
    2. `iter(valueVector[0])`：这行代码尝试对`valueVector[0]`进行迭代。如果`valueVector[0]`是一个可迭代对象（例如列表或元组），则`iter`函数将返回一个迭代器。如果`valueVector[0]`不是一个可迭代对象，将引发`TypeError`异常。
    3. `result = []`：初始化一个空列表`result`，用来存储计算结果。
    4. `for iVector in valueVector:`：这个循环遍历`valueVector`中的每个元素。如果`valueVector`是一个列表或元组，每个`iVector`将是`valueVector`中的一个元素。
    5. `dotProduct = np.vdot(iVector, iVector)`：使用NumPy的`vdot`函数计算`iVector`和自身的点积。点积是两个向量的对应元素相乘后求和的结果。
    6. `magnitude = np.sqrt(dotProduct)`：计算点积的平方根，得到`iVector`的模（或长度）。
    7. `result.append(magnitude)`：将计算得到的模添加到结果列表`result`中。
    8. `except TypeError:`：如果`try`块中的代码抛出了`TypeError`异常（例如`valueVector[0]`不可迭代），则执行`except`块中的代码。
    9. `dotProduct = np.vdot(valueVector, valueVector)`：如果`valueVector`本身是一个向量，那么使用`vdot`函数计算`valueVector`和自身的点积。
    10. `magnitude = np.sqrt(dotProduct)`：计算点积的平方根，得到`valueVector`的模。
    11. `result = magnitude`：由于`valueVector`是一个单一向量，所以结果是一个单一数值，直接赋值给`result`。
    12. `return result`：函数返回`result`，它可能是一个包含多个模的列表，或者是一个单一的模数值。
    总结来说，这段代码的目的是计算一个向量列表中每个向量的模，或者如果只给定了一个向量，则计算该向量的模。如果输入是一个向量列表，它会返回一个包含每个向量模的列表；如果输入是一个单一向量，它会返回该向量的模。
    """
    if not isinstance(valueVector, np.ndarray):
        raise TypeError("输入参数必须是一个 NumPy 数组")
    if valueVector.ndim == 1:
        # 输入是单个向量，返回单一模数
        return np.linalg.norm(valueVector)
    elif valueVector.ndim == 2:
        # 输入是向量列表，返回每个向量的模
        return np.linalg.norm(valueVector, axis=1)
    else:
        raise ValueError("valueVector 必须是一维或二维的 NumPy 数组。")

def cfdDot(Sf, U_f):
    if isinstance(Sf,Q_) and isinstance(U_f,Q_):
            return Q_(cfdDot_np(Sf.value, U_f.value),Sf.dimension*U_f.dimension)
    
    elif isinstance(Sf,Q_) and isinstance(U_f,np.ndarray):
        return Q_(cfdDot_np(Sf.value, U_f), Sf.dimension)
    
    elif isinstance(Sf,np.ndarray) and isinstance(U_f,np.ndarray):
        return cfdDot_np(Sf, U_f)
    
    else:
        raise ValueError("输入参数必须是 Quantity 或 NumPy 数组。")

def cfdDot_np(Sf, U_f):
    """
    计算每个面的面积向量 Sf 与速度向量 U_f 的点积。
    
    参数：
    - Sf: 面面积向量数组，形状为 (..., dim)。
    - U_f: 面上的速度向量数组，形状为 (..., dim)。
    
    返回：
    - flux: 每个面的通量值数组，形状为 (...,)。
    
    支持一维、二维和三维向量。
    
    示例：
    >>> Sf = np.array([[1, 2, 3], [4, 5, 6]])
    >>> U_f = np.array([[7, 8, 9], [10, 11, 12]])
    >>> cfdDot(Sf, U_f)
    array([ 50, 154])
    """
    # 检查输入数组的形状是否匹配
    if Sf.shape != U_f.shape:
        raise ValueError(f"Shape mismatch: Sf.shape={Sf.shape} and U_f.shape={U_f.shape} must be the same.")
    return np.einsum('...i,...i->...', Sf, U_f)

def cfdUnit(vector):
    """
    这段Python代码定义了一个名为`cfdUnit`的函数，它接收一个名为`vector`的参数，该参数应该是一个NumPy数组。这个函数的目的是将输入的数组`vector`中的每个向量标准化，使其成为单位向量（即模为1的向量）。下面是对这段代码的详细解释：
    1. `def cfdUnit(vector):`：定义一个函数`cfdUnit`，它接收一个参数`vector`。
    2. `return vector/np.linalg.norm(vector,axis=1)[:,None]`：函数的返回语句执行了以下操作：
    - `np.linalg.norm(vector,axis=1)`：使用NumPy的`linalg.norm`函数计算`vector`中每个向量的模（长度）。参数`axis=1`指定了沿着第二个轴（即行方向）计算每个向量的模。
    - `[:,None]`：这个操作将上一步得到的模数组增加一个维度，使得它可以与`vector`进行广播（broadcasting）操作。`[:,None]`相当于`np.newaxis`，它在数组的形状中添加了一个维度，位置在第二个轴（列）上。
    - `vector/...`：将`vector`中的每个向量除以对应的模，从而得到单位向量。由于之前通过`[:,None]`增加了维度，所以这个除法操作可以沿着行进行广播，即每个行向量都除以其对应的模。
    总结来说，`cfdUnit`函数通过计算输入数组中每个向量的模，然后将每个向量除以其模，来返回一个包含单位向量的数组。这样，每个输出向量的模都是1，它们的方向与输入向量相同。
    举个例子，如果输入的`vector`是这样的二维数组：
    ```python
    vector = np.array([[4, 0], [0, 3]])
    ```
    那么`cfdUnit`函数将返回：
    ```python
    cfdUnit(vector) = np.array([[1, 0], [0, 1]])
    ```
    因为第一个向量的模是4，第二个向量的模是3，分别除以它们的模后，我们得到了单位向量。
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError("输入参数必须是一个 NumPy 数组")
    # 计算范数，并避免除以0
    epsilon = 1e-10  # 设置一个小的数，避免除以0
    norms = np.linalg.norm(vector, axis=1)
    norms[norms == 0] = epsilon  # 将0范数替换为epsilon

    # 进行除法操作，得到单位向量
    return vector / norms[:, np.newaxis]
    # return vector/np.linalg.norm(vector,axis=1)[:,None]

def cfdScalarList(*args):
        """
        Initializes a scalar list.
        
        Parameters:
        - *args: Variable length argument list.
        
        Returns:
        - the_scalar_list: A NumPy array initialized based on the input arguments.
        """
        
        # 如果没有提供参数或提供的参数为空，返回空列表
        if not args:
            return []
        
        # 如果提供了一个参数，假设这是一个长度 n，初始化为0的列表
        if len(args) == 1:
            n = args[0]
            the_scalar_list = np.zeros(n, dtype=float)
        
        # 如果提供了两个参数，假设这是长度 n 和一个值，用该值初始化列表
        elif len(args) == 2:
            n = args[0]
            value = args[1]
            the_scalar_list = np.full(n, value, dtype=float)
        
        return the_scalar_list

        # # 使用示例
        # # 初始化一个长度为5的列表，所有元素为0
        # scalar_list1 = cfd_scalar_list(5)

        # # 初始化一个长度为3的列表，所有元素为1.0
        # scalar_list2 = cfd_scalar_list(3, 1.0)

def cfdResidual(rc, method='norm'):
    if method == 'RMS':
        rc_res = np.sqrt(np.mean(rc ** 2))
    elif method == 'norm':
        rc_res = np.linalg.norm(rc)
    elif method == 'mean':
        rc_res = np.mean(np.abs(rc))
    elif method == 'max':
        rc_res = np.max(np.abs(rc))
    elif method =='sum':
        rc_res = np.sum(np.abs(rc))
    return rc_res