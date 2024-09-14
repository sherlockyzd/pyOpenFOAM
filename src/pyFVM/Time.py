import time as timer

class Time():
    """Manages simulation's time related properties
    这段Python代码定义了一个名为 `Time` 的类，用于管理模拟的时间相关属性，特别是在执行瞬态（即随时间变化的）模拟时。以下是对类的构造器和方法的详细解释：
    1. **类定义**：
    - `class Time():` 定义了一个名为 `Time` 的类，用于跟踪和更新模拟时间。

    2. **构造器**：
    - `def __init__(self, Region):` 构造器接收一个参数 `Region`，它是一个包含模拟区域信息的对象。

    3. **时间初始化**：
    - `self.tic = timer.time()`：记录模拟开始时的时间戳。
    - `self.startTime`：根据 `Region` 对象中的控制字典（`controlDict`）设置模拟的起始时间。
    - `self.currentTime`：初始化为起始时间，并将在模拟过程中更新。

    4. **时间设置逻辑**：
    - 根据 `controlDict['startFrom']` 的值，确定模拟的起始时间：
        - `'startTime'`：使用 `controlDict` 中指定的起始时间。
        - `'latestTime'`：使用 `Region.timeSteps` 中的最大值作为当前时间，即从最新的时间步开始。
        - `'firstTime'`：使用 `Region.timeSteps` 中的最小值作为当前时间，即从最早的时间步开始。

    5. **打印起始时间**：
    - `print('Start time: %.2f' % self.currentTime)`：打印模拟的起始时间。

    6. **结束时间设置**：
    - `self.endTime`：从 `controlDict` 中获取模拟的结束时间。

    7. **更新运行时间方法**：
    - `def cfdUpdateRunTime(self):` 更新模拟的当前运行时间和CPU时间。

    8. **打印当前时间方法**：
    - `def cfdPrintCurrentTime(self):` 打印模拟的当前时间。

    9. **执行瞬态循环方法**：
    - `def cfdDoTransientLoop(self):` 检查模拟的运行时间是否已达到结束时间，并返回相应的布尔值。

    ### 注意事项：
    - 类使用了 `time` 模块，并将其重命名为 `timer`，以便调用 `time()` 函数获取当前时间戳。
    - `Region` 对象预期包含 `dictionaries.controlDict` 和 `timeSteps` 属性，这些属性包含控制模拟的配置信息和时间步目录列表。
    - `self.cpuTime` 用于跟踪瞬态循环开始以来的CPU时间。
    `Time` 类是模拟过程中时间管理的关键组件，它提供了一种机制来跟踪和更新模拟时间，确保模拟按照预定的时间步长运行，并在达到设定的结束时间时停止。
    """
    def __init__(self,Region):
        
        ##Time at beginning of transient loop
        self.tic=timer.time()
        ##Instance of simulation's region class
        # self.region=Region
        if Region.dictionaries.controlDict['startFrom']=='startTime':
            
                ##Specified start time of the simulation
                self.startTime=float(Region.dictionaries.controlDict['startTime'])
                
                ## The current time elapsed since the start of the simulation
                self.currentTime = self.startTime
    
        elif Region.dictionaries.controlDict['startFrom']=='latestTime':
                self.currentTime = float(max(Region.timeSteps))
    
        elif Region.dictionaries.controlDict['startFrom']=='firstTime':
                self.currentTime = float(min(Region.timeSteps))
    
        print('Start time: %.2f' % self.currentTime)
        
        ##Specified end time of the simulation
        self.endTime = float(Region.dictionaries.controlDict['endTime'])
        self.deltaT  = Region.dictionaries.controlDict['deltaT']
        self.writeInterval=int(Region.dictionaries.controlDict['writeInterval'])
    
    def cfdUpdateRunTime(self):
        """Increments the simulation's runTime, updates cpu time
        """
        self.currentTime = self.currentTime + self.deltaT
        ## The elapsed cpu clock time since starting the transient loop
        self.cpuTime=timer.time()-self.tic
        print('cpu time: %0.4f [s]\n' % self.cpuTime)
        
    
    def cfdPrintCurrentTime(self):
        """Prints the current runTime"""
        print('\n\n Time: %0.4f [s]\n' % self.currentTime)
        
    def cfdDoTransientLoop(self):
        """
        Checks too see if simulation runTime has reached the simulation's end time
        """
        if self.currentTime < self.endTime:
            return True
        else:
            return False

    def cfdDoWriteTime(self):
        """
        检查每隔writeInterval时间步长是否需要写入数据
        """
        if int(self.currentTime/self.deltaT) % int(self.writeInterval)== 0:
            return True
        else:
            return False