# base.py

from cfdtool.quantities import Quantity

class DimensionChecked:
    '''
    DimensionChecked 提供量纲元数据管理。
    
    子类（Field、Gradient）通过设置 self.expected_dimensions 字典
    来记录各属性的预期量纲，供 .dimension 属性查询。
    '''
    def __init__(self):
        if not hasattr(self, 'expected_dimensions'):
            raise ValueError("No expected_dimensions")
