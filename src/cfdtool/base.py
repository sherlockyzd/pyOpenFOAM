# base.py

# from dataclasses import dataclass, fields,field
from config import ENABLE_DIMENSION_CHECK
from cfdtool.quantities import Quantity

# @dataclass
class DimensionChecked:
    '''
    DimensionChecked 使用了 @dataclass 装饰器，但没有显式定义 __init__ 方法。
    @dataclass 会自动生成一个 __init__ 方法，该方法初始化所有定义的字段。
    DimensionChecked 定义了 __post_init__ 方法，用于在初始化后执行量纲检查。
    '''
    # 定义一个内部字典，用于存储字段名到预期量纲的映射
    # expected_dimensions: dict = field(default_factory=dict, init=False, repr=False)
    # _class_field_map: dict = field(init=False, default_factory=dict, repr=False)
    def __init__(self):
        # 如果 `expected_dimensions` 尚未设置，则初始化为空字典
        if not hasattr(self, 'expected_dimensions'):
            raise ValueError("No expected_dimensions")
         # 如果启用了量纲检查，进行量纲检查
        if ENABLE_DIMENSION_CHECK:
            self.check_dimensions()

    def check_dimensions(self):
        """
        在初始化之后进行量纲检查。
        仅检查 expected_dimensions 中指定的字段。
        """
        for name, expected_dim in self.expected_dimensions.items():
            try:
                value = getattr(self, name)
            except AttributeError:
                raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")
            # print(f"Checking dimension for '{name}': expected {expected_dim}, got {value.dimension}")  # 调试信息

            if isinstance(value, Quantity) and expected_dim is not None:
                if value.dimension != expected_dim:
                    raise ValueError(
                        f"量纲不匹配在字段 '{name}': 预期 {expected_dim}, 但得到 {value.dimension}"
                    )

    def __setattr__(self, name, value):
        """
        在属性赋值时进行量纲检查。
        """
        if name == 'expected_dimensions':
            # 不对 expected_dimensions 本身进行量纲检查
            super().__setattr__(name, value)
            return
        
        # 如果启用了量纲检查，进行量纲检查
        if ENABLE_DIMENSION_CHECK and name in self.__dict__.get('expected_dimensions', {}):
            expected_dim = self.expected_dimensions[name]
            if isinstance(value, Quantity) and expected_dim is not None:
                if value.dimension != expected_dim:
                    raise ValueError(
                        f"赋值时量纲不匹配 '{name}': 预期 {expected_dim}, 但得到 {value.dimension}"
                    )
        super().__setattr__(name, value)

