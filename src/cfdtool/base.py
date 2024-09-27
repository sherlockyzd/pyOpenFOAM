# base.py

from dataclasses import dataclass, fields,field
from config import ENABLE_DIMENSION_CHECK
from cfdtool.quantities import Quantity
# from cfdtool.dimensions import Dimension

@dataclass
class DimensionChecked:
    # 定义一个内部字典，用于存储字段名到预期量纲的映射
    expected_dimensions: dict = field(default_factory=dict, init=False, repr=False)
    # _class_field_map: dict = field(init=False, default_factory=dict, repr=False)

    expected_dimensions: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        # 确保 expected_dimensions 已经正确设置
        if not hasattr(self, 'expected_dimensions'):
            self.expected_dimensions = {}
        
        # 量纲检查机制在 __post_init__ 后执行
        if ENABLE_DIMENSION_CHECK:
            self.check_dimensions()

    def check_dimensions(self):
        """
        在初始化之后进行量纲检查。
        """
        for f in fields(self):
            value = getattr(self, f.name)
            expected_dim = self.expected_dimensions.get(f.name, None)
            if isinstance(value, Quantity) and expected_dim is not None:
                if value.dimension != expected_dim:
                    raise ValueError(
                        f"Dimension mismatch in field '{f.name}': expected {expected_dim}, got {value.dimension}"
                    )

    def __setattr__(self, name, value):
        """
        在属性赋值时进行量纲检查。
        """
        if ENABLE_DIMENSION_CHECK and hasattr(self, 'expected_dimensions') and name in self.expected_dimensions:
            expected_dim = self.expected_dimensions[name]
            if isinstance(value, Quantity) and expected_dim is not None:
                if value.dimension != expected_dim:
                    raise ValueError(
                        f"Dimension mismatch when assigning to '{name}': expected {expected_dim}, got {value.dimension}"
                    )
        super().__setattr__(name, value)

