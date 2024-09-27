import cfdtool.IO as io
import numpy as np
import cfdtool.Interpolate as interp
import cfdtool.Math as mth
from cfdtool.quantities import Quantity as Q_
import cfdtool.dimensions as dm


def initializeMdotFromU(Region):
    """
    初始化面上的质量流量 phi，通过将速度 U 和密度 rho 从单元格插值到面上，并计算质量流量。
    
    参数：
    - Region: 包含流体区域信息的对象。
    
    过程：
    1. 使用线性插值方法将速度 U 和密度 rho 从单元格插值到面上。
    2. 计算每个面的质量流量 phi = rho_f * (Sf ⋅ U_f)。
    
    返回：
    - self.phi: 形状为 (numberOfFaces, numberOfComponents) 的质量流量数组。
    """
    U_f=interp.cfdinterpolateFromElementsToFaces(Region,'linear',Region.fluid['U'].phi.value)
    rho_f=interp.cfdinterpolateFromElementsToFaces(Region,'linear',Region.fluid['rho'].phi.value)
    Sf=Region.mesh.faceSf
    #calculate mass flux through faces, 必须写成二维数组的形式，便于后续与U的数组比较运算!
        # 确保插值结果的形状匹配
    if U_f.ndim != 2 or rho_f.ndim != 2:
        io.cfdError('插值后的 U_f 和 rho_f 必须是二维数组')
    
    if Sf.shape[0] != U_f.shape[0] or Sf.shape[0] != rho_f.shape[0]:
        io.cfdError('Sf、U_f 和 rho_f 的面数量不匹配')
    
    # 计算通量 Sf ⋅ U_f，得到每个面的流量，形状为 (nFaces, 1)
    flux =mth.cfdDot(Sf, U_f)[:, np.newaxis]  # 使用 np.einsum 进行高效的点积计算
    
    # 计算质量流量 phi = rho_f * flux，形状为 (nFaces, 1)
    Region.fluid['mdot_f'].phi = Q_(rho_f * flux,dm.flux_dim ) # 形状: (nFaces, 1)
    
    # 检查 phi 是否包含非有限值（如 NaN 或无穷大）
    if not np.all(np.isfinite(Region.fluid['mdot_f'].phi.value)):
        io.cfdError('计算得到的质量流量 phi 包含非有限值')            