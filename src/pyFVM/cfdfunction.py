import cfdtool.IO as io
import numpy as np
import cfdtool.Interpolate as interp
import cfdtool.Math as mth
# from cfdtool.quantities import Quantity as Q_
# import cfdtool.dimensions as dm


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
    # U_f=interp.cfdInterpolateFromElementsToFaces(Region,'linear',Region.fluid['U'].phi)
    # rho_f=interp.cfdInterpolateFromElementsToFaces(Region,'linear',Region.fluid['rho'].phi)
    #calculate mass flux through faces, 必须写成二维数组的形式，便于后续与U的数组比较运算!
    # 计算通量 Sf ⋅ U_f，得到每个面的流量，形状为 (nFaces, 1)
    # 计算质量流量 phi = rho_f * flux，形状为 (nFaces, 1)
    Region.fluid['mdot_f'].phi = cal_flux(Region.fluid['U'].phi, Region.fluid['rho'].phi, Region) # 形状: (nFaces, 1)
    
    # 检查 phi 是否包含非有限值（如 NaN 或无穷大）
    if not np.all(np.isfinite(Region.fluid['mdot_f'].phi.value)):
        io.cfdError('计算得到的质量流量 phi 包含非有限值') 

def cal_flux(U_phi,rho_phi,Region):
    U_f=interp.cfdInterpolateFromElementsToFaces(Region,'linear',U_phi)
    rho_f=interp.cfdInterpolateFromElementsToFaces(Region,'linear',rho_phi)
    Sf=Region.mesh.faceSf
    return rho_f*mth.cfdDot(Sf, U_f)[:, np.newaxis]