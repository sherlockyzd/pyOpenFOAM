import numpy as np
import os
import matplotlib.pyplot as plt

def plotResidualHistory(Region):
    """
    绘制所有方程的残差随时间变化曲线（对数坐标）。
    假设 self.model.residuals 结构如下：
      {
        eqName: {
          'residuals': [...],    # 对 U：列表中每个分量各自为一个列表；对其它方程：一个列表
          'time': [...],         # 时间列表
          'iterations': [...]    # 迭代步数列表（可选）
        },
        ...,
        'sumRes': float           # 总残差（累计或最新值）
      }
    """
    res_dict = Region.model.residuals
    # 开启交互模式
    # if interactive:
    plt.ion()
    
    plt.figure("Residual History")
    plt.clf()  # 清除之前的图

    plt.figure(1)
    for eq, data in res_dict.items():
        if eq == 'sumRes':
            continue
        times = data['time']
        # 对 U 分量单独绘制
        if eq == 'U':
            for comp_idx, comp_res in enumerate(data['residuals']):
                plt.semilogy(times, comp_res, label=f'U[{comp_idx}]')
        else:
            plt.semilogy(times, data['residuals'], label=eq)

    # 可选：绘制累计总残差（非迭代序列，仅做参考）
    if 'sumRes' in res_dict:
        # 若 sumRes 为标量序列，则需与 times 对齐；否则跳过
        if isinstance(res_dict['sumRes'], (list, tuple)):
            plt.semilogy(times, res_dict['sumRes'], '--', label='sumRes')

    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.title('Residual History')
    plt.legend()
    plt.grid(True)
    plt.draw()
    # if interactive:
    plt.pause(0.001)  # 让图形界面刷新，但不阻塞

# 将方法绑定到你的 solver/region 对象，即可直接调用：
# solver.plotResidualHistory()


def cfdPlotRes(path,equations):
    # Read data from file
    data = np.loadtxt(path+os.sep+'convergence'+os.sep+'convergenceUp.out', skiprows=1)
    
    # Return if no data yet is stored
    if data.size == 0:
        return
    
    # Extract data columns
    iters = data[:, 0]
    currentTime = data[:, 1]
    UxResInit = data[:, 2]
    UyResInit = data[:, 3]
    UzResInit = data[:, 4]
    pResInit = data[:, 5]
    TResInit = data[:, 9]
    
    # Clear previous plot
    plt.clf()
    
    # Plot residuals for each equation
    legendStringArray = []
    # theEquationNames = model.equations
    
    for iEquation in equations:
        if iEquation.name == 'U':
            legendStringArray.append('Ux')
            legendStringArray.append('Uy')
            legendStringArray.append('Uz')
            
            plt.semilogy(iters, UxResInit, '-xr')
            plt.hold(True)
            plt.semilogy(iters, UyResInit, '-og')
            plt.hold(True)
            plt.semilogy(iters, UzResInit, '-+b')
            plt.hold(True)
        elif iEquation.name == 'p':
            legendStringArray.append('p-mass')
            plt.semilogy(iters, pResInit, '-<k')
        elif iEquation.name == 'T':
            legendStringArray.append('T')
            plt.semilogy(iters, TResInit, '->c')
    
    # Set plot labels and legend
    plt.xlabel('Global Iterations')
    plt.ylabel('Scaled RMS Residuals')
    plt.legend(legendStringArray)
    
    # Set plot grid and axis limits
    plt.grid(True)
    plt.axis('tight')
    plt.ylim(1e-6, 1e2)
    
    # Pause to show the plot
    plt.pause(0.01)