import os
import sys
import time as time
# 获取当前脚本文件所在的目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
# Add the path to the pyFVM directory
src_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
print(f"src 路径: {src_path}")  # 打印路径，确保正确
sys.path.insert(0, src_path)
import pyFVM.Region as Rg

start=time.time()
cfdCase=Rg.Region(current_dir)
cfdCase.RunCase()
# region_vars=vars(cfdCase)
print(time.time()-start)
