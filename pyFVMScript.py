
import os
# import sys
import pyFVM.Region as Region
import time as time


# print(sys.path[1])
# sys.path.insert(1, '.')
start=time.time()
cfdCase=Region.Region(os.getcwd())
cfdCase.RunCase()
# region_vars=vars(cfdCase)
print(time.time()-start)
