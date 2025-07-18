import ArcanePython
import Arcane
import sys
import numpy
import ctypes as C
from Arcane import ArcaneLauncher

import os
gvar0 = 1

true_py_context1 = ArcanePython.SubDomainContext()
def my_test_func(sd_context):
    print("HELLO context=",sd_context,flush=True)
    true_py_context = ArcanePython.SubDomainContext()
    global gvar0
    gvar0 = gvar0 + 1
    mesh0 = sd_context.DefaultMesh
    node_coord = sd_context.GetVariable(mesh0,"NodeCoord")
    nd_node_coord = node_coord.GetNDArray()
    pressure = sd_context.GetVariable(mesh0,"Pressure")
    nd_pressure = pressure.GetNDArray()
    print("NodeCoord shape=",nd_node_coord.shape," v=",nd_node_coord,flush=True)
    nd_node_coord[0][0] = 0.01
    print("Pressure shape=",nd_pressure.shape," v=",nd_pressure,flush=True)
    nd_pressure[0] = 1.2
    print("Pressure2=",nd_pressure)
    print("--Endcallback",flush=True)

args = Arcane.CommandLineArguments.Create(sys.argv)
ArcaneLauncher.Init(args)

print("MY_TEST_BEGIN",flush=True)
print("MY_TEST_PATH=",__file__,flush=True)

ArcanePython.func1()
ArcanePython._utils.func0()

app_info = ArcaneLauncher.ApplicationInfo;
app_info.SetCodeName("ArcaneTest")
app_build_info = ArcaneLauncher.ApplicationBuildInfo;
app_build_info.AddDynamicLibrary("arcane_tests_lib")
r = ArcaneLauncher.Run();
print("MY_TEST1_FINAL gvar0",gvar0)
exit(r)
