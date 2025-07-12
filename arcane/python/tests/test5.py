import ArcanePython
import Arcane
import numpy as np

from Arcane import ArcaneLauncher

import os

def my_callback(ctx):
    sd = ctx.CreateSequentialSubDomain();
    mrm = Arcane.MeshReaderMng(sd);
    mesh_file_name = os.path.join(os.path.dirname(__file__),"../../maillages/tube5x5x100.vtk")
    mesh = mrm.ReadMesh("Mesh1",mesh_file_name);
    print("NB_CELL=",mesh.NbCell())
    # Create a variable named 'Density'
    var_density = Arcane.VariableCellReal(Arcane.VariableBuildInfo(mesh,"Density"))
    print("VarDensity=",var_density);
    print("VarDensityName=",var_density.Variable().FullName());
    # Create a context to convert the variable to an NumPy Array
    sd_context = Arcane.Python.SubDomainContext(sd)
    print("Context=",sd_context);
    density = sd_context.GetVariable(mesh, "Density")
    print("Density=",density);
    py_density = density.GetNDArray()
    print("PyDensity=",py_density);
    return 0

args = Arcane.CommandLineArguments.Create({})
ArcaneLauncher.Init(args)

callback1 = ArcaneLauncher.DirectExecutionContextDelegate(my_callback)
ArcaneLauncher.Run(callback1)
