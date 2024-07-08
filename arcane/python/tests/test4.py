import ArcanePython
import Arcane

from Arcane import ArcaneLauncher

import os

def my_callback(ctx):
    sd = ctx.CreateSequentialSubDomain();
    mrm = Arcane.MeshReaderMng(sd);
    mesh_file_name = os.path.join(os.path.dirname(__file__),"../../maillages/tube5x5x100.vtk")
    mesh = mrm.ReadMesh("Mesh1",mesh_file_name);
    print("NB_CELL=",mesh.NbCell())
    return 0

args = Arcane.CommandLineArguments.Create({})
ArcaneLauncher.Init(args)

callback1 = ArcaneLauncher.DirectExecutionContextDelegate(my_callback)
ArcaneLauncher.Run(callback1)
