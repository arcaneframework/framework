import Arcane
import Arcane.Python
import numpy as np

print("Hello (script4) from python")


def context_func1(sd_context: Arcane.Python.SubDomainContext):
    mesh = sd_context.DefaultMesh
    density = sd_context.GetVariable(mesh, "Density")
    nd_a1: np.ndarray = density.GetNDArray()
    print(nd_a1)
    print("Shape", nd_a1.shape)
    nd_sum = np.sum(nd_a1)
    np_test = np.random.rand(40)
    print("NP_TEST=", np_test.shape, np_test)
    print("NP_TEST[25]=", np_test[25])
    print("V[25]", nd_a1[25])
    print("Sum", nd_sum)
    final_a = nd_a1 + 3.5
    print("Final_A", final_a)
    density.SetNDArray(final_a)
    print("End of test 'func1' for script4", flush=True)


print("End of init test", flush=True)
