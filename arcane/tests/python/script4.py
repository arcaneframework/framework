import Arcane
import numpy as np

print("Hello (script4) from python")

def context_func1(sd_context):
    nd_a1 = sd_context.GetNDArray("Density");
    print(nd_a1)
    print("Shape",nd_a1.shape)
    #nd_a1.reshape(1250)
    sum = np.sum(nd_a1)
    np_test = np.random.rand(40)
    print("NP_TEST=",np_test.shape,np_test)
    print("NP_TEST[25]=",np_test[25])
    print("V[25]",nd_a1[25])
    print("Sum",sum)
    sum2 = np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.float)
    print("Sum2",sum2)
    #sum3 = np.sum(nd_a1, dtype=np.float)
    #print("Sum2",sum3)
    print("End of test 'func1' for script4", flush=True)

print("End of init test", flush=True)
