import Arcane

print("Hello (script2) from python")

zz = Arcane.RealArray()
nb_call = 1

zz.Resize(3)
print(zz.Size)
assert zz.Size==3, "Bad Size"

def func1():
    global nb_call
    nb_call += 5
    print("This is func1 !",nb_call)
    print(Arcane.RealArray)
    zz.Resize(nb_call)
    print("ZZ Size",zz.Size)
    zz[nb_call-1] = nb_call
    print(zz[nb_call-1])
    assert zz[nb_call-1]==nb_call, "Bad value zz[0]"
    print("End of test 'func1'", flush=True)
print("NbCall=",nb_call)
print("End of init test", flush=True)
