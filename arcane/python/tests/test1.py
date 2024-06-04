import ArcanePython
import Arcane

print(Arcane.RealArray)
zz = Arcane.RealArray()
zz.Resize(3)

print(zz.Size)
assert zz.Size==3, "Bad Size"

zz[0] = 5.2
print(zz[0])
assert zz[0]==5.2, "Bad value zz[0]"
